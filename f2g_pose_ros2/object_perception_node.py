"""Explicit head/wrist perception, one optional shared F2G model, atomic snapshots."""
import copy
import json
import time
import uuid
from collections import deque

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Header, String
from core_r1_interfaces.msg import MultiViewSnapshot, ObservationRef, ObservationTarget, TargetCloud
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

from .object_geometry import estimate, fuse, resolve_modes, voxelize


def stamp_ns(header):
    return header.stamp.sec * 10**9 + header.stamp.nanosec


class ObjectPerceptionNode(Node):
    def __init__(self):
        super().__init__('r1_object_perception')
        defaults = dict(camera_mode='head_then_wrist', cloud_source='auto', head_f2g=True,
                        wrist_f2g=False, preload_models=False, target_label='cup', base_frame='base_link',
                        max_capture_age=180., max_pair_skew=.05,
                        standoff_dist=.50, elevation_deg=48., azimuth_deg=25.)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        get = lambda key: self.get_parameter(key).value
        self.mode, self.frame, self.label = get('camera_mode'), get('base_frame'), get('target_label')
        self.source = resolve_modes(self.mode, get('cloud_source'))
        self.enabled = {view: bool(get(view + '_f2g')) for view in ('head', 'wrist')}
        self.max_age, self.skew = get('max_capture_age'), get('max_pair_skew')
        if not np.isfinite([self.max_age, self.skew]).all() or min(self.max_age, self.skew) <= 0:
            raise ValueError('Capture age and RGB/depth pairing tolerance must be positive and finite')
        self.bridge, self.tf = CvBridge(), Buffer(cache_time=Duration(seconds=self.max_age + 30.))
        self.listener = TransformListener(self.tf, self)
        self.estimator = None
        active_views = ('head',) if self.mode == 'head' else ('wrist',) if self.mode == 'wrist' else ('head', 'wrist')
        if get('preload_models') and any(self.enabled[v] for v in active_views):
            self.init_estimator()
        self.session, self.generation = str(uuid.uuid4()), 0
        self.views, self.fusion_metrics = {}, None
        self.pending = {}
        self.buffers = {view: {kind: deque(maxlen=8 if kind == 'rgb' else 32) for kind in ('rgb', 'depth')}
                        for view in ('head', 'wrist')}
        self.info, self.depth_info, self.pinned, self.pin_parts, self.headers, self.packets = {}, {}, {}, {}, {}, {}
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.snapshot_pub = self.create_publisher(MultiViewSnapshot, '/perc/object/snapshot', latched)
        self.geometry_pub = self.create_publisher(String, '/perc/object/geometry', latched)
        self.status_pub = self.create_publisher(String, '/perc/object/status', latched)
        self.observation_pub = self.create_publisher(ObservationTarget, '/perc/head/observation_target', latched)
        self.view_publishers = {}
        self.observed_publishers, self.observed_views = {}, {}
        active = ('head',) if self.mode == 'head' else ('wrist',) if self.mode == 'wrist' else ('head', 'wrist')
        for view in (*active, 'fused'):
            self.view_publishers[view] = (
                self.create_publisher(PointCloud2, f'/perc/object/{view}/cloud', latched),
                self.create_publisher(PoseStamped, f'/perc/object/{view}/pose', latched),
                self.create_publisher(Vector3Stamped, f'/perc/object/{view}/dimensions', latched))
        for view in active:
            self.observed_publishers[view] = self.create_publisher(
                PointCloud2, f'/perc/object/{view}/observed_cloud', latched)
            camera = '/camera/head_d455' if view == 'head' else '/camera/left_d405'
            prefix = '/perc/head' if view == 'head' else '/perc/grasp'
            for kind, topic in (('rgb', camera + '/color/image_raw'),
                                ('depth', camera + ('/depth/image_rect_raw' if view == 'head' else '/aligned_depth_to_color/image_raw'))):
                self.create_subscription(Image, topic,
                    lambda msg, v=view, k=kind: self.image(v, k, msg), sensor)
            self.create_subscription(CameraInfo, camera + '/color/camera_info',
                lambda msg, v=view: self.info.update({v: msg}), sensor)
            self.create_subscription(Header, prefix + '/segmentation_input',
                lambda msg, v=view: self.pin(v, msg), 10)
            mask_topic = prefix + ('/mask' if view == 'head' else '/selected_mask')
            self.create_subscription(Image, mask_topic, lambda msg, v=view: self.mask(v, msg), 10)
            if view == 'head':
                self.create_subscription(CameraInfo, camera + '/depth/camera_info',
                    lambda msg: self.depth_info.update({'head': msg}), sensor)
                self.create_subscription(Image, '/perc/head/acquisition_rgb',
                    lambda msg: self.image('head', 'rgb', msg), 5)
            if view == 'wrist':
                # Reliable, exact acquisition copies survive the delayed segmentation.
                for kind, topic in (('rgb', prefix + '/acquisition_rgb'), ('depth', prefix + '/acquisition_depth')):
                    self.create_subscription(Image, topic,
                        lambda msg, v=view, k=kind: self.image(v, k, msg), 5)
                self.create_subscription(TargetCloud, prefix + '/target_cloud', self.packet, 10)
        # Retrying also resolves messages delivered in a different DDS order.
        self.create_timer(.1, self.process_pending)
        self.get_logger().info(f'camera_mode={self.mode}, cloud_source={self.source}, F2G={self.enabled}')

    def init_estimator(self):
        if self.estimator is not None:
            return
        # Existing wrapper configures the vendored model import paths.
        from .f2g_pose_node import _f2g_root
        from f2g_pose.estimator import PoseEstimator
        import os
        self.get_logger().info('Loading shared F2G model on GPU...')
        self.estimator = PoseEstimator(
            checkpoint=os.path.join(_f2g_root, 'weights/f2g-pose.pt'),
            config=os.path.join(_f2g_root, 'configs/sope.yaml'),
            radio_repository=os.path.join(_f2g_root, 'third_party_local/RADIO'),
            radio_checkpoint=os.path.join(_f2g_root, 'weights/radio-v2.5-l_half.pth.tar'), device='cuda:0')

    def image(self, view, kind, message):
        self.buffers[view][kind].append(message)
        # A pin survives beyond the short streaming ring buffers.
        if view in self.headers:
            self.try_pin(view)

    def pin(self, view, header):
        self.headers[view] = header
        self.pinned.pop(view, None)
        self.pin_parts[view] = {}
        self.try_pin(view)

    def try_pin(self, view):
        target = stamp_ns(self.headers[view])
        buffers = self.buffers[view]
        parts = self.pin_parts.setdefault(view, {})
        rgb = next((m for m in reversed(buffers['rgb']) if stamp_ns(m.header) == target), None)
        if rgb is not None:
            parts['rgb'] = rgb
        if buffers['depth']:
            depth = min(buffers['depth'], key=lambda m: abs(stamp_ns(m.header) - target))
            if abs(stamp_ns(depth.header) - target) / 1e9 <= self.skew:
                previous = parts.get('depth')
                if previous is None or abs(stamp_ns(depth.header)-target) < abs(stamp_ns(previous.header)-target):
                    parts['depth'] = depth
        if 'rgb' in parts and 'depth' in parts:
            info_dict = getattr(self, 'info', {})
            depth_info_dict = getattr(self, 'depth_info', {})
            if view in info_dict:
                parts['info'] = copy.deepcopy(info_dict[view])
            if view in depth_info_dict:
                parts['depth_info'] = copy.deepcopy(depth_info_dict[view])
            self.pinned[view] = (parts['rgb'], parts['depth'])

    def mask(self, view, message):
        self.pending[view] = (message, time.monotonic())
        if view not in self.headers or stamp_ns(self.headers[view]) != stamp_ns(message.header):
            self.pin(view, message.header)

    def packet(self, message):
        self.packets[stamp_ns(message.target.acquisition_header)] = message
        while len(self.packets) > 4:
            del self.packets[next(iter(self.packets))]

    def process_pending(self):
        for view, (mask, started) in list(self.pending.items()):
            pair = self.pinned.get(view)
            if (pair is None or stamp_ns(pair[0].header) != stamp_ns(mask.header)
                    or view not in self.info or (view == 'head' and view not in self.depth_info) or (view == 'wrist' and stamp_ns(mask.header) not in self.packets)):
                if time.monotonic() - started > 10:
                    self.views.pop(view, None)
                    self.views.pop('fused', None)
                    self.clear_cloud(view)
                    self.clear_cloud('fused')
                    self.error(self.pairing_error(view, mask), view)
                    del self.pending[view]
                continue
            del self.pending[view]
            try:
                info = self.pin_parts.get(view, {}).get('info', self.info.get(view))
                depth_info = self.pin_parts.get(view, {}).get('depth_info', self.depth_info.get(view))
                geometry, ref = self.predict(view, mask, pair[0], pair[1], info, depth_info)
                self.views[view] = (geometry, ref)
                self.publish_view(view, geometry, mask.header)
                if view == 'head':
                    self.publish_observation(geometry, mask.header)
                # A new capture invalidates the previous fused result.
                self.views.pop('fused', None)
                self.clear_cloud('fused', geometry)
                self.fusion_metrics = None
                if 'head' in self.views and 'wrist' in self.views:
                    try:
                        merged, self.fusion_metrics = fuse(self.views['head'][0], self.views['wrist'][0], self.max_age)
                        self.views['fused'] = (merged, None)
                        self.publish_view('fused', merged, mask.header)
                    except Exception as exc:
                        self.error(str(exc), 'fused')
                self.publish_selected()
            except Exception as exc:
                self.views.pop(view, None)
                self.views.pop('fused', None)
                self.clear_cloud(view)
                self.clear_cloud('fused')
                self.error(f'{view}: {exc}', view)

    def pairing_error(self, view, mask):
        target = stamp_ns(mask.header)
        parts = self.pin_parts.get(view, {})
        depth = self.buffers[view]['depth']
        skew = min((abs(stamp_ns(m.header)-target)/1e9 for m in depth), default=None)
        return (f'{view}: acquisition pairing failed at {target}; '
                f'exact_rgb={"rgb" in parts}, matched_depth={"depth" in parts}, '
                f'color_info={view in self.info}, depth_info={view in self.depth_info}, '
                f'nearest_buffered_depth_skew_s={skew}, '
                f'wrist_selection={target in self.packets}. No current-frame substitution was made.')

    def clear_cloud(self, source, geometry=None):
        from .cloud_utils import points_to_pointcloud2
        header = self.header(geometry) if geometry is not None else Header(frame_id=self.frame)
        empty = points_to_pointcloud2(np.zeros((0, 3)), header)
        self.view_publishers[source][0].publish(empty)
        if source in self.observed_publishers:
            self.observed_publishers[source].publish(empty)

    def error(self, message, source):
        if source == 'fused' and self.source != 'fused':
            self.get_logger().warn(f'Optional fusion unavailable; selected {self.source} cloud remains independent: {message}')
        else:
            self.get_logger().error(message)
        self.status_pub.publish(String(data=json.dumps({'status': 'failed', 'message': message,
            'source': source, 'stamp_ns': self.get_clock().now().nanoseconds})))

    def predict(self, view, mask_message, rgb_message, depth_message, info=None, depth_info=None):
        # Importing this module also locates the local F2G dependencies; no model
        # is allocated unless an enabled view actually requires it.
        from .cloud_utils import transform_stamped_to_matrix
        from sensor_msgs_py import point_cloud2 as pc2
        import cv2
        header = rgb_message.header
        acquisition = stamp_ns(header)
        if not 0 <= (self.get_clock().now().nanoseconds - acquisition) / 1e9 <= self.max_age:
            raise ValueError('Capture is stale or has a future timestamp')
        if info is None:
            info = self.info[view]
        if depth_info is None:
            depth_info = self.depth_info.get(view)
        if (header.frame_id != info.header.frame_id or (view == 'wrist' and depth_message.header.frame_id != info.header.frame_id)):
            raise ValueError('Registered RGB/depth/CameraInfo optical frames must match')
        rgb = self.bridge.imgmsg_to_cv2(rgb_message, 'rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_message, 'passthrough')
        mask = self.bridge.imgmsg_to_cv2(mask_message, 'mono8') > 0
        K = np.array(info.k, dtype=np.float64).reshape(3, 3)
        input_scale = .001 if depth_message.encoding in ('16UC1', 'mono16') else 1.
        if mask.shape != rgb.shape[:2] or (info.height,info.width) != rgb.shape[:2]:
            raise ValueError('RGB/mask/CameraInfo dimensions must match')
        if view == 'head':
            from .rgbd import register_depth, rectify_color
            if depth_info is None:
                raise ValueError('Raw head depth and corrected depth calibration do not match')
            if (depth_info.header.frame_id != depth_message.header.frame_id or
                    (depth_info.height,depth_info.width) != depth.shape):
                raise ValueError('Raw head depth and corrected depth calibration do not match')
            kd = np.array(depth_info.p).reshape(3,4)[:,:3]
            if kd[0,0] <= 0 or kd[1,1] <= 0:
                kd = np.array(depth_info.k).reshape(3,3)
            extrinsics = self.tf.lookup_transform(header.frame_id, depth_message.header.frame_id,
                rclpy.time.Time.from_msg(depth_message.header.stamp), timeout=Duration(seconds=.2))
            depth = register_depth(depth.astype(np.float32)*input_scale, kd, K,
                transform_stamped_to_matrix(extrinsics), rgb.shape[:2])
            if info.distortion_model not in ('', 'plumb_bob', 'rational_polynomial'):
                raise ValueError('Unsupported raw head color distortion model')
            rgb,mask = rectify_color(rgb, mask, K, info.d)
            input_scale = 1.  # Registered depth is in color-frame metres.
        if rgb.shape[:2] != depth.shape:
            raise ValueError('Registered RGB/depth dimensions must match')
        # Head uses the largest connected component. Wrist uses the existing
        # depth-validated selected mask and target packet.
        if view == 'head':
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            if count < 2:
                raise ValueError('No segmented object')
            mask = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        z = depth.astype(np.float32, copy=False) * input_scale
        valid = np.isfinite(z) & (z > .05) & (z < 3.5)
        K = np.asarray(info.k).reshape(3, 3)
        if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError('Invalid camera intrinsics')
        tf = self.tf.lookup_transform(self.frame, header.frame_id, rclpy.time.Time.from_msg(header.stamp),
                                      timeout=Duration(seconds=.2))
        T = transform_stamped_to_matrix(tf)
        project = lambda y, x: np.column_stack(((x-K[0,2])*z[y,x]/K[0,0],
                                                 (y-K[1,2])*z[y,x]/K[1,1], z[y,x]))
        y, x = np.nonzero(mask & valid)
        points = project(y, x)
        if len(points) < 50:
            raise ValueError('Fewer than 50 valid masked depth points')
        oy, ox = np.nonzero((valid & ~mask)[::8, ::8])
        oy, ox = oy * 8, ox * 8
        obstacles = project(oy, ox) @ T[:3, :3].T + T[:3, 3]
        if view == 'wrist':
            packet = self.packets[acquisition]
            if packet.invalidated or not packet.target.instance_validated or packet.target.class_label.casefold() != self.label.casefold():
                raise ValueError('Wrist selection is invalid or has a different target label')
            data = pc2.read_points_numpy(packet.points, field_names=['x', 'y', 'z', 'intensity'])
            data = data[np.isfinite(data).all(axis=1)]
            points, obstacles = data[data[:,3] > .5,:3], data[data[:,3] <= .5,:3]
            packet_tf = self.tf.lookup_transform(self.frame, packet.points.header.frame_id,
                rclpy.time.Time.from_msg(packet.points.header.stamp), timeout=Duration(seconds=.2))
            P = transform_stamped_to_matrix(packet_tf)
            points, obstacles = (points @ P[:3,:3].T + P[:3,3], obstacles @ P[:3,:3].T + P[:3,3])
        else:
            points = points @ T[:3, :3].T + T[:3, 3]
        self.get_logger().info(f'{view}: paired RGB-D at {acquisition}; observed target={len(points)} points; F2G={self.enabled[view]}')
        observed = estimate(points, obstacles, acquisition, view)
        pose = dimensions = None
        if self.enabled[view]:
            if self.estimator is None:
                self.init_estimator()
            self.get_logger().info(f'{view}: running F2G shape/pose inference...')
            result = self.estimator.predict(rgb=rgb, depth=np.where(valid, depth, 0), K=K.astype(np.float32),
                mask=mask.astype(np.uint8)*255,
                depth_scale=input_scale, return_shape=True)
            points = result['points_cam_m'] @ T[:3,:3].T + T[:3,3]
            pose, dimensions = T @ result['T_cam_obj'], result['size_m']
        geometry = (estimate(points, observed.obstacles, acquisition, view, pose, dimensions,
                             obstacles_are_voxelized=True) if self.enabled[view] else observed)
        self.observed_views[view] = observed
        self.get_logger().info(f'{view}: object ready, cloud={len(geometry.points)} points, center={geometry.center.tolist()}, dimensions={geometry.dimensions.tolist()}')
        ref = ObservationRef()
        ref.header = copy.deepcopy(header)
        ref.header.frame_id = self.frame
        ref.source_id = view + ('_f2g' if self.enabled[view] else '_depth')
        ref.acquisition_id = acquisition
        from .cloud_utils import matrix_to_pose
        ref.camera_pose = matrix_to_pose(T)
        ref.valid_depth_fraction = float(np.count_nonzero(valid & mask) / max(1, np.count_nonzero(mask)))
        # No registration was attempted for an individual view.
        ref.registration_residual = float('nan')
        return geometry, ref

    def header(self, geometry):
        header = Header(frame_id=self.frame)
        header.stamp.sec, header.stamp.nanosec = divmod(geometry.stamp_ns, 10**9)
        return header

    def publish_view(self, source, geometry, unused_header=None):
        from .cloud_utils import points_to_pointcloud2, matrix_to_pose
        header = self.header(geometry)
        cloud, pose_pub, size_pub = self.view_publishers[source]
        cloud.publish(points_to_pointcloud2(geometry.points, header))
        if source in self.observed_views:
            self.observed_publishers[source].publish(
                points_to_pointcloud2(self.observed_views[source].points, header))
        pose = PoseStamped(header=header, pose=matrix_to_pose(geometry.pose))
        dims = Vector3Stamped(header=header)
        dims.vector.x, dims.vector.y, dims.vector.z = map(float, geometry.model_dimensions)
        pose_pub.publish(pose)
        size_pub.publish(dims)

    def publish_observation(self, geometry, unused_header):
        from grasp_pipeline._geometry import compute_isometric_view
        msg = ObservationTarget(header=self.header(geometry), target_label=self.label)
        # Confidence is not a measured probability of 3-D pose accuracy.
        msg.confidence = 1.
        msg.target_centroid.x, msg.target_centroid.y, msg.target_centroid.z = map(float, geometry.center)
        msg.target_dimensions.x, msg.target_dimensions.y, msg.target_dimensions.z = map(float, geometry.dimensions)
        msg.standoff_distance = float(self.get_parameter('standoff_dist').value)
        msg.elevation_deg = float(self.get_parameter('elevation_deg').value)
        msg.azimuth_deg = float(self.get_parameter('azimuth_deg').value)
        result = compute_isometric_view(geometry.center.tolist(), msg.standoff_distance,
                                        msg.elevation_deg, msg.azimuth_deg, 'left')
        for name, pos, quat in (('wrist_camera_pose', 'cam_pos', 'cam_quat_xyzw'),
                                ('arm_target_pose', 'ee_pos', 'ee_quat_xyzw')):
            pose = getattr(msg, name)
            pose.position.x, pose.position.y, pose.position.z = map(float, result[pos])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, result[quat])
        self.observation_pub.publish(msg)

    def publish_selected(self):
        from .cloud_utils import points_to_pointcloud2
        if self.source not in self.views:
            return
        geometry = self.views[self.source][0]
        contributing = ('head', 'wrist') if self.source == 'fused' else (self.source,)
        snapshot = MultiViewSnapshot(header=self.header(geometry), session_uuid=self.session,
                                     target_label=self.label)
        self.generation += 1
        snapshot.snapshot_generation = self.generation
        snapshot.static_scene_verified = False  # No stationary-camera verification exists here.
        snapshot.fused_target_cloud = points_to_pointcloud2(geometry.points, snapshot.header)
        snapshot.scene_obstacles = points_to_pointcloud2(geometry.obstacles, snapshot.header)
        snapshot.observations = [copy.deepcopy(self.views[v][1]) for v in contributing]
        if self.source == 'fused':
            for ref in snapshot.observations:
                ref.registration_residual = self.fusion_metrics['rmse_m']
        lo, hi = geometry.points.min(axis=0), geometry.points.max(axis=0)
        for field, values in (('centroid', geometry.center), ('bounds_min', lo), ('bounds_max', hi)):
            vector = getattr(snapshot, field)
            vector.x, vector.y, vector.z = map(float, values)
        # Atomic metadata is correlated with the snapshot generation, never paired
        # by arrival order with an unrelated latched dimensions topic.
        metadata = dict(session_uuid=self.session, generation=self.generation, frame=self.frame,
                        target=self.label, camera_mode=self.mode, cloud_source=self.source,
                        f2g=self.enabled, fusion=self.fusion_metrics, geometries={})
        for source, (value, _) in self.views.items():
            metadata['geometries'][source] = dict(center=value.center.tolist(), dimensions=value.dimensions.tolist(),
                model_pose=value.pose.tolist(), model_dimensions=value.model_dimensions.tolist(),
                stamp_ns=value.stamp_ns, completed=value.completed)
            if source in self.observed_views:
                observed = self.observed_views[source]
                metadata['geometries'][source]['observed'] = dict(
                    stamp_ns=observed.stamp_ns, center=observed.center.tolist(),
                    dimensions=observed.dimensions.tolist(), points=len(observed.points))
        self.geometry_pub.publish(String(data=json.dumps(metadata, allow_nan=False)))
        self.snapshot_pub.publish(snapshot)
        self.status_pub.publish(String(data=json.dumps({'status': 'ready', 'generation': self.generation,
            'source': self.source, 'stamp_ns': self.get_clock().now().nanoseconds})))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ObjectPerceptionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # ROS may invalidate a pending take_message during SIGINT shutdown.
        # Preserve runtime failures while the context is still active.
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
