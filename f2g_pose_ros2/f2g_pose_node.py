#!/usr/bin/env python3
import os
import sys

def _resolve_conda_site():
    py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    candidates = [
        os.environ.get('F2G_CONDA_SITE', ''),
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib', py_ver, 'site-packages')
        if os.environ.get('CONDA_PREFIX') else '',
        os.path.expanduser(f'~/miniconda3/envs/grasp/lib/{py_ver}/site-packages'),
        os.path.expanduser(f'~/anaconda3/envs/grasp/lib/{py_ver}/site-packages'),
    ]
    for site in candidates:
        if site and os.path.isdir(site) and site not in sys.path:
            sys.path.insert(0, site)
            break


def _resolve_f2g_root():
    env_root = os.environ.get('F2G_POSE_ROOT', '')
    if env_root and os.path.isdir(env_root):
        return env_root
    pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    local_root = os.path.join(pkg_root, 'third_party', 'F2G-Pose')
    if os.path.isdir(local_root):
        return local_root
    try:
        from ament_index_python.packages import get_package_share_directory
        share_root = os.path.join(get_package_share_directory('f2g_pose_ros2'), 'third_party', 'F2G-Pose')
        if os.path.isdir(share_root):
            return share_root
    except Exception:
        pass
    return local_root


_resolve_conda_site()
_f2g_root = _resolve_f2g_root()

_f2g_src = os.path.join(_f2g_root, 'src')
if os.path.exists(_f2g_src) and _f2g_src not in sys.path:
    sys.path.insert(0, _f2g_src)

import collections
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
import message_filters

import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import visualization_msgs.msg
from std_srvs.srv import Trigger, SetBool

import tf2_ros
from tf2_ros import Buffer, TransformListener

from core_r1_interfaces.msg import MultiViewSnapshot, ObservationRef, ObservationTarget, TargetAcquisition, TargetCloud
from .image_buffer import ImageBuffer
from .acquisition import SelectionGuard, validate_acquisition, stamp_ns

from .cloud_utils import (
    points_to_pointcloud2,
    pointcloud2_to_points,
    pointcloud2_to_labeled_points,
    transform_stamped_to_matrix,
    transform_points,
    matrix_to_pose,
    create_bbox_marker,
    voxel_grid_downsample,
)
from .fusion import align_and_fuse_shapes


class F2GPoseNode(Node):
    def __init__(self, estimator=None):
        super().__init__('f2g_pose_node')
        self.get_logger().info('Initializing F2G-Pose Node...')

        # Parameters
        self.declare_parameter('f2g_mode', 'fusion')  # 'fusion' (Option A, default) or 'direct_head' (Option C)
        self.declare_parameter('target_label', '')
        self.declare_parameter('target_object', '')
        self.declare_parameter('checkpoint', os.path.join(_f2g_root, 'weights', 'f2g-pose.pt'))
        self.declare_parameter('config_path', os.path.join(_f2g_root, 'configs', 'sope.yaml'))
        self.declare_parameter('radio_repo', os.path.join(_f2g_root, 'third_party_local', 'RADIO'))
        self.declare_parameter('radio_checkpoint', os.path.join(_f2g_root, 'weights', 'radio-v2.5-l_half.pth.tar'))
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('preload_models', True)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('head_color_topic', '/camera/head_d455/color/image_raw')
        self.declare_parameter('head_depth_topic', '/camera/head_d455/depth/image_rect_raw')
        self.declare_parameter('head_info_topic', '/camera/head_d455/color/camera_info')
        self.declare_parameter('head_depth_info_topic', '/camera/head_d455/depth/camera_info')
        self.declare_parameter('head_mask_topic', '/perc/head/mask')
        self.declare_parameter('wrist_color_topic', '/camera/left_d405/color/image_raw')
        self.declare_parameter('wrist_depth_topic', '/camera/left_d405/aligned_depth_to_color/image_raw')
        self.declare_parameter('wrist_info_topic', '/camera/left_d405/color/camera_info')
        self.declare_parameter('wrist_mask_topic', '/perc/grasp/selected_mask')
        self.declare_parameter('wrist_cloud_topic', '/perc/grasp/cleaned_points')
        self.declare_parameter('wrist_f2g_enabled', True)
        self.declare_parameter('wrist_acquisition_topic', '/perc/grasp/acquisition')
        self.declare_parameter('max_acquisition_age', 180.)
        self.declare_parameter('max_acquisition_bytes', 64*1024*1024)
        self.declare_parameter('capture_dir', os.environ.get('F2G_CAPTURE_DIR', ''))
        self.declare_parameter('fusion_enable_icp', True)
        for name, value in dict(fusion_voxel_size=.003, fusion_overlap_distance=.012,
                fusion_max_rotation_deg=5., fusion_max_centroid_shift=.01,
                fusion_max_point_shift=.015, fusion_max_alignment_distance=.03,
                fusion_min_alignment_fraction=.3, fusion_visibility_radius_pixels=8.,
                fusion_visibility_depth_tolerance=.0005).items():
            self.declare_parameter(name, value)
        self.declare_parameter('standoff_dist', 0.50)
        self.declare_parameter('elevation_deg', 48.0)
        self.declare_parameter('azimuth_deg', 25.0)
        self.declare_parameter('arm_side', 'left')
        self.declare_parameter('auto_trigger', False)
        self.declare_parameter('publish_rate_limit', 2.0)

        self.f2g_mode = self.get_parameter('f2g_mode').get_parameter_value().string_value
        self.target_label = self.get_parameter('target_label').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.standoff_dist = self.get_parameter('standoff_dist').get_parameter_value().double_value
        self.elevation_deg = self.get_parameter('elevation_deg').get_parameter_value().double_value
        self.azimuth_deg = self.get_parameter('azimuth_deg').get_parameter_value().double_value
        self.arm_side = self.get_parameter('arm_side').get_parameter_value().string_value
        self.auto_trigger = self.get_parameter('auto_trigger').get_parameter_value().bool_value
        self.publish_rate_limit = self.get_parameter('publish_rate_limit').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=self.get_parameter('max_acquisition_age').value + 30.))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State storage
        self.latest_color = None
        self.latest_depth = None
        self.latest_info = None
        self.latest_depth_info = None
        image_budget = int(self.declare_parameter("image_history_bytes", 134217728).value)
        if image_budget < 4:
            raise ValueError("image_history_bytes must allow four image rings")
        self.color_buffer = ImageBuffer(max_bytes=image_budget // 4)
        self.depth_buffer = ImageBuffer(max_bytes=image_budget // 4)
        self.pinned_rgbd = None
        self.latest_wrist_color = None
        self.latest_wrist_depth = None
        self.latest_wrist_info = None
        self.latest_wrist_mask = None
        self.wrist_color_buffer = ImageBuffer(max_bytes=image_budget // 4)
        self.wrist_depth_buffer = ImageBuffer(max_bytes=image_budget // 4)
        self._startup_triggered = False
        self.latest_completed_cloud_base = None
        self.latest_object_pose_base = None
        self.latest_object_dimensions = None
        self.latest_T_cam_obj = None
        self.latest_head_header = None
        self.last_predict_time = 0.0
        self.snapshot_generation = 0
        self.session_uuid = str(uuid.uuid4())
        self.wrist_selection = SelectionGuard()
        self.last_wrist_clock = self.get_clock().now().nanoseconds
        self.latest_wrist_prediction = None

        # Publishers
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_completed_cloud = self.create_publisher(
            sensor_msgs.msg.PointCloud2, '/perc/f2g/completed_cloud', latched_qos)
        self.pub_object_pose = self.create_publisher(
            geometry_msgs.msg.PoseStamped, '/perc/f2g/object_pose', latched_qos)
        self.pub_object_dimensions = self.create_publisher(
            geometry_msgs.msg.Vector3Stamped, '/perc/f2g/object_dimensions', latched_qos)
        self.pub_bbox_marker = self.create_publisher(
            visualization_msgs.msg.Marker, '/perc/f2g/bbox_marker', 10)
        self.pub_fused_cloud = self.create_publisher(
            sensor_msgs.msg.PointCloud2, '/perc/f2g/fused_target_cloud', latched_qos)
        self.pub_snapshot = self.create_publisher(
            MultiViewSnapshot, '/perc/multiview/snapshot', latched_qos)
        self.pub_fusion_status = self.create_publisher(std_msgs.msg.String, '/perc/f2g/status', latched_qos)
        self.pub_observation_target = self.create_publisher(
            ObservationTarget, '/perc/head/observation_target', latched_qos)

        # Services & Clients
        self.srv_trigger = self.create_service(Trigger, '~/predict', self.handle_trigger_srv)
        self.srv_set_mode = self.create_service(SetBool, '~/set_mode', self.handle_set_mode_srv)
        self.head_trigger_client = self.create_client(Trigger, '/perc/head/trigger')
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Use preloaded estimator or initialize if not provided
        self.model_loads = int(estimator is not None)
        self.model_inferences = 0
        self.model_busy = False
        self.model_error = ""
        self.last_inference_seconds = None
        self.first_inference_seconds = None
        self.srv_model_status = self.create_service(Trigger, "~/model_status", self.handle_model_status)
        self.estimator = estimator
        if self.estimator is None and self.get_parameter('preload_models').value:
            self.init_estimator()
        self.init_subscribers()
        self.get_logger().info(f'F2G-Pose Node ready in mode: [{self.f2g_mode}]')

    def init_estimator(self):
        checkpoint = os.path.expanduser(self.get_parameter('checkpoint').get_parameter_value().string_value) or os.path.join(_f2g_root, 'weights', 'f2g-pose.pt')
        config_path = os.path.expanduser(self.get_parameter('config_path').get_parameter_value().string_value) or os.path.join(_f2g_root, 'configs', 'sope.yaml')
        radio_repo = os.path.expanduser(self.get_parameter('radio_repo').get_parameter_value().string_value) or os.path.join(_f2g_root, 'third_party_local', 'RADIO')
        radio_ckpt = os.path.expanduser(self.get_parameter('radio_checkpoint').get_parameter_value().string_value) or os.path.join(_f2g_root, 'weights', 'radio-v2.5-l_half.pth.tar')
        device = self.get_parameter('device').get_parameter_value().string_value

        self.get_logger().info(f'Loading F2G-Pose weights from: {checkpoint}')
        from f2g_pose.estimator import PoseEstimator
        self.estimator = PoseEstimator(
            checkpoint=checkpoint,
            config=config_path,
            radio_repository=radio_repo,
            radio_checkpoint=radio_ckpt,
            device=device
        )
        self.model_loads += 1
        self.get_logger().info('F2G-Pose model loaded successfully on GPU.')

    def handle_model_status(self, request, response):
        response.success = self.estimator is not None
        response.message = json.dumps(dict(loaded=response.success, warmed=self.model_inferences > 0,
            busy=self.model_busy, loads=self.model_loads, unloads=0, error=self.model_error,
            head_rgbd=dict(color=self.latest_color is not None, depth=self.latest_depth is not None,
                           calibration=self.latest_info is not None),
            first_inference_seconds=self.first_inference_seconds,
            last_inference_seconds=self.last_inference_seconds))
        return response

    def _predict(self, **kwargs):
        started = time.monotonic()
        self.model_busy = True
        self.model_error = ""
        try:
            result = self.estimator.predict(**kwargs)
            self.model_inferences = getattr(self, 'model_inferences', 0) + 1
            return result
        except Exception as error:
            self.model_error = str(error)
            raise
        finally:
            self.model_busy = False
            self.last_inference_seconds = time.monotonic() - started
            if getattr(self, 'first_inference_seconds', None) is None:
                self.first_inference_seconds = self.last_inference_seconds

    def init_subscribers(self):
        color_topic = self.get_parameter('head_color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('head_depth_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('head_info_topic').get_parameter_value().string_value
        mask_topic = self.get_parameter('head_mask_topic').get_parameter_value().string_value
        wrist_topic = self.get_parameter('wrist_cloud_topic').get_parameter_value().string_value

        self.get_logger().info(f'Subscribing head: color={color_topic}, depth={depth_topic}, mask={mask_topic}')
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.sub_color = self.create_subscription(
            sensor_msgs.msg.Image,
            color_topic,
            self.color_callback,
            sensor_qos
        )
        self.sub_depth = self.create_subscription(
            sensor_msgs.msg.Image,
            depth_topic,
            self.depth_callback,
            sensor_qos
        )
        self.sub_info = self.create_subscription(
            sensor_msgs.msg.CameraInfo,
            info_topic,
            self.info_callback,
            sensor_qos
        )
        depth_info_topic = self.get_parameter('head_depth_info_topic').get_parameter_value().string_value
        self.sub_depth_info = self.create_subscription(
            sensor_msgs.msg.CameraInfo,
            depth_info_topic,
            self.depth_info_callback,
            sensor_qos
        )

        # Pin the exact RGB-D frame when SAM3 begins encoding
        self.sub_seg_input = self.create_subscription(
            std_msgs.msg.Header,
            '/perc/head/segmentation_input',
            self.segmentation_input_callback,
            10
        )

        # Subscribe directly to head mask (arrives ~24s after SAM3 trigger)
        self.sub_mask = self.create_subscription(
            sensor_msgs.msg.Image,
            mask_topic,
            self.mask_callback,
            10
        )
        self.startup_timer = self.create_timer(1.0, self._startup_trigger_check)

        # Wrist cloud and RGB-D subscribers for Option A (Arm F2G + Wrist Cloud Fusion)
        wrist_color_topic = self.get_parameter('wrist_color_topic').get_parameter_value().string_value
        wrist_depth_topic = self.get_parameter('wrist_depth_topic').get_parameter_value().string_value
        wrist_info_topic = self.get_parameter('wrist_info_topic').get_parameter_value().string_value
        wrist_mask_topic = self.get_parameter('wrist_mask_topic').get_parameter_value().string_value
        self.get_logger().info(
            f'Subscribing wrist: color={wrist_color_topic}, depth={wrist_depth_topic}, mask={wrist_mask_topic}, cloud={wrist_topic}'
        )
        wrist_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        # Selection, images and calibration arrive atomically. Live image/mask
        # topics are previews, never substitutes for this immutable acquisition.
        self.sub_wrist_acquisition = self.create_subscription(TargetAcquisition,
            self.get_parameter('wrist_acquisition_topic').value, self.wrist_acquisition_callback, 1)
        self.sub_wrist_selection = self.create_subscription(TargetCloud,
            '/perc/grasp/target_cloud', self.wrist_selection_callback, 1)

    @staticmethod
    def _stamp_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _resolve_cam_frame(self, cam_frame: str) -> str:
        """Resolve camera optical frame in TF (handles head_color_optical_frame vs head_d455_color_optical_frame)."""
        for parent in (self.base_frame, 'torso_link4'):
            if self.tf_buffer.can_transform(parent, cam_frame, rclpy.time.Time()):
                return cam_frame
            if cam_frame.startswith('head_') and not cam_frame.startswith('head_d455_'):
                alt_frame = 'head_d455_' + cam_frame[len('head_'):]
                if self.tf_buffer.can_transform(parent, alt_frame, rclpy.time.Time()):
                    return alt_frame
        return ''

    def _lookup_T_base_cam(self, cam_frame: str, stamp) -> np.ndarray:
        """Look up T_base_cam from TF at exact acquisition stamp with fallback to latest available."""
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                cam_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except Exception as ex:
            self.get_logger().warn(
                f"TF lookup at stamp ({stamp.sec}.{stamp.nanosec}) failed: {ex}. Falling back to latest transform (Time(0))."
            )
            tf_stamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                cam_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        return transform_stamped_to_matrix(tf_stamped)

    def color_callback(self, color_msg):
        self.latest_color = color_msg
        self.color_buffer.append(color_msg)

    def depth_callback(self, depth_msg):
        self.latest_depth = depth_msg
        self.depth_buffer.append(depth_msg)

    def info_callback(self, info_msg):
        """Cache latest CameraInfo intrinsics."""
        self.latest_info = info_msg

    def depth_info_callback(self, info_msg):
        self.latest_depth_info = info_msg

    def wrist_color_callback(self, color_msg):
        self.latest_wrist_color = color_msg
        self.wrist_color_buffer.append(color_msg)

    def wrist_depth_callback(self, depth_msg):
        self.latest_wrist_depth = depth_msg
        self.wrist_depth_buffer.append(depth_msg)

    def wrist_info_callback(self, info_msg):
        self.latest_wrist_info = info_msg

    def wrist_mask_callback(self, mask_msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
            if int(np.count_nonzero(mask)) > 0:
                self.latest_wrist_mask = mask_msg
        except Exception:
            self.latest_wrist_mask = mask_msg

    def _startup_trigger_check(self):
        """Trigger initial single-frame Head SAM3 pass once camera streams and trigger service are ready."""
        if not self.auto_trigger or self._startup_triggered:
            self.startup_timer.cancel()
            return
        if self.latest_color is None or self.latest_depth is None or self.latest_info is None:
            self.get_logger().info(
                f"Waiting for head camera streams (color={self.latest_color is not None}, "
                f"depth={self.latest_depth is not None}, info={self.latest_info is not None})...",
                throttle_duration_sec=3.0
            )
            return
        if self.head_trigger_client.service_is_ready():
            self._startup_triggered = True
            self.startup_timer.cancel()
            self.pinned_rgbd = (self.latest_color, self.latest_depth, self.latest_info, getattr(self, 'latest_depth_info', None))
            self.get_logger().info(
                f"Triggering initial single-frame Head SAM3 segmentation for '{self.target_label}'..."
            )
            self.head_trigger_client.call_async(Trigger.Request())
        else:
            self.get_logger().info(
                "Waiting for /perc/head/trigger service to become ready...",
                throttle_duration_sec=3.0
            )

    def segmentation_input_callback(self, header_msg):
        """Pin the RGB-D frame corresponding to the image SAM3 just started encoding."""
        target_ns = header_msg.stamp.sec * 10**9 + header_msg.stamp.nanosec
        matched_color = next((m for m in reversed(self.color_buffer)
                              if (m.header.stamp.sec * 10**9 + m.header.stamp.nanosec) == target_ns), None)
        if matched_color is None:
            matched_color = self.latest_color
        matched_depth = None
        if self.depth_buffer:
            matched_depth = min(self.depth_buffer,
                                key=lambda m: abs((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec) - target_ns))
            if abs((matched_depth.header.stamp.sec * 10**9 + matched_depth.header.stamp.nanosec) - target_ns) / 1e9 > 0.05:
                matched_depth = None
        if matched_depth is None:
            matched_depth = self.latest_depth
        if matched_color is not None and matched_depth is not None and self.latest_info is not None:
            self.pinned_rgbd = (matched_color, matched_depth, self.latest_info, getattr(self, 'latest_depth_info', None))

    def mask_callback(self, mask_msg):
        """Receive segmentation mask (even ~24s after acquisition) and run F2G-Pose."""
        try:
            raw_mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
            if int(np.count_nonzero(raw_mask)) == 0 and self.pinned_rgbd is None:
                # Ignore 0-pixel invalidation masks emitted when target_object parameter changes
                return
        except Exception:
            pass

        matched = self.pinned_rgbd
        if matched is None and self.latest_color is not None and self.latest_depth is not None and self.latest_info is not None:
            matched = (self.latest_color, self.latest_depth, self.latest_info, getattr(self, 'latest_depth_info', None))

        if matched is None:
            self.get_logger().warn('Received head mask, but no RGB-D frame is buffered yet.')
            return

        color_msg, depth_msg, info_msg, depth_info_msg = (
            matched if len(matched) == 4 else (*matched, getattr(self, 'latest_depth_info', None))
        )
        self.head_observation_callback(color_msg, depth_msg, info_msg, mask_msg, depth_info_msg)

    def _on_set_parameters(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name.startswith('fusion_') and p.name != 'fusion_enable_icp' or p.name in ('max_acquisition_age', 'max_acquisition_bytes'):
                if not isinstance(p.value, (int, float)) or not math.isfinite(p.value) or p.value <= 0:
                    return SetParametersResult(successful=False, reason=p.name + ' must be finite and positive')
            if p.name == 'fusion_min_alignment_fraction' and p.value > 1:
                return SetParametersResult(successful=False, reason='Alignment fraction must not exceed 1')
        for p in params:
            if p.name in ('target_label', 'target_object'):
                new_label = str(p.value).strip()
                if new_label and new_label.casefold() != (self.target_label or '').casefold():
                    self.latest_wrist_prediction = None
                    self._clear_wrist_output(self.get_clock().now().to_msg())
                    self.pinned_rgbd = None
                    self.latest_wrist_mask = None
                    self.latest_completed_cloud_base = None
                    self.latest_object_pose_base = None
                    self.latest_object_dimensions = None
                    self.latest_T_cam_obj = None
                self.target_label = new_label
                self.get_logger().info(f"Target label dynamically updated to: '{self.target_label}'")
        return SetParametersResult(successful=True)

    def handle_set_mode_srv(self, request, response):
        """Set mode: True -> 'fusion' (Option A), False -> 'direct_head' (Option C)."""
        self.f2g_mode = 'fusion' if request.data else 'direct_head'
        response.success = True
        response.message = f"Switched f2g_mode to '{self.f2g_mode}'"
        self.get_logger().info(response.message)
        return response

    def handle_trigger_srv(self, request, response):
        """Request head segmentation; F2G runs when its resulting mask arrives."""
        if not self.head_trigger_client.service_is_ready():
            response.success = False
            response.message = "Head segmentation service /perc/head/trigger is not ready."
            return response
        self.head_trigger_client.call_async(Trigger.Request())
        response.success = True
        response.message = "Head segmentation requested; F2G results will publish after mask completion."
        return response

    def head_observation_callback(self, color_msg, depth_msg, info_msg, mask_msg, depth_info_msg=None):
        now = time.time()
        cam_frame = self._resolve_cam_frame(color_msg.header.frame_id)
        if not cam_frame:
            self.last_predict_time = now
            self.get_logger().warn(
                f"Waiting for TF connection: '{color_msg.header.frame_id}' -> '{self.base_frame}'. "
                f"Please ensure MoveIt (start_moveit.sh) or robot_state_publisher is running.",
                throttle_duration_sec=5.0
            )
            return

        try:
            # 1. Convert ROS images to numpy
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            # Binarize mask
            mask = (mask > 0).astype(np.uint8) * 255
            nz = int(np.count_nonzero(mask))

            if nz < 20:
                self.get_logger().warn(
                    f"Head mask for '{self.target_label}' is empty ({nz} non-zero px); skipping F2G-Pose."
                )
                return

            self.last_predict_time = now
            self.get_logger().info(
                f"Received head mask ({nz} px) for '{self.target_label}'. Running F2G-Pose 3D shape completion..."
            )
            t0 = time.perf_counter()

            rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
            input_scale = 0.001 if depth_msg.encoding in ('16UC1', 'mono16') else 1.0
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_m = depth.astype(np.float32) * input_scale
            else:
                depth_m = depth.astype(np.float32)
                depth = (depth * 1000.0).astype(np.uint16)

            # Camera intrinsics K matrix (3x3)
            K = np.array(info_msg.k, dtype=np.float32).reshape(3, 3)

            # Perform depth-to-color registration if depth info is available
            actual_depth_scale = self.depth_scale
            if depth_info_msg is not None and hasattr(depth_info_msg, 'k'):
                try:
                    from .rgbd import register_depth, rectify_color
                    kd = np.array(depth_info_msg.p).reshape(3, 4)[:, :3]
                    if kd[0, 0] <= 0 or kd[1, 1] <= 0:
                        kd = np.array(depth_info_msg.k).reshape(3, 3)
                    try:
                        extrinsics = self.tf_buffer.lookup_transform(
                            color_msg.header.frame_id,
                            depth_msg.header.frame_id,
                            rclpy.time.Time.from_msg(depth_msg.header.stamp),
                            timeout=rclpy.duration.Duration(seconds=0.2)
                        )
                    except Exception:
                        extrinsics = self.tf_buffer.lookup_transform(
                            color_msg.header.frame_id,
                            depth_msg.header.frame_id,
                            rclpy.time.Time(),
                            timeout=rclpy.duration.Duration(seconds=0.5)
                        )
                    depth_registered = register_depth(
                        depth_m, kd, K,
                        transform_stamped_to_matrix(extrinsics),
                        rgb.shape[:2]
                    )
                    if getattr(info_msg, 'distortion_model', '') in ('', 'plumb_bob', 'rational_polynomial'):
                        rgb, mask = rectify_color(rgb, mask, K, getattr(info_msg, 'd', []))
                    depth = depth_registered
                    actual_depth_scale = 1.0
                except Exception as ex:
                    self.get_logger().warn(f"Depth-to-color registration omitted: {ex}")

            # 2. Predict 6D pose and completed 3D shape
            if self.estimator is None:
                self.init_estimator()
            res = self._predict(
                rgb=rgb,
                depth=depth,
                K=K,
                mask=mask,
                depth_scale=actual_depth_scale,
                return_shape=True
            )
            dt = time.perf_counter() - t0

            T_cam_obj = res['T_cam_obj']
            size_m = res['size_m']
            points_cam_m = res.get('points_cam_m', None)

            if points_cam_m is None or len(points_cam_m) == 0:
                self.get_logger().warn('F2G-Pose returned empty point cloud.')
                return

            # 3. Lookup TF to transform from camera frame to base_frame
            T_base_cam = self._lookup_T_base_cam(cam_frame, color_msg.header.stamp)

            # 4. Transform pose and points to base_frame
            T_base_obj = T_base_cam @ T_cam_obj
            pose_base = matrix_to_pose(T_base_obj)
            camera_pose_base = matrix_to_pose(T_base_cam)
            points_base = transform_points(points_cam_m, T_base_cam)
            head_obstacles_base = None
            if self.f2g_mode == 'direct_head':
                head_obstacles_base = self._extract_head_obstacles(
                    depth, mask, K, T_base_cam, pose_base, actual_depth_scale
                )

            # Cache latest results
            self.latest_completed_cloud_base = points_base
            self.latest_object_pose_base = pose_base
            self.latest_object_dimensions = size_m
            self.latest_T_cam_obj = T_cam_obj
            self.latest_head_header = color_msg.header
            self.last_predict_time = time.time()

            # 5. Publish estimates
            self.publish_head_estimates(
                points_base,
                pose_base,
                size_m,
                color_msg.header,
                obstacles_base=head_obstacles_base,
                camera_pose_base=camera_pose_base,
            )
            self.get_logger().info(
                f"F2G-Pose completed 3D shape ({len(points_base)} pts) in {dt:.2f}s at "
                f"({pose_base.position.x:.3f}, {pose_base.position.y:.3f}, {pose_base.position.z:.3f}) -> published to Rerun & /perc/f2g/completed_cloud"
            )

        except Exception as e:
            self.get_logger().error(f'Error during F2G-Pose prediction: {e}', throttle_duration_sec=2.0)

    @staticmethod
    def _extract_head_obstacles(depth, mask, K, T_base_cam, obj_pose_base, depth_scale, radius=0.40):
        """Extract single-camera non-mask scene obstacles in base_link for direct_head collision checking."""
        depth_arr = np.asarray(depth, dtype=np.float32)
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        if depth_arr.ndim != 2 or mask_arr.shape[:2] != depth_arr.shape:
            return np.zeros((0, 3), dtype=np.float32)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            return np.zeros((0, 3), dtype=np.float32)
        h, w = depth_arr.shape
        stride = 1 if min(h, w) <= 64 else 4
        ys = np.arange(0, h, stride, dtype=np.float32)
        xs = np.arange(0, w, stride, dtype=np.float32)
        depth_sub = depth_arr[::stride, ::stride] * float(depth_scale)
        mask_sub = mask_arr[::stride, ::stride] > 0
        valid = (~mask_sub) & np.isfinite(depth_sub) & (depth_sub > 0.1) & (depth_sub < 2.0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)
        u_grid, v_grid = np.meshgrid(xs, ys)
        z = depth_sub[valid]
        x = (u_grid[valid] - cx) * z / fx
        y = (v_grid[valid] - cy) * z / fy
        pts_cam = np.column_stack((x, y, z)).astype(np.float32)
        pts_base = transform_points(pts_cam, T_base_cam)
        if obj_pose_base is not None and len(pts_base) > 0:
            center = np.array([
                float(obj_pose_base.position.x),
                float(obj_pose_base.position.y),
                float(obj_pose_base.position.z),
            ], dtype=np.float32)
            dists = np.linalg.norm(pts_base - center, axis=1)
            pts_base = pts_base[dists <= float(radius)]
        return pts_base

    def publish_head_estimates(
        self,
        points_base,
        pose_base,
        size_m,
        src_header,
        obstacles_base=None,
        camera_pose_base=None,
    ):
        stamp = src_header.stamp
        base_header = std_msgs.msg.Header()
        base_header.stamp = stamp
        base_header.frame_id = self.base_frame

        # A. Publish completed point cloud
        cloud_msg = points_to_pointcloud2(points_base, base_header)
        self.pub_completed_cloud.publish(cloud_msg)

        # B. Publish 6D pose stamped
        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header = base_header
        pose_msg.pose = pose_base
        self.pub_object_pose.publish(pose_msg)

        # C. Publish 3D bounding box dimensions
        dim_msg = geometry_msgs.msg.Vector3Stamped()
        dim_msg.header = base_header
        dim_msg.vector.x = float(size_m[0])
        dim_msg.vector.y = float(size_m[1])
        dim_msg.vector.z = float(size_m[2])
        self.pub_object_dimensions.publish(dim_msg)

        # D. Publish RViz oriented bounding box marker
        marker_msg = create_bbox_marker(
            pose=pose_base,
            dimensions=size_m,
            frame_id=self.base_frame,
            stamp=base_header,
            color=(0.1, 0.9, 0.2, 0.5)
        )
        self.pub_bbox_marker.publish(marker_msg)

        # E. Publish ObservationTarget for R1 orbital planner (in Mode A)
        obs_msg = ObservationTarget()
        obs_msg.header = base_header
        obs_msg.target_label = self.target_label
        obs_msg.confidence = 1.0
        obs_msg.target_centroid.x = float(pose_base.position.x)
        obs_msg.target_centroid.y = float(pose_base.position.y)
        obs_msg.target_centroid.z = float(pose_base.position.z)
        obs_msg.target_dimensions.x = float(size_m[0])
        obs_msg.target_dimensions.y = float(size_m[1])
        obs_msg.target_dimensions.z = float(size_m[2])
        obs_msg.standoff_distance = float(self.standoff_dist)
        obs_msg.elevation_deg = float(self.elevation_deg)
        obs_msg.azimuth_deg = float(self.azimuth_deg)

        try:
            from grasp_pipeline._geometry import compute_isometric_view
            res = compute_isometric_view(
                [pose_base.position.x, pose_base.position.y, pose_base.position.z],
                self.standoff_dist,
                self.elevation_deg,
                self.azimuth_deg,
                self.arm_side
            )
            obs_msg.wrist_camera_pose.position.x = float(res["cam_pos"][0])
            obs_msg.wrist_camera_pose.position.y = float(res["cam_pos"][1])
            obs_msg.wrist_camera_pose.position.z = float(res["cam_pos"][2])
            obs_msg.wrist_camera_pose.orientation.x = float(res["cam_quat_xyzw"][0])
            obs_msg.wrist_camera_pose.orientation.y = float(res["cam_quat_xyzw"][1])
            obs_msg.wrist_camera_pose.orientation.z = float(res["cam_quat_xyzw"][2])
            obs_msg.wrist_camera_pose.orientation.w = float(res["cam_quat_xyzw"][3])

            obs_msg.arm_target_pose.position.x = float(res["ee_pos"][0])
            obs_msg.arm_target_pose.position.y = float(res["ee_pos"][1])
            obs_msg.arm_target_pose.position.z = float(res["ee_pos"][2])
            obs_msg.arm_target_pose.orientation.x = float(res["ee_quat_xyzw"][0])
            obs_msg.arm_target_pose.orientation.y = float(res["ee_quat_xyzw"][1])
            obs_msg.arm_target_pose.orientation.z = float(res["ee_quat_xyzw"][2])
            obs_msg.arm_target_pose.orientation.w = float(res["ee_quat_xyzw"][3])
        except Exception as e:
            self.get_logger().debug(f'compute_isometric_view error: {e}')

        self.pub_observation_target.publish(obs_msg)

        # F. In Mode C (direct_head): Immediately publish MultiViewSnapshot
        if self.f2g_mode == 'direct_head':
            self.emit_snapshot(
                fused_points=points_base,
                obstacles=obstacles_base,
                pose=pose_base,
                dimensions=size_m,
                source_id="head_f2g_direct",
                stamp=src_header.stamp,
                camera_pose=camera_pose_base,
            )

    def _clear_wrist_output(self, stamp):
        header = std_msgs.msg.Header(frame_id=self.base_frame, stamp=stamp)
        self.pub_fused_cloud.publish(points_to_pointcloud2(np.zeros((0, 3), dtype=np.float32), header))
        self.emit_snapshot(np.zeros((0, 3), dtype=np.float32), source_id='invalidated', stamp=stamp)
        self.latest_wrist_prediction = None

    def _fusion_status(self, context, state, reason='', **details):
        payload = dict(schema_version=1, source='wrist', status=state, message=reason,
            stamp_ns=stamp_ns(context.acquisition_header.stamp),
            frame_id=context.acquisition_header.frame_id, target=context.class_label or self.target_label,
            perception_epoch=context.perception_epoch, perception_generation=context.perception_generation,
            target_generation=context.target_generation, acquisition_id=context.acquisition_id,
            instance_id=context.instance_id, session_uuid=self.session_uuid,
            snapshot_generation=self.snapshot_generation, emitted_ns=self.get_clock().now().nanoseconds)
        payload.update(details)
        self.pub_fusion_status.publish(std_msgs.msg.String(data=json.dumps(payload, allow_nan=False)))
        if state == 'failed':
            self.get_logger().warn(f'Wrist F2G acquisition {context.acquisition_id} failed: {reason}')

    def wrist_selection_callback(self, packet):
        if self.f2g_mode != 'fusion':
            return
        if self.wrist_selection.observe(packet.target, invalidated=packet.invalidated) and packet.invalidated:
            self._clear_wrist_output(packet.target.acquisition_header.stamp)
            self._fusion_status(packet.target, 'invalidated', packet.target.quality_message)

    def _run_wrist_f2g(self, acquisition, T_base_cam):
        if not self.get_parameter('wrist_f2g_enabled').value:
            raise ValueError('Wrist F2G is disabled; no completed wrist shape can be produced')
        color, depth_msg, info = acquisition.color, acquisition.depth, acquisition.camera_info
        mask = self.bridge.imgmsg_to_cv2(acquisition.selected_mask, desired_encoding='mono8')
        if np.count_nonzero(mask) < 20:
            raise ValueError('Selected wrist mask has fewer than 20 usable pixels')
        rgb = self.bridge.imgmsg_to_cv2(color, desired_encoding='rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        K = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        if len(info.d) and not np.allclose(info.d, 0):
            maps = cv2.initUndistortRectifyMap(K, np.asarray(info.d), None, K,
                                             (color.width, color.height), cv2.CV_32FC1)
            rgb = cv2.remap(rgb, *maps, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, *maps, cv2.INTER_NEAREST)
            mask = cv2.remap(mask, *maps, cv2.INTER_NEAREST)
        if self.estimator is None:
            self.init_estimator()
        started = time.perf_counter()
        result = self._predict(rgb=rgb, depth=depth, K=K, mask=mask,
            depth_scale=.001 if depth_msg.encoding == '16UC1' else 1., return_shape=True)
        points = np.asarray(result.get('points_cam_m', []), dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 10 or not np.isfinite(points).all():
            raise ValueError('Wrist F2G returned an empty or nonfinite completed shape')
        transform = np.asarray(result['T_cam_obj'])
        dimensions = np.asarray(result['size_m'])
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.min(dimensions) <= 0):
            raise ValueError('Wrist F2G returned invalid object geometry')
        points_base = transform_points(points, T_base_cam)
        pose = matrix_to_pose(T_base_cam @ transform)
        header = std_msgs.msg.Header(frame_id=self.base_frame, stamp=color.header.stamp)
        self.pub_completed_cloud.publish(points_to_pointcloud2(points_base, header))
        self.pub_object_pose.publish(geometry_msgs.msg.PoseStamped(header=header, pose=pose))
        dim = geometry_msgs.msg.Vector3Stamped(header=header)
        dim.vector.x, dim.vector.y, dim.vector.z = map(float, dimensions)
        self.pub_object_dimensions.publish(dim)
        self.pub_bbox_marker.publish(create_bbox_marker(pose=pose, dimensions=dimensions,
            frame_id=self.base_frame, stamp=header, color=(.1, .9, .2, .5)))
        self.get_logger().info(f'Arm F2G completed {len(points_base)} points in {time.perf_counter()-started:.2f}s')
        return dict(points=points_base, pose=pose, dimensions=dimensions, K=K)

    def wrist_acquisition_callback(self, acquisition):
        if self.f2g_mode != 'fusion':
            return
        context = acquisition.target.target
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self.last_wrist_clock:
            self.wrist_selection.reset()
            self._clear_wrist_output(self.get_clock().now().to_msg())
        self.last_wrist_clock = now_ns
        if (context.class_label and context.class_label.casefold() != self.target_label.casefold()) or not self.wrist_selection.can_process(context):
            return
        try:
            validate_acquisition(acquisition, self.target_label, now_ns,
                self.get_parameter('max_acquisition_age').value,
                self.get_parameter('max_acquisition_bytes').value)
            if not self.wrist_selection.observe(context, processing=True):
                return
            self._clear_wrist_output(context.acquisition_header.stamp)
            self._fusion_status(context, 'processing')
            points, obstacles = pointcloud2_to_labeled_points(acquisition.target.points)
            if len(points) < 10:
                raise ValueError(f'Insufficient selected wrist points ({len(points)} < 10)')
            # Historical transforms must already exist; never wait in this
            # callback or substitute the latest moving-camera transform.
            def camera_transform(frame):
                if frame == self.base_frame:
                    return np.eye(4)
                tf = self.tf_buffer.lookup_transform(self.base_frame, frame,
                    rclpy.time.Time.from_msg(context.acquisition_header.stamp))
                return transform_stamped_to_matrix(tf)
            T_base_cam = camera_transform(acquisition.color.header.frame_id)
            T_base_cloud = camera_transform(acquisition.target.points.header.frame_id)
            points = transform_points(points, T_base_cloud)
            obstacles = transform_points(obstacles, T_base_cloud)
            prediction = self._run_wrist_f2g(acquisition, T_base_cam)
            diagnostics = {}
            names = dict(voxel_size='voxel_size', overlap_distance='overlap_distance',
                max_rotation_deg='max_rotation_deg', max_centroid_shift='max_centroid_shift',
                max_point_shift='max_point_shift', max_alignment_distance='max_alignment_distance',
                min_alignment_fraction='min_alignment_fraction',
                visibility_radius_pixels='visibility_radius_pixels',
                visibility_depth_tolerance='visibility_depth_tolerance')
            settings = {key: self.get_parameter('fusion_' + param).value for key, param in names.items()}
            fused = align_and_fuse_shapes(prediction['points'], points, **settings,
                enable_icp_refinement=self.get_parameter('fusion_enable_icp').value,
                camera_from_base=np.linalg.inv(T_base_cam), camera_matrix=prediction['K'], diagnostics=diagnostics)
            if len(fused) < 10:
                raise ValueError('Fusion produced fewer than 10 points')
            self.latest_wrist_prediction = prediction
            header = std_msgs.msg.Header(frame_id=self.base_frame, stamp=acquisition.color.header.stamp)
            self.pub_fused_cloud.publish(points_to_pointcloud2(fused, header))
            snapshot = self.emit_snapshot(
                fused,
                obstacles=obstacles,
                pose=prediction['pose'],
                dimensions=prediction['dimensions'],
                source_id='wrist_f2g_cloud_fusion',
                stamp=header.stamp,
                camera_pose=matrix_to_pose(T_base_cam),
            )
            self._fusion_status(context, 'ready', target_points=len(points), fused_points=len(fused),
                                fusion=diagnostics)
            directory = self.get_parameter('capture_dir').value
            if directory:
                output = Path(directory).expanduser() / self.session_uuid / str(snapshot.snapshot_generation)
                try:
                    output.mkdir(parents=True, exist_ok=False)
                    from rclpy.serialization import serialize_message
                    (output / 'acquisition.cdr').write_bytes(serialize_message(acquisition))
                    np.savez_compressed(output / 'fusion.npz', completed=prediction['points'],
                        observed=points, obstacles=obstacles, fused=fused, camera_from_base=np.linalg.inv(T_base_cam),
                        camera_matrix=prediction['K'])
                    (output / 'manifest.json').write_text(json.dumps(dict(session_uuid=self.session_uuid,
                        snapshot_generation=snapshot.snapshot_generation, acquisition_id=context.acquisition_id,
                        perception_epoch=context.perception_epoch, target=context.class_label,
                        stamp_ns=stamp_ns(header.stamp), fusion=diagnostics), indent=2))
                except Exception as exc:
                    self.get_logger().warn(f'Could not save wrist fusion capture: {exc}')
        except Exception as exc:
            self._clear_wrist_output(context.acquisition_header.stamp)
            self._fusion_status(context, 'failed', str(exc))

    def emit_snapshot(self, fused_points, obstacles=None, pose=None, dimensions=None, source_id="f2g_pose", stamp=None, camera_pose=None):
        """Construct and publish a standard MultiViewSnapshot message."""
        self.snapshot_generation += 1
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        snap = MultiViewSnapshot()
        snap.header.stamp = stamp
        snap.header.frame_id = self.base_frame
        snap.session_uuid = self.session_uuid
        snap.snapshot_generation = self.snapshot_generation
        snap.target_label = self.target_label or "object"
        snap.static_scene_verified = False

        # Fused target cloud
        base_header = std_msgs.msg.Header()
        base_header.stamp = stamp
        base_header.frame_id = self.base_frame
        snap.fused_target_cloud = points_to_pointcloud2(fused_points, base_header)

        # Scene obstacles
        if obstacles is not None and len(obstacles) > 0:
            snap.scene_obstacles = points_to_pointcloud2(obstacles, base_header)
        else:
            snap.scene_obstacles = points_to_pointcloud2(np.zeros((0, 3), dtype=np.float32), base_header)

        # Centroid
        if pose is not None:
            snap.centroid.x = float(pose.position.x)
            snap.centroid.y = float(pose.position.y)
            snap.centroid.z = float(pose.position.z)
        elif len(fused_points) > 0:
            c = np.mean(fused_points, axis=0)
            snap.centroid.x = float(c[0])
            snap.centroid.y = float(c[1])
            snap.centroid.z = float(c[2])

        # Bounds min and max
        if len(fused_points) > 0:
            bmin = np.min(fused_points, axis=0)
            bmax = np.max(fused_points, axis=0)
            snap.bounds_min.x = float(bmin[0])
            snap.bounds_min.y = float(bmin[1])
            snap.bounds_min.z = float(bmin[2])
            snap.bounds_max.x = float(bmax[0])
            snap.bounds_max.y = float(bmax[1])
            snap.bounds_max.z = float(bmax[2])

        # Observations
        obs = ObservationRef()
        obs.header.stamp = stamp
        obs.header.frame_id = self.base_frame
        obs.source_id = source_id
        obs.acquisition_id = self.snapshot_generation
        if camera_pose is not None:
            obs.camera_pose = camera_pose
        elif pose is not None:
            obs.camera_pose = pose
        obs.valid_depth_fraction = 1.0
        obs.registration_residual = float('nan')
        snap.observations.append(obs)

        self.pub_snapshot.publish(snap)
        self.get_logger().info(
            f'MultiViewSnapshot gen={self.snapshot_generation} published on /perc/multiview/snapshot ({len(fused_points)} points)'
        )
        return snap


def main(args=None):
    rclpy.init(args=args)
    node = F2GPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
