"""Exercise F2G's real acquisition adapter with an injected CPU predictor."""
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock
import json
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from core_r1_interfaces.msg import TargetAcquisition, TargetCloud, TargetContext
from f2g_pose_ros2.acquisition import SelectionGuard
from f2g_pose_ros2.f2g_pose_node import F2GPoseNode


def make_adapter():
    bridge = CvBridge()
    header = Header(frame_id='wrist_optical')
    header.stamp.sec = 11
    color = bridge.cv2_to_imgmsg(np.full((32,32,3), 17, np.uint8), 'rgb8')
    depth = bridge.cv2_to_imgmsg(np.full((32,32), 400, np.uint16), '16UC1')
    mask = bridge.cv2_to_imgmsg(np.full((32,32), 255, np.uint8), 'mono8')
    color.header = depth.header = mask.header = header
    info = CameraInfo(header=header, width=32, height=32, k=[400.,0.,15.5,0.,400.,15.5,0.,0.,1.])
    x, y = np.meshgrid((np.arange(32)-15.5)*.001, (np.arange(32)-15.5)*.001)
    front = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .4))).astype(np.float32)
    completed = np.vstack((front, front+[0,0,.004])).astype(np.float32)
    labeled = np.column_stack((front, np.ones(len(front)))).tolist() + [[.1,.1,.4,0.]]
    fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
              for i,name in enumerate(('x','y','z','intensity'))]
    cloud = point_cloud2.create_cloud(header, fields, labeled)
    context = TargetContext(acquisition_header=header, geometry_header=header,
        source_id='/camera/left_d405/color/image_raw', class_label='cup', instance_validated=True,
        instance_id='cup-1', perception_epoch='selection-1', perception_generation=1,
        target_generation=1, acquisition_id=1, final_target_points=len(front))
    packet = TargetAcquisition(target=TargetCloud(points=cloud, target=context), color=color,
        depth=depth, selected_mask=mask, camera_info=info)
    parameters = dict(wrist_f2g_enabled=True, max_acquisition_age=180., max_acquisition_bytes=64*1024*1024,
        fusion_voxel_size=.001, fusion_overlap_distance=.012, fusion_max_rotation_deg=5.,
        fusion_max_centroid_shift=.01, fusion_max_point_shift=.015, fusion_max_alignment_distance=.03,
        fusion_min_alignment_fraction=.3, fusion_visibility_radius_pixels=8.,
        fusion_visibility_depth_tolerance=.0005, fusion_enable_icp=False, capture_dir='')
    transform = TransformStamped()
    transform.transform.rotation.w = 1.
    T = np.eye(4); T[2,3] = .402
    predictor = Mock()
    predictor.predict.return_value = dict(points_cam_m=completed, T_cam_obj=T, size_m=np.array([.031,.031,.004]))
    node = NS(f2g_mode='fusion', target_label='cup', base_frame='base_link', wrist_selection=SelectionGuard(),
        last_wrist_clock=12*10**9, session_uuid='fusion-session', snapshot_generation=0,
        bridge=bridge, estimator=predictor, tf_buffer=Mock(),
        get_clock=lambda: NS(now=lambda:NS(nanoseconds=12*10**9)), get_logger=Mock(),
        get_parameter=lambda name: NS(value=parameters[name]), latest_wrist_prediction=None,
        latest_completed_cloud_base=np.full((10,3), 100.))
    node.tf_buffer.lookup_transform.return_value = transform
    for name in ('pub_fused_cloud','pub_snapshot','pub_fusion_status','pub_completed_cloud',
                 'pub_object_pose','pub_object_dimensions','pub_bbox_marker'):
        setattr(node, name, Mock())
    for name in ('_clear_wrist_output','emit_snapshot','_fusion_status','_run_wrist_f2g','_predict'):
        setattr(node, name, MethodType(getattr(F2GPoseNode, name), node))
    return node, packet


def status(node):
    return json.loads(node.pub_fusion_status.publish.call_args.args[0].data)


def test_atomic_adapter_uses_captured_rgbd_and_keeps_original_header():
    node, packet = make_adapter()
    F2GPoseNode.wrist_acquisition_callback(node, packet)
    assert status(node)['status'] == 'ready', status(node)
    kwargs = node.estimator.predict.call_args.kwargs
    assert np.all(kwargs['rgb'] == 17) and np.all(kwargs['depth'] == 400)
    assert kwargs['depth_scale'] == .001
    snapshots = [c.args[0] for c in node.pub_snapshot.publish.call_args_list if c.args[0].fused_target_cloud.width]
    assert len(snapshots) == 1
    assert snapshots[0].header.stamp == packet.color.header.stamp
    assert snapshots[0].header.frame_id == 'base_link'
    # Phase 2A invariant: snap.centroid is the object center (z=0.402), while
    # obs.camera_pose holds the wrist camera pose T_base_cam (z=0.0).
    assert abs(snapshots[0].centroid.z - 0.402) < 1e-4
    assert len(snapshots[0].observations) == 1
    assert abs(snapshots[0].observations[0].camera_pose.position.z - 0.0) < 1e-4
    assert status(node)['perception_epoch'] == 'selection-1'
    assert status(node)['acquisition_id'] == 1
    assert np.max(node.latest_wrist_prediction['points']) < 1.
    for call in node.tf_buffer.lookup_transform.call_args_list:
        assert call.args[2].nanoseconds == 11*10**9
    count = node.pub_snapshot.publish.call_count
    F2GPoseNode.wrist_acquisition_callback(node, packet)
    assert node.pub_snapshot.publish.call_count == count
    node.estimator.predict.assert_called_once()


def test_missing_historical_tf_reports_failure_without_prediction_or_latest_tf():
    node, packet = make_adapter()
    node.tf_buffer.lookup_transform.side_effect = RuntimeError('historical TF unavailable')
    F2GPoseNode.wrist_acquisition_callback(node, packet)
    assert status(node)['status'] == 'failed' and 'historical TF' in status(node)['message']
    node.estimator.predict.assert_not_called()
    assert node.tf_buffer.lookup_transform.call_count == 1
    assert not node.pub_snapshot.publish.call_args.args[0].fused_target_cloud.width


def test_empty_selected_mask_never_reuses_a_previous_head_prediction():
    node, packet = make_adapter()
    packet.selected_mask.data = bytes(32*32)
    F2GPoseNode.wrist_acquisition_callback(node, packet)
    assert status(node)['status'] == 'failed' and '20 usable pixels' in status(node)['message']
    node.estimator.predict.assert_not_called()
    assert node.latest_wrist_prediction is None


def test_invalidation_prevents_late_acquisition_from_clearing_or_republishing():
    node, packet = make_adapter()
    F2GPoseNode.wrist_acquisition_callback(node, packet)
    context = packet.target.target
    context.acquisition_id = 2
    invalidation = TargetCloud(target=context, invalidated=True)
    F2GPoseNode.wrist_selection_callback(node, invalidation)
    count = node.pub_snapshot.publish.call_count
    context.acquisition_id = 1
    F2GPoseNode.wrist_acquisition_callback(node, packet)
    assert node.pub_snapshot.publish.call_count == count
    assert status(node)['status'] == 'invalidated'


def test_missing_selected_instance_reports_failure_for_the_requested_capture():
    node, packet = make_adapter()
    packet.target.target.class_label = ''
    packet.target.target.instance_id = ''
    packet.target.target.instance_validated = False
    F2GPoseNode.wrist_acquisition_callback(node, packet)
    assert status(node)['status'] == 'failed' and status(node)['target'] == 'cup'
    node.estimator.predict.assert_not_called()


def test_capture_archive_keeps_original_bytes_and_fusion_inputs(tmp_path):
    node, packet = make_adapter()
    original_parameter = node.get_parameter
    node.get_parameter = lambda name: NS(value=str(tmp_path)) if name == 'capture_dir' else original_parameter(name)
    F2GPoseNode.wrist_acquisition_callback(node, packet)
    from rclpy.serialization import deserialize_message
    files = list(tmp_path.rglob('acquisition.cdr'))
    assert len(files) == 1
    saved = deserialize_message(files[0].read_bytes(), TargetAcquisition)
    assert saved == packet
    from f2g_pose_ros2.audit_fusion import compare
    result = compare(files[0].with_name('fusion.npz'))
    assert result['icp_enabled']['status'] == result['icp_disabled']['status'] == 'ok'
    assert result['physical_accuracy_verified'] is False


def test_direct_head_populates_single_camera_obstacles_and_camera_pose():
    node, packet = make_adapter()
    node.f2g_mode = 'direct_head'
    node.depth_scale = 0.001
    node.standoff_dist = 0.50
    node.elevation_deg = 48.0
    node.azimuth_deg = 25.0
    node.arm_side = 'left'
    node.pub_observation_target = Mock()
    T_base_cam = np.eye(4, dtype=np.float32)
    T_base_cam[2, 3] = 1.2
    node._resolve_cam_frame = lambda frame: frame
    node._lookup_T_base_cam = lambda frame, stamp: T_base_cam
    node.publish_head_estimates = MethodType(F2GPoseNode.publish_head_estimates, node)
    node._extract_head_obstacles = F2GPoseNode._extract_head_obstacles

    mask_arr = np.zeros((32, 32), dtype=np.uint8)
    mask_arr[12:20, 12:20] = 255
    mask_msg = node.bridge.cv2_to_imgmsg(mask_arr, 'mono8')
    mask_msg.header = packet.color.header
    depth_arr = np.full((32, 32), 800, dtype=np.uint16)
    depth_msg = node.bridge.cv2_to_imgmsg(depth_arr, '16UC1')
    depth_msg.header = packet.color.header
    T_cam_obj = np.eye(4, dtype=np.float32)
    T_cam_obj[2, 3] = 0.8
    x, y = np.meshgrid((np.arange(12, 20) - 15.5) * 0.002, (np.arange(12, 20) - 15.5) * 0.002)
    obj_pts = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.8))).astype(np.float32)
    node.estimator.predict.return_value = dict(
        points_cam_m=obj_pts,
        T_cam_obj=T_cam_obj,
        size_m=np.array([0.04, 0.04, 0.04], dtype=np.float32),
    )

    F2GPoseNode.head_observation_callback(node, packet.color, depth_msg, packet.camera_info, mask_msg)
    snapshots = [c.args[0] for c in node.pub_snapshot.publish.call_args_list if c.args[0].fused_target_cloud.width]
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.scene_obstacles.width > 0
    assert abs(snap.centroid.z - 2.0) < 1e-4
    assert len(snap.observations) == 1
    assert snap.observations[0].source_id == 'head_f2g_direct'
    assert abs(snap.observations[0].camera_pose.position.z - 1.2) < 1e-4
