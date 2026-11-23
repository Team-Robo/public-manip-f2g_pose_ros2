"""Exercise camera pinning and the actual predictor adapter without DDS/GPU."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from f2g_pose_ros2.object_perception_node import ObjectPerceptionNode


def header(sec=10, nano=0):
    h = Header(frame_id='camera_color_optical_frame')
    h.stamp.sec, h.stamp.nanosec = sec, nano
    return h


def predictor(monkeypatch, enabled=False):
    rgb = CvBridge().cv2_to_imgmsg(np.zeros((8,8,3), np.uint8), 'rgb8')
    depth = CvBridge().cv2_to_imgmsg(np.arange(64,dtype=np.uint16).reshape(8,8)+1000, '16UC1')
    mask = CvBridge().cv2_to_imgmsg(np.ones((8,8),np.uint8)*255, 'mono8')
    rgb.header=header();depth.header=header();mask.header=header()
    info=CameraInfo(header=header(),height=8,width=8,k=[100.,0.,3.5,0.,100.,3.5,0.,0.,1.])
    monkeypatch.setitem(sys.modules,'f2g_pose_ros2.cloud_utils', NS(
        transform_stamped_to_matrix=lambda tf: np.eye(4), matrix_to_pose=lambda T: Pose()))
    points=np.array([[x,y,z] for x in (-.02,0,.02) for y in (-.03,0,.03) for z in (.5,.52,.54)])
    estimator=Mock()
    T=np.eye(4);T[2,3]=.52
    estimator.predict.return_value=dict(points_cam_m=points,T_cam_obj=T,size_m=np.array([.04,.06,.04]))
    node=NS(bridge=CvBridge(), info={'head':info}, depth_info={'head':info}, enabled={'head':enabled},
        estimator=estimator, observed_views={}, max_age=180., frame='base_link',tf=Mock(),
        get_clock=lambda: NS(now=lambda: NS(nanoseconds=11*10**9)),get_logger=lambda:Mock())
    return node,mask,rgb,depth


def test_disabled_f2g_uses_measured_points_and_does_not_call_model(monkeypatch):
    node,mask,rgb,depth=predictor(monkeypatch)
    geometry,ref=ObjectPerceptionNode.predict(node,'head',mask,rgb,depth)
    node.estimator.predict.assert_not_called()
    assert not geometry.completed and geometry.stamp_ns==10**10
    assert len(geometry.points)>10 and ref.source_id=='head_depth'
    assert ref.valid_depth_fraction==1.
    assert np.isnan(ref.registration_residual)


def test_enabled_f2g_returns_model_pose_dimensions_and_cloud(monkeypatch):
    node,mask,rgb,depth=predictor(monkeypatch,True)
    geometry,ref=ObjectPerceptionNode.predict(node,'head',mask,rgb,depth)
    assert geometry.completed and ref.source_id=='head_f2g'
    np.testing.assert_allclose(geometry.model_dimensions,[.04,.06,.04])
    assert node.estimator.predict.call_args.kwargs['depth_scale']==1.


def test_unregistered_depth_is_rejected_before_inference(monkeypatch):
    node,mask,rgb,depth=predictor(monkeypatch,True)
    depth.header.frame_id='camera_depth_optical_frame'
    with pytest.raises(ValueError,match='calibration do not match'):
        ObjectPerceptionNode.predict(node,'head',mask,rgb,depth)
    node.estimator.predict.assert_not_called()


def test_pinned_frame_survives_streaming_buffer_eviction():
    from collections import deque
    rgb,depth=NS(header=header()), NS(header=header(nano=10_000_000))
    node=NS(headers={'head':header()}, buffers={'head':{'rgb':deque([rgb]),'depth':deque([depth])}},
            pinned={},pin_parts={},skew=.05)
    ObjectPerceptionNode.try_pin(node,'head')
    node.buffers['head']['rgb'].clear()
    ObjectPerceptionNode.try_pin(node,'head')
    assert node.pinned['head']==(rgb,depth)


def test_mismatched_timestamp_is_never_substituted_with_latest_rgb():
    from collections import deque
    node=NS(headers={'head':header()}, buffers={'head':{'rgb':deque([NS(header=header(11))]),
                'depth':deque([NS(header=header(11))])}},pinned={},pin_parts={},skew=.05)
    ObjectPerceptionNode.try_pin(node,'head')
    assert not node.pinned


def test_camera_transform_uses_original_capture_time_after_slow_segmentation(monkeypatch):
    node,mask,rgb,depth=predictor(monkeypatch)
    node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=100*10**9))
    geometry,_=ObjectPerceptionNode.predict(node,'head',mask,rgb,depth)
    assert geometry.stamp_ns==10*10**9
    assert node.tf.lookup_transform.call_args.args[2].nanoseconds==10*10**9


@pytest.mark.parametrize('preload', [False, True])
@pytest.mark.parametrize('mode,views', [('head', {'head','fused'}),
                                      ('wrist', {'wrist','fused'}),
                                      ('head_then_wrist', {'head','wrist','fused'})])
def test_constructor_preserves_ros_node_properties(monkeypatch,mode,views,preload):
    """Run the complete constructor with Node's real property descriptors intact."""
    import f2g_pose_ros2.object_perception_node as module
    from rclpy.node import Node
    parameters={}
    monkeypatch.setattr(Node,'__init__',lambda self,*a,**kw:setattr(self,'_publishers',[]))
    def declare(self,name,value):
        parameters[name] = mode if name == 'camera_mode' else preload if name == 'preload_models' else value
    monkeypatch.setattr(Node,'declare_parameter',declare)
    monkeypatch.setattr(Node,'get_parameter',lambda self,name:NS(value=parameters[name]))
    monkeypatch.setattr(Node,'create_publisher',Mock())
    monkeypatch.setattr(Node,'create_subscription',Mock())
    monkeypatch.setattr(Node,'create_timer',Mock())
    monkeypatch.setattr(Node,'get_logger',lambda self:Mock())
    monkeypatch.setattr(module,'Buffer',Mock())
    monkeypatch.setattr(module,'TransformListener',Mock())
    loader = Mock()
    monkeypatch.setattr(ObjectPerceptionNode, "init_estimator", loader)
    node=ObjectPerceptionNode()
    assert loader.call_count == int(preload and mode != "wrist")
    assert set(node.view_publishers)==views
    assert list(node.publishers)==[]  # Still ROS Node's publisher iterator.
    assert node.estimator is None  # Startup doesn't allocate a model.


def test_depth_pin_survives_until_reliable_sam_rgb_copy_arrives():
    from collections import deque
    original=header()
    depth=NS(header=header(nano=10_000_000))
    node=NS(headers={'head':original}, buffers={'head':{'rgb':deque(), 'depth':deque([depth])}},
            pin_parts={},pinned={},skew=.05)
    ObjectPerceptionNode.try_pin(node,'head')
    assert node.pin_parts['head']['depth'] is depth and not node.pinned
    node.buffers['head']['depth'].clear()
    exact_rgb=NS(header=original)
    node.buffers['head']['rgb'].append(exact_rgb)
    ObjectPerceptionNode.try_pin(node,'head')
    assert node.pinned['head']==(exact_rgb,depth)


@pytest.mark.parametrize('source,method', [('wrist','warn'),('head','warn'),('fused','error')])
def test_optional_fusion_failure_keeps_source_specific_status(monkeypatch,source,method):
    import json
    logger=Mock();status=Mock()
    node=NS(source=source,get_logger=lambda:logger,status_pub=status,
            get_clock=lambda:NS(now=lambda:NS(nanoseconds=100)))
    ObjectPerceptionNode.error(node,'Capture separation exceeds limit','fused')
    getattr(logger,method).assert_called_once()
    payload=json.loads(status.publish.call_args.args[0].data)
    assert payload['source']=='fused' and payload['status']=='failed'


@pytest.mark.parametrize('interruption,active', [(KeyboardInterrupt(),False),
    (RuntimeError('take_message invalid during shutdown'),False),
    (RuntimeError('real processing failure'),True)])
def test_shutdown_is_idempotent_without_hiding_active_runtime_errors(monkeypatch,interruption,active):
    import f2g_pose_ros2.object_perception_node as module
    node=Mock()
    monkeypatch.setattr(module,'ObjectPerceptionNode',lambda:node)
    monkeypatch.setattr(module.rclpy,'init',Mock())
    monkeypatch.setattr(module.rclpy,'spin',Mock(side_effect=interruption))
    monkeypatch.setattr(module.rclpy,'ok',lambda:active)
    cleanup=Mock();monkeypatch.setattr(module.rclpy,'try_shutdown',cleanup)
    if active:
        with pytest.raises(RuntimeError,match='real processing'):
            module.main()
    else:
        module.main()
    node.destroy_node.assert_called_once()
    cleanup.assert_called_once()


def test_measured_baseline_is_preserved_when_f2g_replaces_grasp_cloud(monkeypatch):
    node,mask,rgb,depth=predictor(monkeypatch,True)
    completed,_=ObjectPerceptionNode.predict(node,'head',mask,rgb,depth)
    observed=node.observed_views['head']
    assert completed.completed and not observed.completed
    assert observed.stamp_ns==completed.stamp_ns
    assert not np.allclose(observed.center,completed.center)
    assert len(observed.points)>10


def test_comparison_clouds_are_published_in_same_frame_with_same_acquisition(monkeypatch):
    node,mask,rgb,depth=predictor(monkeypatch,True)
    completed,_=ObjectPerceptionNode.predict(node,'head',mask,rgb,depth)
    utils=sys.modules['f2g_pose_ros2.cloud_utils']
    monkeypatch.setattr(utils,'points_to_pointcloud2',lambda points,h:NS(header=h,points=points),raising=False)
    pubs=(Mock(),Mock(),Mock())
    node.view_publishers={'head':pubs};node.observed_publishers={'head':Mock()}
    node.header=lambda g:ObjectPerceptionNode.header(node,g)
    ObjectPerceptionNode.publish_view(node,'head',completed)
    output=pubs[0].publish.call_args.args[0]
    baseline=node.observed_publishers['head'].publish.call_args.args[0]
    assert output.header==baseline.header
    assert output.header.frame_id=='base_link' and output.header.stamp.sec==10
    np.testing.assert_allclose(baseline.points,node.observed_views['head'].points)


def test_strided_obstacle_projection_matches_full_resolution_index_grid(monkeypatch):
    from f2g_pose_ros2.object_geometry import voxelize
    node,_,_,_=predictor(monkeypatch)
    h,w=72,96
    K=np.array([[100.,0,47.5],[0,100.,35.5],[0,0,1.]])
    info=CameraInfo(header=header(),height=h,width=w,k=K.reshape(-1).tolist())
    node.info['head']=info;node.depth_info['head']=info
    color=CvBridge().cv2_to_imgmsg(np.zeros((h,w,3),np.uint8),'rgb8')
    values=np.random.default_rng(74).integers(500,1500,(h,w),dtype=np.uint16)
    values[::9,::7]=0
    depth=CvBridge().cv2_to_imgmsg(values,'16UC1')
    selected=np.zeros((h,w),np.uint8);selected[20:50,30:65]=255
    mask=CvBridge().cv2_to_imgmsg(selected,'mono8')
    for message in (color,depth,mask):message.header=header()
    actual,_=ObjectPerceptionNode.predict(node,'head',mask,color,depth)
    z=values.astype(np.float32)*.001
    valid=np.isfinite(z)&(z>.05)&(z<3.5)
    yy,xx=np.indices(z.shape)
    y,x=np.nonzero(valid&~selected.astype(bool)&(yy%8==0)&(xx%8==0))
    expected=np.column_stack(((x-K[0,2])*z[y,x]/K[0,0],(y-K[1,2])*z[y,x]/K[1,1],z[y,x]))
    np.testing.assert_array_equal(actual.obstacles,voxelize(expected,.012))
