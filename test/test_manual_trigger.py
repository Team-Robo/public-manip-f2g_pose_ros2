"""Startup policy must not disable explicitly requested F2G predictions."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

from f2g_pose_ros2.f2g_pose_node import F2GPoseNode


def test_disabled_auto_trigger_never_requests_startup_segmentation():
    node = NS(auto_trigger=False, _startup_triggered=False,
              startup_timer=Mock(), head_trigger_client=Mock())
    F2GPoseNode._startup_trigger_check(node)
    node.startup_timer.cancel.assert_called_once()
    node.head_trigger_client.call_async.assert_not_called()


def test_enabled_auto_trigger_requests_segmentation_only_once():
    node = NS(auto_trigger=True, _startup_triggered=False, startup_timer=Mock(),
              head_trigger_client=Mock(), latest_color=object(), latest_depth=object(),
              latest_info=object(), target_label='cup', get_logger=Mock())
    F2GPoseNode._startup_trigger_check(node)
    F2GPoseNode._startup_trigger_check(node)
    assert node._startup_triggered
    node.head_trigger_client.call_async.assert_called_once()


def test_manual_trigger_submits_request_even_with_auto_trigger_disabled():
    node = NS(auto_trigger=False, head_trigger_client=Mock())
    response = NS()
    F2GPoseNode.handle_trigger_srv(node, None, response)
    assert response.success and 'requested' in response.message
    node.head_trigger_client.call_async.assert_called_once()


def test_manual_trigger_reports_unavailable_segmentation_service():
    client = Mock()
    client.service_is_ready.return_value = False
    response = NS()
    F2GPoseNode.handle_trigger_srv(NS(head_trigger_client=client), None, response)
    assert not response.success and 'not ready' in response.message
    client.call_async.assert_not_called()


def test_manually_received_mask_reaches_f2g_processing_with_auto_trigger_disabled():
    resolve_frame = Mock(return_value='')
    node = NS(auto_trigger=False, _resolve_cam_frame=resolve_frame,
              base_frame='base_link', get_logger=Mock())
    F2GPoseNode.head_observation_callback(
        node, NS(header=NS(frame_id='camera')), None, None, None)
    resolve_frame.assert_called_once_with('camera')
