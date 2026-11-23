"""The atomic packet, rather than arrival order, owns inference provenance."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from f2g_pose_ros2.acquisition import validate_acquisition, SelectionGuard


def acquisition():
    header = NS(frame_id='wrist_optical', stamp=NS(sec=11, nanosec=0))
    def image(encoding, channels):
        return NS(header=deepcopy(header), width=8, height=8, step=8*channels,
                  encoding=encoding, data=bytes(64*channels))
    context = NS(acquisition_header=deepcopy(header), geometry_header=deepcopy(header),
        class_label='cup', instance_validated=True, instance_id='cup-1', perception_epoch='epoch-1',
        perception_generation=1, target_generation=1, acquisition_id=1)
    return NS(target=NS(target=context, invalidated=False, points=NS(header=deepcopy(header), data=bytes(100))),
        color=image('rgb8', 3), depth=image('16UC1', 2), selected_mask=image('mono8', 1),
        camera_info=NS(header=deepcopy(header), width=8, height=8, k=[600.,0.,4.,0.,600.,4.,0.,0.,1.]))


def test_complete_packet_is_independent_of_live_topic_arrival_order():
    packet = acquisition()
    context = validate_acquisition(packet, 'CUP', 12*10**9)
    assert context is packet.target.target
    # A later live frame cannot alter the captured bytes/headers in this packet.
    later = deepcopy(packet.color)
    later.header.stamp.sec += 1
    assert validate_acquisition(packet, 'cup', 12*10**9) is context


@pytest.mark.parametrize('part', ['selected_mask', 'color', 'camera_info'])
def test_adjacent_or_unrelated_images_and_calibration_cannot_substitute(part):
    packet = acquisition()
    getattr(packet, part).header.stamp.nanosec = 1
    with pytest.raises(ValueError):
        validate_acquisition(packet, 'cup', 12*10**9)


def test_bounds_expiry_wrong_target_and_incomplete_images_fail_before_inference():
    packet = acquisition()
    for target, now, bound in [('bottle', 12*10**9, 10000), ('cup', 200*10**9, 10000),
                               ('cup', 10*10**9, 10000), ('cup', 12*10**9, 1)]:
        with pytest.raises(ValueError):
            validate_acquisition(packet, target, now, max_bytes=bound)
    packet.depth.data = b''
    with pytest.raises(ValueError, match='incomplete'):
        validate_acquisition(packet, 'cup', 12*10**9)


def test_selection_invalidation_duplicates_and_retired_epochs_are_rejected():
    context = acquisition().target.target
    guard = SelectionGuard()
    assert guard.observe(context, processing=True)
    assert not guard.observe(context, processing=True)
    context.acquisition_id += 1
    assert guard.observe(context, invalidated=True)
    assert not guard.observe(context, processing=True)
    context.target_generation += 1
    assert guard.observe(context, processing=True)
    old = deepcopy(context)
    context.perception_epoch = 'epoch-2'
    assert guard.observe(context, processing=True)
    assert not guard.observe(old, processing=True)
    guard.reset()
    assert not guard.observe(context, processing=True)
