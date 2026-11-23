"""Registration must bound object displacement and preserve occluded thin faces."""
import math
import numpy as np
import pytest
from f2g_pose_ros2.fusion import align_and_fuse_shapes, registration_update


def planes(thickness=.004, angle=0.):
    x, y = np.meshgrid(np.linspace(-.015, .015, 15), np.linspace(-.015, .015, 15))
    front = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    R = np.array([[math.cos(angle), 0., math.sin(angle)], [0.,1.,0.], [-math.sin(angle),0.,math.cos(angle)]])
    front = front @ R.T + [0., 0., .4]
    back = front + np.array([0., 0., thickness]) @ R.T
    return front.astype(np.float32), back.astype(np.float32)


def test_small_translation_does_not_hide_large_rotation_displacement():
    points = np.array([[.39,0.,.8], [.4,0.,.8], [.41,0.,.8]])
    T = np.eye(4)
    angle = math.radians(2)
    T[:3,:3] = [[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]]
    moved, accepted, metrics = registration_update(points, T, 5., .01, .015)
    assert not accepted and metrics['centroid_shift_m'] > .03
    assert np.linalg.norm(T[:3,3]) == 0


@pytest.mark.parametrize('thickness,angle', [(.004,0.),(.002,0.),(.004,.4),(.008,-.3)])
def test_measured_front_and_thin_completed_back_remain_separate(thickness, angle):
    front, back = planes(thickness, angle)
    diagnostics = {}
    fused = align_and_fuse_shapes(np.vstack((front, back)), front, enable_icp_refinement=False,
        voxel_size=.001, camera_from_base=np.eye(4), camera_matrix=np.array([[600.,0,320.],[0,600.,240.],[0,0,1.]]),
        diagnostics=diagnostics)
    from scipy.spatial import cKDTree
    distance, _ = cKDTree(fused).query(back)
    assert np.max(distance) < .0015
    assert diagnostics['retained_completed_points'] >= len(back) * .8
    assert diagnostics['pruned_visible_points'] > 0


def test_independently_misaligned_geometry_is_rejected():
    front, back = planes()
    with pytest.raises(ValueError, match='does not align'):
        align_and_fuse_shapes(np.vstack((front, back)) + [1,0,0], front)


def test_identity_icp_preserves_geometry_and_reports_quality():
    front, _ = planes()
    diagnostics = {}
    fused = align_and_fuse_shapes(front, front, diagnostics=diagnostics)
    assert np.isfinite(fused).all()
    assert diagnostics['icp_applied']
    assert diagnostics['max_point_shift_m'] < 1e-5
