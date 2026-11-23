import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f2g_pose_ros2.object_geometry import estimate, fuse, resolve_modes


def cloud():
    return np.array([[x, y, z] for x in (-.02, 0, .02)
                     for y in (-.03, 0, .03) for z in (.5, .52, .54)], dtype=np.float32)


@pytest.mark.parametrize('mode,source,expected', [('head','auto','head'), ('wrist','auto','wrist'),
    ('head_then_wrist','auto','wrist'), ('head_then_wrist','fused','fused'), ('head_then_wrist','head','head')])
def test_modes(mode, source, expected):
    assert resolve_modes(mode, source) == expected


@pytest.mark.parametrize('mode,source', [('head','fused'), ('wrist','head'), ('unknown','auto')])
def test_unavailable_source_is_rejected(mode, source):
    with pytest.raises(ValueError):
        resolve_modes(mode, source)


def test_measured_geometry_is_a_bounds_estimate_and_keeps_obstacles_separate():
    g = estimate(cloud(), [[1, 2, 3]], 123, 'wrist')
    assert not g.completed
    np.testing.assert_allclose(g.center, [0, 0, .52], atol=1e-7)
    np.testing.assert_allclose(g.dimensions, [.04, .06, .04], atol=1e-7)
    np.testing.assert_allclose(g.pose[:3, 3], g.center)
    assert len(g.points) == 27 and len(g.obstacles) == 1


def test_oriented_f2g_dimensions_are_enclosed_in_mtc_axis_aligned_box():
    T = np.eye(4)
    angle = np.pi / 4
    T[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    T[:3, 3] = [0, 0, .52]
    g = estimate(cloud(), [], 123, 'head', T, [.1, .2, .08])
    np.testing.assert_allclose(g.dimensions, [3*.1/np.sqrt(2), 3*.1/np.sqrt(2), .08], atol=1e-7)
    np.testing.assert_allclose(g.model_dimensions, [.1, .2, .08])
    assert g.completed


def test_invalid_model_dimensions_fail_closed():
    with pytest.raises(ValueError, match='Invalid F2G'):
        estimate(cloud(), [], 123, 'head', np.eye(4), [-1, 1, 1])


def fake_icp(monkeypatch, fitness=.8, rmse=.003, translation=0):
    T = np.eye(4); T[0,3] = translation
    reg = NS(fitness=fitness, inlier_rmse=rmse, transformation=T)
    registration = NS(registration_icp=lambda *a, **kw: reg,
        TransformationEstimationPointToPoint=lambda: None,
        ICPConvergenceCriteria=lambda **kw: None)
    monkeypatch.setitem(sys.modules, 'open3d', NS(geometry=NS(PointCloud=lambda: NS()),
        utility=NS(Vector3dVector=lambda v: v), pipelines=NS(registration=registration)))


@pytest.mark.parametrize('fitness,rmse,translation', [(.1,.003,0), (.8,.02,0), (.8,.003,.05)])
def test_bad_registration_never_silently_merges(monkeypatch, fitness, rmse, translation):
    fake_icp(monkeypatch, fitness, rmse, translation)
    g = estimate(cloud(), [], 123, 'head')
    with pytest.raises(ValueError, match='Fusion rejected'):
        fuse(g, g)


def test_fusion_recomputes_bounds_and_preserves_measured_wrist_obstacles(monkeypatch):
    fake_icp(monkeypatch)
    h = estimate(cloud(), [[3,3,3]], 10**9, 'head')
    w = estimate(cloud()+[.001,0,0], [[2,2,2]], 2*10**9, 'wrist')
    g, metrics = fuse(h, w)
    assert g.source == 'fused' and g.stamp_ns == 10**9
    np.testing.assert_allclose(g.obstacles, w.obstacles)
    assert metrics['rmse_m'] == .003
    np.testing.assert_allclose(g.center, w.center, atol=1e-7)


def test_time_separated_views_are_rejected_before_registration():
    h = estimate(cloud(), [], 1, 'head')
    w = estimate(cloud(), [], 200*10**9, 'wrist')
    with pytest.raises(ValueError, match='too far apart'):
        fuse(h, w)


@pytest.mark.parametrize('scale', [1., 100000.])
@pytest.mark.parametrize('leaf', [.003, .012])
def test_voxel_keys_preserve_first_representative_and_input_order(scale,leaf):
    from f2g_pose_ros2.object_geometry import voxelize
    rng=np.random.default_rng(512)
    points=rng.uniform(-scale,scale,(1200,3)).astype(np.float32)
    points[::5]=points[0];points[::13]=np.nan;points[::17]=np.inf
    # Reversed/strided arrays must keep the same logical first-point semantics.
    for input_points in (points,points[::-1,::1],points[::2]):
        finite=input_points[np.isfinite(input_points).all(axis=1)]
        _,indices=np.unique(np.floor(finite/leaf).astype(np.int64),axis=0,return_index=True)
        expected=finite[np.sort(indices)]
        np.testing.assert_array_equal(voxelize(input_points,leaf),expected)


def test_prepared_obstacles_preserve_geometry_without_revoxelizing():
    from f2g_pose_ros2.object_geometry import estimate
    points=np.random.default_rng(11).uniform(.1,.2,(100,3)).astype(np.float32)
    observed=estimate(points,points,17,'head')
    expected=estimate(points,points,17,'head',np.eye(4),np.ones(3)*.1)
    completed=estimate(points,observed.obstacles,17,'head',np.eye(4),np.ones(3)*.1,
                       obstacles_are_voxelized=True)
    assert completed.obstacles is observed.obstacles
    for name in ('points','obstacles','pose','center','dimensions'):
        np.testing.assert_array_equal(getattr(completed,name),getattr(expected,name))
