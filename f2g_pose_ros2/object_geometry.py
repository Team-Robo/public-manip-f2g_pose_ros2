"""Geometry contracts shared by the explicit R1 perception workflows (no ROS)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Geometry:
    points: np.ndarray
    obstacles: np.ndarray
    center: np.ndarray
    dimensions: np.ndarray
    pose: np.ndarray
    model_dimensions: np.ndarray
    stamp_ns: int
    source: str
    completed: bool


def voxelize(points, leaf=.003):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    if len(points) < 256 or not finite.all():
        points = points[finite]
    if len(points) == 0:
        return points
    cells = np.floor(points / leaf).astype(np.int64)
    packed = False
    if len(points) >= 256:
        low, high = cells.min(axis=0), cells.max(axis=0)
        packed = all(int(hi) - int(lo) < (1 << 21) for lo, hi in zip(low, high))
    # Sorting scalar integer keys avoids NumPy's per-row comparison sort. Pack
    # only when all coordinates fit exactly; arbitrary ranges retain the old path.
    if packed:
        keys = (cells[:, 0] - low[0]).astype(np.uint64)
        keys <<= 42
        keys |= (cells[:, 1] - low[1]).astype(np.uint64) << 21
        keys |= (cells[:, 2] - low[2]).astype(np.uint64)
        _, indices = np.unique(keys, return_index=True)
    else:
        _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def bounds(points):
    points = np.asarray(points)
    if len(points) < 10 or not np.isfinite(points).all():
        raise ValueError("Object requires at least 10 finite points")
    lo, hi = points.min(axis=0), points.max(axis=0)
    # A visible planar patch has no measured thickness; use one voxel as
    # numerical thickness, without claiming recovery of the hidden volume.
    size = np.maximum(hi - lo, .003)
    return (lo + hi) / 2, size


def estimate(points, obstacles, stamp_ns, source, pose=None, dimensions=None, *,
             obstacles_are_voxelized=False):
    points = voxelize(points)
    if not obstacles_are_voxelized:
        obstacles = voxelize(obstacles, .012)
    center, size = bounds(points)
    completed = pose is not None
    if completed:
        pose = np.asarray(pose, dtype=float)
        dimensions = np.asarray(dimensions, dtype=float)
        if (pose.shape != (4, 4) or dimensions.shape != (3,) or
                not np.isfinite(pose).all() or not np.isfinite(dimensions).all() or
                np.any(dimensions <= 0)):
            raise ValueError("Invalid F2G pose/dimensions")
        # Enclose the oriented model box in the planning frame, matching MTC's
        # axis-aligned collision object contract. Include completed cloud bounds.
        half = np.abs(pose[:3, :3]) @ (dimensions / 2)
        low = np.minimum(center - size / 2, pose[:3, 3] - half)
        high = np.maximum(center + size / 2, pose[:3, 3] + half)
        center, size = (low + high) / 2, high - low
    else:
        pose = np.eye(4)
        pose[:3, 3] = center
        dimensions = size.copy()
    return Geometry(points, obstacles, center, size, pose, dimensions,
                    int(stamp_ns), source, completed)


def fuse(head, wrist, max_age_difference=180., max_rmse=.01, min_fitness=.5):
    separation = abs(head.stamp_ns - wrist.stamp_ns) / 1e9
    if separation > max_age_difference:
        raise ValueError(f'Head/wrist captures are too far apart: {separation:.1f}s exceeds '
                         f'{max_age_difference:g}s; recapture the static object')
    import open3d as o3d
    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(head.points.astype(float))
    target.points = o3d.utility.Vector3dVector(wrist.points.astype(float))
    reg = o3d.pipelines.registration.registration_icp(
        source, target, .03, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
    correction = np.linalg.norm(reg.transformation[:3, 3])
    angle = np.arccos(np.clip((np.trace(reg.transformation[:3, :3]) - 1) / 2, -1, 1))
    if (not np.isfinite(reg.fitness) or not np.isfinite(reg.transformation).all() or
            reg.fitness < min_fitness or not np.isfinite(reg.inlier_rmse) or
            reg.inlier_rmse > max_rmse or correction > .03 or angle > np.deg2rad(15)):
        raise ValueError(f"Fusion rejected: fitness={reg.fitness:.3f}, "
                         f"rmse={reg.inlier_rmse:.4f}m, correction={correction:.4f}m, "
                         f"rotation={np.rad2deg(angle):.1f}deg")
    aligned = head.points @ reg.transformation[:3, :3].T + reg.transformation[:3, 3]
    from scipy.spatial import cKDTree
    distance, _ = cKDTree(wrist.points).query(aligned)
    points = np.vstack([wrist.points, aligned[distance > .006]])
    # Wrist scene is the current obstacle evidence; head obstacles are unregistered.
    merged = estimate(points, wrist.obstacles, min(head.stamp_ns, wrist.stamp_ns), 'fused',
                      obstacles_are_voxelized=True)
    return merged, {'fitness': float(reg.fitness), 'rmse_m': float(reg.inlier_rmse),
                    'translation_m': float(correction), 'rotation_deg': float(np.rad2deg(angle))}


def resolve_modes(camera_mode, cloud_source='auto'):
    if camera_mode not in ('head', 'wrist', 'head_then_wrist'):
        raise ValueError('camera_mode must be head, wrist, or head_then_wrist')
    available = {'head'} if camera_mode == 'head' else {'wrist'} if camera_mode == 'wrist' else {'head', 'wrist', 'fused'}
    selected = ('head' if camera_mode == 'head' else 'wrist') if cloud_source == 'auto' else cloud_source
    if selected not in available:
        raise ValueError(f'cloud_source={selected} is unavailable in camera_mode={camera_mode}')
    return selected
