"""Bounded registration and visibility-aware fusion of measured/completed surfaces."""
import math
import numpy as np
from scipy.spatial import cKDTree
from .cloud_utils import voxel_grid_downsample


def registration_update(points, transform, max_rotation_deg, max_centroid_shift, max_point_shift):
    """Validate actual object movement, including rotation about the frame origin."""
    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('ICP returned a nonfinite or nonrigid transform')
    R, t = T[:3, :3], T[:3, 3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1., atol=1e-5):
        raise ValueError('ICP returned a nonrigid rotation')
    angle = math.degrees(math.acos(float(np.clip((np.trace(R)-1.)/2., -1., 1.))))
    moved = points @ R.T + t
    center_shift = float(np.linalg.norm(moved.mean(axis=0) - points.mean(axis=0)))
    point_shift = float(np.linalg.norm(moved - points, axis=1).max())
    metrics = dict(rotation_deg=angle, centroid_shift_m=center_shift, max_point_shift_m=point_shift)
    accepted = angle <= max_rotation_deg and center_shift <= max_centroid_shift and point_shift <= max_point_shift
    return moved, accepted, metrics


def _visible_overlap(completed, observed, nearest, distance, overlap, camera_from_base,
                     camera_matrix, radius_pixels, depth_tolerance):
    if camera_from_base is None or camera_matrix is None:
        # Without view provenance, only remove near-exact duplicates. A small
        # Euclidean gap alone cannot distinguish a front surface from a back.
        return distance <= min(overlap, depth_tolerance)
    T, K = np.asarray(camera_from_base), np.asarray(camera_matrix)
    if (T.shape != (4, 4) or K.shape != (3, 3) or not np.isfinite(T).all()
            or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0):
        raise ValueError('Invalid acquisition camera for fusion visibility')
    source = completed @ T[:3, :3].T + T[:3, 3]
    target = observed @ T[:3, :3].T + T[:3, 3]
    if np.any(source[:, 2] <= 0) or np.any(target[:, 2] <= 0):
        raise ValueError('Fusion geometry crosses the acquisition camera plane')
    def project(points):
        return points[:, :2] / points[:, 2, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    pixels = project(source)
    target_pixels = project(target)
    neighborhoods = cKDTree(pixels).query_ball_point(pixels, radius_pixels)
    front_depth = np.array([source[indices, 2].min() for indices in neighborhoods])
    visible = source[:, 2] <= front_depth + depth_tolerance
    measured_overlap = np.linalg.norm(pixels - target_pixels[nearest], axis=1) <= radius_pixels
    return (distance <= overlap) & visible & measured_overlap


def align_and_fuse_shapes(points_f2g, points_wrist, voxel_size=.003, overlap_distance=.012,
        enable_icp_refinement=True, max_rotation_deg=5., max_centroid_shift=.01,
        max_point_shift=.015, max_alignment_distance=.03, min_alignment_fraction=.3,
        camera_from_base=None, camera_matrix=None, visibility_radius_pixels=8.,
        visibility_depth_tolerance=.0005, diagnostics=None):
    """Replace credible visible overlap while retaining occluded back/side geometry.

    Pose limits are explicit initial bounds, not a claim of calibrated accuracy.
    Measured and retained completed surfaces are downsampled separately so thin
    opposite surfaces cannot be averaged into one shared voxel.
    """
    values = (voxel_size, overlap_distance, max_rotation_deg, max_centroid_shift,
              max_point_shift, max_alignment_distance, min_alignment_fraction,
              visibility_radius_pixels, visibility_depth_tolerance)
    if not all(math.isfinite(v) and v > 0 for v in values) or min_alignment_fraction > 1:
        raise ValueError('Fusion bounds must be finite and positive; alignment fraction must be <= 1')
    source, observed = (np.asarray(p, dtype=np.float64) for p in (points_f2g, points_wrist))
    for points in (source, observed):
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError('Fusion requires nonempty finite completed and observed geometry')
    report = diagnostics if diagnostics is not None else {}
    report.update(completed_points=len(source), observed_points=len(observed), icp_applied=False,
        icp_reason='disabled', max_rotation_deg=max_rotation_deg,
        max_centroid_shift_m=max_centroid_shift, max_point_shift_m=max_point_shift)
    def alignment(points):
        distances, _ = cKDTree(points).query(observed, k=1)
        return float(np.mean(distances <= max_alignment_distance))
    initial_alignment = alignment(source)
    report['initial_alignment_fraction'] = initial_alignment
    if initial_alignment < min_alignment_fraction:
        raise ValueError(f'Completed shape does not align with measured wrist geometry ({initial_alignment:.3f})')
    if enable_icp_refinement and min(len(source), len(observed)) > 20:
        import open3d as o3d
        src, tgt = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        src.points, tgt.points = o3d.utility.Vector3dVector(source), o3d.utility.Vector3dVector(observed)
        reg = o3d.pipelines.registration.registration_icp(src, tgt,
            max_correspondence_distance=max_alignment_distance, init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        fitness, residual = float(reg.fitness), float(reg.inlier_rmse)
        report.update(icp_fitness=fitness if math.isfinite(fitness) else None,
                      icp_rmse_m=residual if math.isfinite(residual) else None)
        moved, accepted, metrics = registration_update(source, reg.transformation,
            max_rotation_deg, max_centroid_shift, max_point_shift)
        report.update(metrics)
        quality = math.isfinite(fitness) and math.isfinite(residual) and fitness > .3 and residual < .015
        if accepted and quality and alignment(moved) >= min_alignment_fraction:
            source = moved
            report.update(icp_applied=True, icp_reason='accepted')
        else:
            report['icp_reason'] = 'pose bound or registration quality rejected; independently validated original shape retained'
    distances, nearest = cKDTree(observed).query(source, k=1)
    overlap = _visible_overlap(source, observed, nearest, distances, overlap_distance,
        camera_from_base, camera_matrix, visibility_radius_pixels, visibility_depth_tolerance)
    back = source[~overlap]
    measured = voxel_grid_downsample(observed.astype(np.float32), leaf_size=voxel_size)
    retained = voxel_grid_downsample(back.astype(np.float32), leaf_size=voxel_size) if len(back) else np.empty((0, 3), np.float32)
    fused = np.vstack((measured, retained)).astype(np.float32)
    report.update(pruned_visible_points=int(overlap.sum()), retained_completed_points=len(retained),
                  measured_points=len(measured), fused_points=len(fused), voxel_size_m=voxel_size,
                  visibility='camera_projection' if camera_from_base is not None else 'near_exact_duplicates')
    return fused
