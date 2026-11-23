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

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import std_msgs.msg
import geometry_msgs.msg
import visualization_msgs.msg
import sensor_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2


def points_to_pointcloud2(points: np.ndarray, header: std_msgs.msg.Header) -> sensor_msgs.msg.PointCloud2:
    """Convert (N, 3) numpy array of float32 to sensor_msgs/PointCloud2."""
    if points is None or len(points) == 0:
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        return pc2.create_cloud_xyz32(header, empty_pts)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) point cloud, got shape {points.shape}")
    return pc2.create_cloud_xyz32(header, points)


def pointcloud2_to_points(msg: sensor_msgs.msg.PointCloud2) -> np.ndarray:
    """Extract (N, 3) float32 coordinates from sensor_msgs/PointCloud2."""
    if msg.width * msg.height == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z'])
    # Filter out NaNs / Infs
    mask = np.all(np.isfinite(pts), axis=1)
    return pts[mask].astype(np.float32)


def pointcloud2_to_labeled_points(msg: sensor_msgs.msg.PointCloud2) -> tuple[np.ndarray, np.ndarray]:
    """Extract target (intensity > 0.5) and obstacle (intensity <= 0.5) points from PointCloud2."""
    if msg.width * msg.height == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    has_intensity = any(f.name == 'intensity' for f in msg.fields)
    if has_intensity:
        data = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z', 'intensity'])
        mask = np.all(np.isfinite(data), axis=1)
        data = data[mask]
        target_pts = data[data[:, 3] > 0.5, :3].astype(np.float32)
        obstacle_pts = data[data[:, 3] <= 0.5, :3].astype(np.float32)
        return target_pts, obstacle_pts
    else:
        pts = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z'])
        mask = np.all(np.isfinite(pts), axis=1)
        return pts[mask].astype(np.float32), np.zeros((0, 3), dtype=np.float32)


def transform_stamped_to_matrix(ts: geometry_msgs.msg.TransformStamped) -> np.ndarray:
    """Convert geometry_msgs/TransformStamped to 4x4 homogeneous transformation matrix."""
    t = ts.transform.translation
    q = ts.transform.rotation
    mat = np.eye(4, dtype=np.float32)
    r = R.from_quat([q.x, q.y, q.z, q.w])
    mat[:3, :3] = r.as_matrix().astype(np.float32)
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat


def transform_points(points: np.ndarray, transform_mat: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points by 4x4 matrix."""
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    R_mat = transform_mat[:3, :3]
    t_vec = transform_mat[:3, 3]
    return (points @ R_mat.T + t_vec).astype(np.float32)


def matrix_to_pose(mat: np.ndarray) -> geometry_msgs.msg.Pose:
    """Convert 4x4 matrix to geometry_msgs/Pose."""
    pose = geometry_msgs.msg.Pose()
    pose.position.x = float(mat[0, 3])
    pose.position.y = float(mat[1, 3])
    pose.position.z = float(mat[2, 3])
    r = R.from_matrix(mat[:3, :3])
    q = r.as_quat()  # [x, y, z, w]
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


def voxel_grid_downsample(points: np.ndarray, leaf_size: float = 0.003) -> np.ndarray:
    """Voxel downsample (N, 3) points using Open3D."""
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    downsampled = pcd.voxel_down_sample(voxel_size=leaf_size)
    return np.asarray(downsampled.points, dtype=np.float32)


def create_bbox_marker(
    pose: geometry_msgs.msg.Pose,
    dimensions: np.ndarray,
    frame_id: str,
    stamp: std_msgs.msg.Header,
    marker_id: int = 1,
    color: tuple = (0.0, 1.0, 0.2, 0.6)
) -> visualization_msgs.msg.Marker:
    """Create oriented 3D bounding box marker for RViz."""
    marker = visualization_msgs.msg.Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp.stamp
    marker.ns = "f2g_bbox"
    marker.id = marker_id
    marker.type = visualization_msgs.msg.Marker.CUBE
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.pose = pose
    marker.scale.x = float(dimensions[0])
    marker.scale.y = float(dimensions[1])
    marker.scale.z = float(dimensions[2])
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])
    import builtin_interfaces.msg
    duration = builtin_interfaces.msg.Duration()
    duration.sec = 10
    duration.nanosec = 0
    marker.lifetime = duration
    return marker
