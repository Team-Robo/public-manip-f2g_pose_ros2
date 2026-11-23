"""Register rectified depth into a pinhole color image with calibrated extrinsics."""
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=2)
def _rectification_maps(height, width, k_dtype, k_bytes, d_dtype, d_shape, d_bytes):
    """Retain the two most recent calibrations without borrowing mutable input."""
    import cv2
    K = np.frombuffer(k_bytes, dtype=np.dtype(k_dtype)).reshape(3, 3)
    distortion = np.frombuffer(d_bytes, dtype=np.dtype(d_dtype)).reshape(d_shape)
    maps = cv2.initUndistortRectifyMap(K, distortion, None, K,
                                     (width, height), cv2.CV_32FC1)
    for array in maps:
        array.flags.writeable = False
    return maps


def register_depth(depth_m, depth_k, color_k, color_from_depth, color_shape):
    depth = np.asarray(depth_m, dtype=np.float32)
    kd, kc = np.asarray(depth_k), np.asarray(color_k)
    transform = np.asarray(color_from_depth)
    if (depth.ndim != 2 or kd.shape != (3,3) or kc.shape != (3,3) or
            transform.shape != (4,4) or not np.isfinite(transform).all() or
            not np.isfinite(kd).all() or not np.isfinite(kc).all() or
            min(kd[0,0],kd[1,1],kc[0,0],kc[1,1]) <= 0):
        raise ValueError('Invalid RGB-D registration calibration')
    y,x = np.nonzero(np.isfinite(depth) & (depth > .05) & (depth < 3.5))
    z = depth[y,x]
    xyz = np.column_stack(((x-kd[0,2])*z/kd[0,0], (y-kd[1,2])*z/kd[1,1], z))
    xyz = xyz @ transform[:3,:3].T + transform[:3,3]
    xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:,2] > .05)]
    height,width = color_shape
    u = np.rint(xyz[:,0]*kc[0,0]/xyz[:,2]+kc[0,2]).astype(np.int64)
    v = np.rint(xyz[:,1]*kc[1,1]/xyz[:,2]+kc[1,2]).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    output = np.full(height*width, np.inf, dtype=np.float32)
    # Occlusion-aware z buffer: retain the nearest color-frame depth per pixel.
    np.minimum.at(output, v[inside]*width+u[inside], xyz[inside,2])
    output[~np.isfinite(output)] = 0
    return output.reshape(height,width)


def rectify_color(rgb, mask, K, distortion):
    import cv2
    if not len(distortion) or np.allclose(distortion,0):
        return rgb,mask
    if len(distortion) not in (4,5,8,12,14):
        raise ValueError('Unsupported color distortion coefficients')
    height,width = mask.shape
    K, distortion = np.asarray(K), np.asarray(distortion)
    if height * width <= 4194304:
        mx, my = _rectification_maps(height, width, K.dtype.str, K.tobytes(),
                                    distortion.dtype.str, distortion.shape, distortion.tobytes())
    else:
        # Oversized one-off inputs must not occupy the bounded calibration cache.
        mx, my = cv2.initUndistortRectifyMap(K, distortion, None, K,
                                          (width, height), cv2.CV_32FC1)
    return (cv2.remap(rgb,mx,my,cv2.INTER_LINEAR),
            cv2.remap(mask.astype(np.uint8),mx,my,cv2.INTER_NEAREST)>0)
