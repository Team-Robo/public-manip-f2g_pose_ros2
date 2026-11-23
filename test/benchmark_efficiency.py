"""Reproduce geometry/map microbenchmarks without ROS, cameras or model weights."""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import tracemalloc
import numpy as np

from f2g_pose_ros2.object_geometry import voxelize
from f2g_pose_ros2.rgbd import rectify_color


def reference_voxelize(points, leaf=.003):
    points=np.asarray(points,dtype=np.float32).reshape(-1,3)
    points=points[np.isfinite(points).all(axis=1)]
    if not len(points):return points
    _,indices=np.unique(np.floor(points/leaf).astype(np.int64),axis=0,return_index=True)
    return points[np.sort(indices)]


def reference_rectify(rgb,mask,K,distortion):
    import cv2
    height,width=mask.shape
    mx,my=cv2.initUndistortRectifyMap(K,np.asarray(distortion),None,K,(width,height),cv2.CV_32FC1)
    return cv2.remap(rgb,mx,my,cv2.INTER_LINEAR),cv2.remap(mask.astype(np.uint8),mx,my,cv2.INTER_NEAREST)>0


def measure(function, iterations):
    for _ in range(5):function()
    samples=[]
    for _ in range(iterations):
        start=time.perf_counter_ns();function();samples.append((time.perf_counter_ns()-start)/1e6)
    tracemalloc.start()
    try:
        function();_,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return dict(median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),
                python_numpy_peak_bytes=peak)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=35)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.iterations<1:parser.error('iterations must be positive')
    rng=np.random.default_rng(313)
    result=dict(metadata=dict(platform=platform.platform(),numpy=np.__version__,cpu_count=os.cpu_count(),
                              seed=313,iterations=args.iterations),workloads={})
    for n in (3500,10000,100000):
        points=rng.uniform(-1,1,(n,3)).astype(np.float32);points[::4]=points[0]
        np.testing.assert_array_equal(voxelize(points),reference_voxelize(points))
        result['workloads']['voxelize_'+str(n)]={
            'before':measure(lambda:reference_voxelize(points),args.iterations),
            'after':measure(lambda:voxelize(points),args.iterations),'exact_parity':True}
    rgb=rng.integers(0,256,(480,848,3),dtype=np.uint8)
    mask=rng.integers(0,2,(480,848),dtype=np.uint8)
    K=np.array([[600.,0,424],[0,600,240],[0,0,1]])
    distortion=np.array([.1,-.02,.001,.002,0.])
    for old,new in zip(reference_rectify(rgb,mask,K,distortion),rectify_color(rgb,mask,K,distortion)):
        np.testing.assert_array_equal(old,new)
    result['workloads']['rectification_848x480']={
        'before':measure(lambda:reference_rectify(rgb,mask,K,distortion),args.iterations),
        'after':measure(lambda:rectify_color(rgb,mask,K,distortion),args.iterations),'exact_parity':True}
    text=json.dumps(result,indent=2)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:stream.write(text+'\n')
    print(text)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
