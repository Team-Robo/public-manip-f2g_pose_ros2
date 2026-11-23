from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from f2g_pose_ros2.rgbd import register_depth,rectify_color


def test_identity_registration_preserves_depth_and_invalid_pixels():
    depth=np.arange(25,dtype=np.float32).reshape(5,5)/100+1
    depth[0,0]=0
    K=np.array([[100,0,2],[0,100,2],[0,0,1]],float)
    np.testing.assert_allclose(register_depth(depth,K,K,np.eye(4),(5,5)),depth)


def test_real_baseline_moves_depth_pixels_in_color_frame():
    depth=np.zeros((5,10),np.float32);depth[2,3]=1
    K=np.array([[100,0,4],[0,100,2],[0,0,1]],float)
    T=np.eye(4);T[0,3]=.02
    registered=register_depth(depth,K,K,T,(5,10))
    assert registered[2,5]==1 and registered[2,3]==0


def test_rotated_baseline_is_not_the_upright_driver_alignment():
    depth=np.zeros((5,10),np.float32);depth[2,5]=1
    K=np.array([[100,0,4],[0,100,2],[0,0,1]],float)
    T=np.eye(4);T[0,3]=-.02  # Corrected optical frames after 180-degree rotation.
    registered=register_depth(depth,K,K,T,(5,10))
    assert registered[2,3]==1 and registered[2,7]==0


def test_zbuffer_keeps_nearer_surface_when_depth_pixels_project_to_same_color_pixel():
    depth=np.array([[1.,2.]],np.float32)
    kd=np.array([[100,0,0],[0,100,0],[0,0,1]],float)
    kc=np.eye(3)
    registered=register_depth(depth,kd,kc,np.eye(4),(1,1))
    assert registered[0,0]==1


def test_different_head_rgb_and_depth_resolutions_are_supported():
    depth=np.ones((2,2),np.float32)
    kd=np.eye(3);kc=np.diag([2.,2.,1.])
    registered=register_depth(depth,kd,kc,np.eye(4),(4,4))
    assert registered.shape==(4,4) and registered[2,2]==1


def test_rectification_cache_preserves_pixels_and_invalidates_mutated_calibration():
    import cv2
    from f2g_pose_ros2.rgbd import _rectification_maps
    _rectification_maps.cache_clear()
    rng=np.random.default_rng(71)
    K=np.array([[90.,0.,16.],[0.,95.,12.],[0.,0.,1.]])
    distortion=np.array([.2,-.03,.01,-.02,0.])
    for width,change in [(32,False),(32,False),(32,True),(40,False)]:
        if change:
            K[0,0]+=3.;distortion[0]+=.1
        image=rng.integers(0,256,(24,width,3),dtype=np.uint8)
        mask=rng.integers(0,2,(24,width),dtype=np.uint8)
        mx,my=cv2.initUndistortRectifyMap(K,distortion,None,K,(width,24),cv2.CV_32FC1)
        rgb,selected=rectify_color(image,mask,K,distortion)
        np.testing.assert_array_equal(rgb,cv2.remap(image,mx,my,cv2.INTER_LINEAR))
        np.testing.assert_array_equal(selected,cv2.remap(mask,mx,my,cv2.INTER_NEAREST)>0)
    assert _rectification_maps.cache_info().currsize <= 2
    assert _rectification_maps.cache_info().hits >= 1
