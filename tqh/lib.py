# Copyright 2025 Peter Edward Creasey
# Load the C-library

from __future__ import print_function, division, unicode_literals
from numpy.ctypeslib import ndpointer
from numpy.linalg import eigvalsh
import ctypes
from numpy import float64, empty, array, int32, zeros, float32, require, int64, uint32, complex128, uint16
from numpy import roll, diff, flatnonzero, uint64, cumsum, square, unique
import numpy as np
from os import path
import sys
import sysconfig

_libtqh = None

def _initlib(log=sys.stdout, is_debug=False):
    """ Init the library (if not already loaded) """
    global _libtqh

    if _libtqh is not None:
        return _libtqh

    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    lib_prefix = 'libtqh'
    if is_debug:
        lib_prefix += '_debug'
    name = path.join(path.dirname(path.abspath(__file__)), '../build/'+lib_prefix+suffix)
    if not path.exists(name):
        raise Exception('Library '+str(name)+' does not exist. Maybe you forgot to make it?')

    print('Loading libtqh - C functions for triangulated quadtree heightfield calculations', name, file=log)
    lib = ctypes.cdll.LoadLibrary(name)


    # int raytrace_shadows(const quadtree* restrict heights,
    #			  const float* restrict ray_dir, int32_t* restrict is_shadow)
    func = lib.raytrace_shadows
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_void_p, ndpointer(np.float32),ndpointer(np.int32)]

    # int get_shadow_lit_image(const float x_scale, const quadtree* restrict tree, const float* heights,
    #		 const float* restrict light_dir, const float* light_color, const float* sky_color, uint32_t* restrict out_colors)
    func = lib.get_shadow_lit_image
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_float, ctypes.c_void_p, ndpointer(np.float32), ndpointer(np.float32), ndpointer(np.float32), ndpointer(np.float32), ndpointer(np.uint32)]

    # int raytrace_and_shadow_light(const float* restrict camera_pos, const int nx, const int ny, const int upsample, const float* restrict ray_disps,
    #			      const float x_scale, const quadtree* restrict tree, const float* heights, 
    #			      const float* restrict light_dir, const float* light_color, const float* sky_color, uint32_t* restrict out_colors, float32* restrict hit_fracs)
    func = lib.raytrace_and_shadow_light
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(np.float32), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(np.float32),
                     ctypes.c_float, ctypes.c_void_p, ndpointer(np.float32), 
                     ndpointer(np.float32), ndpointer(np.float32), ndpointer(np.float32), ndpointer(np.uint32), ndpointer(np.float32)]

    # Returns 0 on out-of-memory
    # quadtree* init_tree(const int N, const int M, const int is_periodic, const int use_manhattan, const float* restrict heights);
    func = lib.init_tree
    func.restype = ctypes.c_void_p
    func.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(np.float32)]
    
    # void free_tree(quadtree* tree);
    func = lib.free_tree
    func.restype = None
    func.argtypes = [ctypes.c_void_p]

    _libtqh = lib
    return lib

class Heightfield:
    def __init__(self, heights, is_periodic, use_beams, log=sys.stdout, min_size=5, voxels=None):
        self._heights = np.require(heights, dtype=np.float32, requirements=['C'])
        N,M = self._heights.shape
        self._is_periodic = is_periodic
        self._use_beams = use_beams
        is_not_power2 = lambda x : x<=0 or ((x &(x-1)) !=0)
        if is_periodic and (is_not_power2(N) or is_not_power2(M)):
            raise Exception('Periodic heightfields need power-of-2 sides, but (%d,%d) provided'%(N,M))

        self._shape = (N,M)
        self._log = log
        self.lib = _initlib(self._log)

    def __enter__(self):
        # Allows pythons "with" statement to be used
        print('Building quadtrees', file=self._log)
        self._tree_ptr = self.lib.init_tree(self._shape[0], self._shape[1], int(self._is_periodic), int(self._use_beams), self._heights)
        assert(self._tree_ptr)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Allows pythons "with" statement to be used
        # if traceback is not None then an exception occurred,
        # and we have the chance to do something about it, however
        # currently we just release the object
        self.lib.free_tree(self._tree_ptr)

    def shadow_lit_image(self, x_scale, light_dir, light_color, sky_color):
        N,M = self._shape
        light_dir = np.array([light_dir[0], light_dir[1], light_dir[2]*x_scale], dtype=np.float32)
        light_color = np.require(light_color, dtype=np.float32, requirements=['C'])
        sky_color = np.require(sky_color, dtype=np.float32, requirements=['C'])        

        colors = np.empty((N*M,), dtype=np.uint32)

        print('Tracing rays', file=self._log)
        res = self.lib.get_shadow_lit_image(x_scale, self._tree_ptr, self._heights, light_dir, light_color, sky_color, colors)
    
        assert(res==0)
        colors = np.reshape(colors, (N,M))
        return colors

    def raytrace_image(self, pix_shape, disp_top_left, disp_delta_x, disp_delta_y, x_scale, camera_pos, upsample, light_dir, light_color, sky_color, hit_fracs=None):
        if hit_fracs is not None:
            assert(hit_fracs.size==pix_shape[0]*pix_shape[1])
            assert(hit_fracs.dtype==np.float32)
        else:
            hit_fracs = np.empty(pix_shape, dtype=np.float32)
        assert(upsample > 0)
        camera_pos = np.require([camera_pos[0], camera_pos[1], camera_pos[2]], dtype=np.float32, requirements=['C'])
        ray_disps = np.reshape(np.array([disp_top_left, disp_delta_x, disp_delta_y], dtype=np.float32), (9,))
        light_dir = np.array([light_dir[0], light_dir[1], light_dir[2]*x_scale], dtype=np.float32)
        light_color = np.require(light_color, dtype=np.float32, requirements=['C'])
        sky_color = np.require(sky_color, dtype=np.float32, requirements=['C'])        
        colors = np.empty(pix_shape, dtype=np.uint32)
        
        res = self.lib.raytrace_and_shadow_light(camera_pos, pix_shape[0], pix_shape[1], upsample, ray_disps,
                                                 x_scale, self._tree_ptr, self._heights, light_dir, light_color, sky_color, colors, hit_fracs)
        assert(res==0)
        return np.reshape(colors, (pix_shape[1], pix_shape[0]))
    

def raytrace_shadows(heights, light_dir, log=sys.stdout):
    # int raytrace_shadows(const int N, const int M, const float* restrict heights,
    #			  const float* restrict ray_dir, int32_t* restrict is_shadow)
    lib = _initlib(log)
    heights = np.require(heights, dtype=np.float32, requirements=['C'])
    N,M = heights.shape

    light_dir = np.require(light_dir, dtype=np.float32, requirements=['C'])
    is_shadow = np.empty((N*M,), dtype=np.int32)
    with Heightfield(heights) as h:
        print('Tracing rays', file=log)
        res = lib.raytrace_shadows(h, light_dir, is_shadow)
    
        assert(res==0)
    is_shadow = np.reshape(is_shadow, (N,M))
    return is_shadow
    
    

