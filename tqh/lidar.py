# Copyright 2025 Peter Edward Creasey

import numpy as np
from .lib import raytrace_shadows as check_occlusion, Heightfield
from PIL import Image
from sys import stdout
from os import path
import rasterio

yellow = np.array((1.3, 1.1, 0.5))
sky_blue = np.array((0.175, 0.41, 1.0))

def normalize(v):
    norm = sum(c*c for c in v)**(-0.5)
    return np.array(tuple(c*norm for c in v))

def get_up_right_fwd(phi, theta):
    """
    theta up from horizontal (i.e. positive theta points in positive z)
    phi from the x direction (fwd)
    """
    up_dir = np.array((-np.sin(theta) * np.sin(phi), -np.sin(theta) * np.cos(phi), np.cos(theta)))
    right_dir = np.array((np.cos(phi), -np.sin(phi), 0.0))
    fwd_dir = np.array((np.cos(theta) * np.sin(phi), np.cos(theta)*np.cos(phi), np.sin(theta)))
    return up_dir, right_dir, fwd_dir

    
def get_frustum_displacements(nx, ny, up_dir, right_dir, fwd_dir, fov, fwd_dist, log):
    """
    Get the ray displacements for a camera with angles 
    ray_dist - Distance in the forward direction
    All in radians

    """
    # NB this is wrong, should be tangent
    sin_half_fov = np.sin(fov * 0.5)
    
    # Orthogonal camera orients
    print('  Forward dir', fwd_dir, 'Right dir', right_dir, 'up dir', up_dir, file=log)        
    # Incremental displacement from moving pixel in x and y
    delta_x = fwd_dist * 2.0*sin_half_fov* right_dir / (0.5 * nx)
    delta_y = -2 * fwd_dist * sin_half_fov* up_dir / (0.5 * nx)

    # Top left (x=0,y=0) pixel)
    top_left = fwd_dist * (fwd_dir - 2.0*sin_half_fov * right_dir + 2.0*sin_half_fov*(ny/nx) * up_dir)

    return top_left, delta_x, delta_y

def raytrace_perspective_heightfield(heights, x_scale, centre, light_dir, light_clr, sky_clr, pix_shape, camera_angles, upsample=1, log=stdout, is_periodic=False):
    # For an (N,M) array of heights
    # Render from a perspective looking at the centre point

    # Choose an appropriate offset
    bottom_left = 0

    bottom_left = (bottom_left,)*3
    
    # Set up the camera position
    phi,theta,fov_deg,dist_mult = camera_angles

    up_dir, right_dir, fwd_dir = get_up_right_fwd(phi, theta)

    dist_away = max(heights.shape)*dist_mult
    camera_pos = [bl + c - f*dist_away for bl,c,f in zip(bottom_left, (centre[0],centre[1],0), fwd_dir)]

    ray_dist = 2 * max(heights.shape)
    top_left, delta_x, delta_y = get_frustum_displacements(pix_shape[0], pix_shape[1], up_dir, right_dir, fwd_dir, fov_deg * np.pi /180, ray_dist, log)
    
    top_left[2] *= x_scale
    delta_x[2] *=  x_scale 
    delta_y[2] *=  x_scale
    
    camera_pos[2] = camera_pos[2] * x_scale

    print('Camera pos', camera_pos, file=log)
    print('Bottom left offset', bottom_left[0], file=log)
    
    use_beams = pix_shape[0]*pix_shape[1] > 1e7
    if use_beams:
        print('Building Manhattan bounds for beams', file=log)
        
    with Heightfield(heights, is_periodic, use_beams, log) as h:
        print('Tracing rays for %dx%d pixels'%(pix_shape[1],pix_shape[0]), 'buffer = {:,} MB'.format((pix_shape[0]*pix_shape[1]*4)//(1024*1024)), file=log)
        print('%dx%d samples per pixel'%(upsample,upsample), ' ={:,} rays'.format(pix_shape[0]*pix_shape[1]*upsample*upsample), file=log)
        rgba = h.raytrace_image(pix_shape, top_left, delta_x, delta_y, x_scale,camera_pos, upsample, light_dir, light_clr, sky_clr)
        return rgba

def perspective_shadow_lit(lidar_data, save_name=None, upsample=1, focus = (7161, 7629), 
                                camera_angles=(1.4*np.pi, -0.9, 35,0.4), light_dir = (-0.3,0.5,1.0), light_clr = yellow, sky_clr=sky_blue, pix_shape = (3600,2500), is_periodic=False,log=stdout):
    

    arr, subsample_lidar, x_scale, left = lidar_data

    print('Heightfield of shape', arr.shape, file=log)
    # camera target

    focus = [focus[0] - left[0], focus[1] - left[1]]
    if subsample_lidar > 1:
        focus = [focus[0]//subsample_lidar, focus[1]//subsample_lidar]
    print('Focus point in heightfield (%d,%d)'%(focus[0],focus[1]), file=log)

    
    print('Heightfield values in', arr.min(), arr.max())
    # Building texture
    light_dir = normalize(light_dir)

    v = raytrace_perspective_heightfield(arr, x_scale, focus, light_dir, light_clr, sky_clr, pix_shape=(pix_shape[0],pix_shape[1]), camera_angles=camera_angles, upsample=upsample, log=log, is_periodic=is_periodic)

    rgba = np.reshape(v.view(np.uint8), v.shape + (4,))
    
    print('Displaying/saving', file=log)

    if save_name is not None:

        im = Image.fromarray(rgba)
        im.save(save_name)
    else:
        import pylab as pl        
        pl.figure(figsize=(24,12))
        ax = pl.imshow(rgba)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        pl.tight_layout()
        pl.show()


def elliptical_trajectory(far_dist_frac=0.5, close_dist_frac=0.1, side_dist_frac=0.3,
                          phi0=-1.35, theta=-0.8, fov_deg=40, light_dir = (-0.5,0.3,0.2)):
    # Ellipse starting at far point, moving to close and back
    a = 0.5 * (far_dist_frac + close_dist_frac)
    b = side_dist_frac
    # Compute distances numerically
    angle = np.arange(1000)*2*np.pi / (1000-1)
    angle_mid = 0.5 * (angle[:-1] + angle[1:])
    dx = np.zeros_like(angle)
    dx[1:] = np.cumsum(np.sqrt(np.square(np.sin(angle_mid))*a*a + np.square(np.cos(angle_mid))*b*b))
    frac_to_angle = dx / dx[-1]    
    get_angle = lambda frac : np.interp(frac, frac_to_angle, angle)
    light_dir = normalize(light_dir)
    
    def get_trajectory(focus, heights_shape):

        dir0 = np.array([np.sin(phi0), np.cos(phi0)])
        dir1 = np.array([np.cos(phi0), -np.sin(phi0)])


        
        def get_camera_angles_pos(frac):
            ellipse_angle = get_angle(frac)
            ellipse_delta = [a*np.cos(ellipse_angle) + 0.5 *(far_dist_frac - close_dist_frac), b*np.sin(ellipse_angle)]
            horiz_dist_mult = np.sqrt(np.square(ellipse_delta).sum())

            ellipse_rot = np.array(ellipse_delta)/horiz_dist_mult
            # Camera directions
            fwd_dir = np.cos(theta)*(ellipse_rot[0]*dir0 + ellipse_rot[1]*dir1)
            fwd_dir = np.array([fwd_dir[0], fwd_dir[1], np.sin(theta)])
            right_dir = normalize([fwd_dir[1], -fwd_dir[0], 0])
            up_dir = -np.sin(theta)*(ellipse_delta[0]*dir0 + ellipse_delta[1]*dir1)/horiz_dist_mult
            up_dir = np.array([up_dir[0], up_dir[1], np.cos(theta)])

            # Horizontal distance from focus
            horiz_dist_away = max(heights_shape)*horiz_dist_mult
            
            dist_away = horiz_dist_away / np.cos(theta)
            # Set up the camera position            
            camera_pos = [c - f*dist_away for c,f in zip((focus[0],focus[1],0), fwd_dir)]

            light_rot = np.array([light_dir[0]*ellipse_rot[0] + light_dir[1]*ellipse_rot[1],
                         light_dir[1]*ellipse_rot[0] - light_dir[0]*ellipse_rot[1], light_dir[2]])
            
            return fov_deg, up_dir, right_dir, fwd_dir, camera_pos, light_rot
    
        return get_camera_angles_pos
    return get_trajectory
    
def circular_trajectory(camera_angles=(-0.43*np.pi, -0.8, 40,0.5), light_dir = (-0.5,0.3,0.2)):
    def build_circular_trajectory(focus, heights_shape):
    
        phi0,theta,fov_deg,dist_mult = camera_angles
        dist_away = max(heights_shape)*dist_mult
        light_dir = normalize(light_dir)        
        def get_camera_angles_pos(frac):
            theta = theta0
            delta_phi = frac * 0.5*np.pi *0.5 
            phi = phi0+delta_phi
            
            # Set up the camera position
            up_dir, right_dir, fwd_dir = get_up_right_fwd(phi, theta)            
            
            camera_pos = [c - f*dist_away for c,f in zip((focus[0],focus[1],0), fwd_dir)]
            
            cos_delta_phi  = np.cos(delta_phi)
            sin_delta_phi = np.sin(delta_phi)
            light_rot = (light_dir[0]*cos_delta_phi + light_dir[1]*sin_delta_phi, -light_dir[0]*sin_delta_phi + light_dir[1]*cos_delta_phi, light_dir[2])

            return fov_deg, up_dir, right_dir, fwd_dir, camera_pos, light_rot
    
        return get_camera_angles_pos
    return build_circular_trajectory


def shadow_lit_rotation_movie(log, lidar_data, get_camera_trajectory, x_scale=0.5, n_rot=25, save_path='output/frames', save_prefix='frame', upsample=2, point = (7161, 7629), 
                              light_clr = yellow, sky_clr=sky_blue, pix_shape = (1024,768), is_periodic=False):
    
    from os import path
    if not path.exists(save_path):
        raise Exception('Could not find %s'%save_path)

    save_names = [path.join(save_path, '%s_%05d.png'%(save_prefix, i)) for i in range(n_rot)]
#    for save_name in save_names:
#        if path.exists(save_name):
#            raise Exception('%s already exists'%save_name)
    
    arr, subsample_lidar, left = lidar_data

    print('Heightfield of shape', arr.shape, file=log)
    # camera target

    focus = [point[0] - left[0], point[1] - left[1]]
    if subsample_lidar > 1:
        x_scale *= subsample_lidar
        focus = [focus[0]//subsample_lidar, focus[1]//subsample_lidar]
    print('Focus point in heightfield (%d,%d)'%(focus[0],focus[1]), file=log)
    print('Used x_scale (after subsampling)', x_scale, file=log)

    # Exclude the -9999 values
    min_val = arr.ravel()[arr.ravel() > -9998].min()
    heights = np.maximum(arr, min_val)
    
    print('Heightfield values in', heights.min(), heights.max())

    # Building texture
    use_beams = True# pix_shape[0]*pix_shape[1] > 1e7
    if use_beams:
        print('Building Manhattan bounds for beams', file=log)

    # build camera trajectory
    get_camera_angle_pos = get_camera_trajectory(focus, heights.shape)
    
    with Heightfield(heights, is_periodic, use_beams, log) as h:
        print('Tracing rays for %dx%d pixels'%(pix_shape[1],pix_shape[0]), 'buffer = {:,} MB'.format((pix_shape[0]*pix_shape[1]*4)//(1024*1024)), file=log)
        print('%dx%d samples per pixel'%(upsample,upsample), ' ={:,} rays'.format(pix_shape[0]*pix_shape[1]*upsample*upsample), file=log)

        for i,save_name in enumerate(save_names):
            if path.exists(save_name):
                continue


            # Set up the camera position
            fov_deg, up_dir, right_dir, fwd_dir, camera_pos, light_dir = get_camera_angle_pos(i / n_rot)            
            print('up', up_dir, 'right', right_dir, 'fwd', fwd_dir, 'fov', fov_deg, 'degrees')
            
            ray_dist = 2 * max(heights.shape)
            top_left, delta_x, delta_y = get_frustum_displacements(pix_shape[0], pix_shape[1], up_dir, right_dir, fwd_dir, fov_deg * np.pi /180, ray_dist, log)
            
            top_left[2] *= x_scale
            delta_x[2] *=  x_scale 
            delta_y[2] *=  x_scale
        
            camera_pos[2] = camera_pos[2] * x_scale
        
            
            v = h.raytrace_image(pix_shape, top_left, delta_x, delta_y, x_scale,camera_pos, upsample, light_dir, light_clr, sky_clr)
            rgba = np.reshape(v.view(np.uint8), v.shape + (4,))

            print('Saving', save_name, file=log)
            im = Image.fromarray(rgba)
            im.save(save_name)

def load_tiles(names, base_dir='.', subsample_lidar=1, left=(0,0), right=(10000,10000), min_val=None, log=stdout):

    arr = None
    x_scale = 1
    ims = [[path.join(base_dir, name) for name in row] for row in names]

    files = [[rasterio.open(im) for im in row] for row in ims]

    # Size of v[::subsample_lidar]
    subsampled_size = lambda x : (x+subsample_lidar - 1)//subsample_lidar
    # Get the bounds
    v0 = [subsampled_size(row[0].shape[0]) for row in files]
    v1 = [subsampled_size(im.shape[1]) for im in files[0]]

    arr = np.empty((sum(v0), sum(v1)), dtype=np.float32)

    bad_data = -1e10
    max_data = -1e10
    min_data = 1e10
    
    for i,row in enumerate(files):
        for j,f in enumerate(row):
            x_scale = float(f.res[0]) # currently assume pixels square
            bad_data = max(bad_data, float(f.nodata))

            i0,i1 = (sum(v0[:i]), sum(v0[:i+1]))
            j0,j1 = (sum(v1[:j]), sum(v1[:j+1]))
            print('Reading %d,%d from'%(i1-i0,j1-j0),names[i][j], file=log)
            v = f.read()[0,::subsample_lidar,::subsample_lidar]
            good_data = v.ravel()[v.ravel()>bad_data+1]
            min_data = min(min_data, good_data.min())
            max_data = max(max_data, good_data.max())
            arr[i0:i1, j0:j1] = v
            f.close()

    print('Heights in %.2f to %.2f cells'%(min_data, max_data), file=log)
    if min_val is None:
        min_val = min_data
    else:
        print('Clamping to at least', min_val, file=log)
    np.maximum(arr, min_val, out=arr)

    print('x_scale', x_scale, file=log)
    if subsample_lidar != 1:
        x_scale = x_scale * subsample_lidar        
        print('Used x_scale (after subsampling)', x_scale, file=log)    

    return (arr, subsample_lidar, x_scale, left)
