#include "float3.h"
#include "heightfield_ray.h" // also quadtree_cursor_s
#include <assert.h>
#include <math.h> // floorf
#ifndef NDEBUG
  #include <stdio.h>
#endif
#include <stdlib.h> // malloc, calloc, free

// Very simple lighting function
int get_shadow_lit_image(const float x_scale, const quadtree* restrict tree, const float* heights,
			 const float* restrict light_dir, const float* light_color, const float* sky_color, uint32_t* restrict out_colors)
{
  /*
    Loop over every point in the heightfield, for everything in shadow (according to light_dir), define the exposure
    as

       exposure[i=0,1,2] = norm.z * sky_color[i]

    and where the light_dir is unoccluded and dor(light_dir, norm) >= 0

       exposure[i] = dot(norm, light_dir) * light_color[i] + norm.z * sky_color[i]

    and then the output colors as

       out_color[i=0,1,2] = clamp(256 * (1 - exp(-exposure[i])), 0, 255)
       out_color[3] = 255

   */
  const int N = (int)tree->shape.i, M=(int)tree->shape.j;
  
  heightfield_ray_coroutine_s rc;
  rc.qrc.cursor = init_cursor(N > M ? N : M);
  
  const float length = N+M;
  const float ray_disp[3] = {light_dir[0]*length, light_dir[1]*length, light_dir[2]*length};

  const float inv_l = 1.0f / sqrtf((light_dir[0]*light_dir[0] + light_dir[1]*light_dir[1])*x_scale*x_scale + light_dir[2]*light_dir[2]);
  const float light_dir_scale[3] = {light_dir[0]*x_scale*inv_l, light_dir[1]*x_scale*inv_l, light_dir[2]*inv_l};
  
  for (int i=0, idx=0; i<N; ++i)
  {
    float ray_pos[3];
    for (int j=0; j<M; ++j, ++idx)
    {
      // These gradients are in the frame of the heightfield, which can be scaled
      float grad_x = 0.0f, grad_y = 0.0f;

      // Boundaries
      if (i==0)
	grad_x = heights[M + j] - heights[j];
      else if (i+1==N)
	grad_x = heights[(N-1)*M + j] - heights[(N-2)*M + j];
      else
	grad_x = 0.5f * (heights[(i+1)*M + j] - heights[(i-1)*M + j]);
			 
      if (j==0)
	grad_y = heights[i*M + 1] - heights[i*M];
      else if (j+1==M)
	grad_y = heights[(i+1)*M - 1] - heights[(i+1)*M - 2];
      else
	grad_y = 0.5f * (heights[i*M + j + 1] - heights[i*M + j - 1]);

      // Perpendicular to surface
      const float perp[2] = {-grad_x, -grad_y};

      // Normal to surface in scaled space
      const float inv_n = 1.0f / sqrtf(grad_x*grad_x + grad_y*grad_y + x_scale*x_scale);
      const float norm_scale[3] = {perp[0] * inv_n, perp[1] * inv_n, inv_n*x_scale};

      // Sky color is always present
      float exposure[3] = {sky_color[0] * norm_scale[2],
	                   sky_color[1] * norm_scale[2],
			   sky_color[2] * norm_scale[2]};

      const float norm_dot_light = norm_scale[0]*light_dir_scale[0] +
	                           norm_scale[1]*light_dir_scale[1] +
	                           norm_scale[2]*light_dir_scale[2];

      // Facing the light direction, check for illumination
      if (norm_dot_light > 0.0f)
      {
	ray_pos[0] = i;      
	ray_pos[1] = j;
	ray_pos[2] = heights[i*M+j];

	heightfield_hit_s hit;
	heightfield_ray_s ray;	
	heightfield_ray_init(ray_disp, &ray);
	if (!heightfield_intersect(&rc.qrc.cursor, ray_pos, &ray, &rc, tree, heights, &hit))
	{
	  for (int k=0;k<3;++k)
	    exposure[k] += light_color[k] * norm_dot_light;
	}
      }
      out_colors[idx] = 0xFF000000;
      for (int k=0;k<3;++k)
      {
	const int f = (int)floorf(256.0f * (1.0f - exp(-exposure[k])));
	const uint32_t cpt = f < 0 ? 0 : (f > 255 ? 255 : (uint32_t)f);
	out_colors[idx] |= cpt << (k*8);
      }
    }
  }
  return 0;
}

// Returns a unit vector
static inline pseudovec3f_s norm_from_grad(const heightfield_gradient grad, const float x_scale)
{
  const float perp[2] = {-grad.dh_di, -grad.dh_dj};
  
  // Normal to surface in scaled space
  const float inv_n = 1.0f / sqrtf(grad.dh_di*grad.dh_di + grad.dh_dj*grad.dh_dj + x_scale*x_scale);
  return (pseudovec3f_s){perp[0] * inv_n, perp[1] * inv_n, inv_n*x_scale};
}

static inline float dot(const pseudovec3f_s pvec, const vec3f_s vec)
{
  return pvec.x*vec.x +  pvec.y*vec.y + pvec.z*vec.z;
}

// For unit vec u \in S2, and n normal to surface, find outgoing for lambertian (always positive in n). NOT NORMALISED
static inline float3 sphere_normal_to_lambertian(const float3 u, const float3 n)
{
  const float3 L = float3_add(n, u);
  // Almost surely non-zero, but in FP can happen
  if (float3_L2(L) < 1e-6f)
  {
    return float_mul_float3(2.0f, n); // Clever impls can rely on being on sphere centred at n, i.e. |L|^2 = 2 dot(n,L)
  }

  return L;
}

// Returns the fraction of the Lambdertain BRDF in the upper hemisphere defined by u
static inline float lambertian_restricted(const float3 s, const float3 u, const float3 n, float3* x)
{
  const float uDotN = float3_inner(n,u);
  const float w = 0.5f*(1.f+uDotN);

  // Almost no overlap. Return weight 0 (i.e. ignore result, set to upward vector)
  if (w < 1e-6f)
  {
    *x = u;
    return 0.f;
  }

  const float3 L = sphere_normal_to_lambertian(s, n);
  const float nDotL = float3_inner(n, L);
  const float invLenL = 1.0f / sqrtf(2.0f * nDotL);
  
  const float uDotL = float3_inner(L, u);
  
  // Complete overlap, no correction
  const float one_minus_w = 1-w;
  if (one_minus_w < 1e-6f) 
  {
    // Rare case
    const float scale = (uDotL < 0)  ? -invLenL : invLenL;
    *x = float_mul_float3(scale, L);
    return w;
  }

  const float3 f = float3_fma(u,-uDotN,n);
  const float fDotL = float3_inner(L, f);

  // Alt formula lenPcL = length(cross(p,L))
  const float lenPcL = sqrtf(float3_L2(float3_fma(float_mul_float3(uDotL,n), -nDotL, u)));
  const float alpha = lenPcL - fDotL;
  const float v = 0.25f*(2*lenPcL - w*alpha)*alpha/one_minus_w;
  const float tau = 0.25*alpha/w; // NB different definition from script since absorbed fDotL to make x displacement from L    

  *x = float_mul_float3(invLenL, float3_fma(float3_fma(L,tau,f), sqrt(v) - nDotL,n));
#ifndef NDEBUG
  {
    const float x2 = float3_L2(*x);
    const float xDotN = float3_inner(*x, n);
    const float xDotU = float3_inner(*x, u);    
    if (x2 < 0.95f || x2 > 1.05f || !(xDotU>=0.f) || xDotN < 0.0f)
    {
      printf("x %f %f %f \n", x->abc[0], x->abc[1], x->abc[2]);
      printf("f %f %f %f \n", f.abc[0], f.abc[1], f.abc[2]);
      printf("n %f %f %f \n", n.abc[0], n.abc[1], n.abc[2]);      
      printf("L %f %f %f \n", L.abc[0], L.abc[1], L.abc[2]);
      printf("w %f tau %f v %f lenPcL %f xDotN %f", w, tau, v, lenPcL, xDotN);
      assert(0);
    }
  }
#endif
  return w;
}	  
typedef struct
{
  heightfield_ray_s* beam_rays;
  uint32_t n;
  uint32_t idx; // idx of current ray
  heightfield_beam_coroutine_s beam_walk; // Beam started far along (warm) ray
  
} upsample_coroutine_s;

static void fill_beam_and_rays(const vec3f_s* delta_upsample, const vec3f_s* ray_disp0, const unsigned n, const float inv_n, heightfield_ray_s* beam_rays, heightfield_ray_s* mid_ray, heightfield_beam_s* beam)
{
  vec3f_s mid_disp = {0.f, 0.f, 0.f};
  for (unsigned i = 0; i < n; ++i)
  {
    const vec3f_s ray_disp = vec_add(delta_upsample[i], *ray_disp0);
    mid_disp = vec_fma(mid_disp, inv_n, ray_disp); // TODO could precompute
    heightfield_ray_init(ray_disp.xyz, &beam_rays[i]);
    if (beam)
    {
      if (i==0)
      {
	heightfield_beam_init_ray(&beam_rays[i], beam);
      }
      else
      {
	heightfield_beam_enlarge_to_ray(&beam_rays[i], beam);	  
      }
    }
  }

  if (beam)
  {
    // Mid-ray    
    heightfield_ray_init(mid_disp.xyz, mid_ray);
  }
}
static void upsample_coroutine_start(heightfield_ray_s* beam_rays, const heightfield_beam_s* beam, heightfield_ray_s* mid_ray, const quadtree_cursor_s* cursor, const float* camera_pos, const uint32_t n2, const quadtree* tree,
				     const heightfield_beam_coroutine_s* super_beam_walk, upsample_coroutine_s* res)
{
  res->beam_rays = beam_rays;
  res->n = n2;
  res->idx = 0;
#ifndef NDEBUG
  res->beam_walk.count = 0;  
#endif

  if (beam)
  {
    if (super_beam_walk)
    {
      // Use super-beam to accelerate fine_beam
      heightfield_beam_warm_start_fine_beam(camera_pos, mid_ray, beam, super_beam_walk, &res->beam_walk);
      heightfield_trace_beam_warm_start(camera_pos, mid_ray, beam, tree, &res->beam_walk);      
    }
    else
    {
      // For first ray, do the beam walk
      heightfield_trace_beam_to_last_unobstructed_crossing(cursor, camera_pos, mid_ray, beam, tree, &res->beam_walk);
    }
#ifndef NDEBUG
    if (res->beam_walk.count < 1 && !super_beam_walk)
    {
      print_heightfield_beam(beam);
      print_heightfield_beam_coroutine(&res->beam_walk);	    
      assert(0);
    }
#endif
  }
}

typedef struct
{
  upsample_coroutine_s upsampler;
  heightfield_beam_s beam; // Beam covering the sampled rays  
  vec3f_s ray_disp; // Displacement corresponding to top-left of pixel
  heightfield_ray_s mid_ray;  // Ray at centre of beam covering upsampler samples
} pixel_s;


typedef struct
{
  vec3f_s topleft, delta_x, delta_y, camera_pos;
  uint2 shape;
  // Upsampling
  uint32_t upsample2;
  float inv_upsample2;
  vec3f_s* delta_upsample; // upsample2 deltas into pixels
  heightfield_ray_s* beam_rays; // upsample2 rays
  
  // Updates along coroutine
  uint2 idx;
  vec3f_s ray_disp_i;
  uint32_t i_pix;

  pixel_s pixels[4]; // 2x2 block of pixels

#ifndef NDEBUG
  uint32_t beam_steps; // Total steps of all beams
  uint32_t super_beam_steps;
#endif

} pixel_ray_grid_s;

static void _pixel_fill_quad_upsamplers(const vec3f_s quad_topleft, const quadtree_cursor_s* cursor, pixel_ray_grid_s* pix, const quadtree* tree)
{
  pix->pixels[0].ray_disp = quad_topleft;
  pix->pixels[1].ray_disp = vec_add(quad_topleft, pix->delta_x);
  pix->pixels[2].ray_disp = vec_add(quad_topleft, pix->delta_y);
  pix->pixels[3].ray_disp = vec_add(pix->pixels[1].ray_disp, pix->delta_y);    
  
  for (unsigned i=0; i<4; ++i)
  {
    fill_beam_and_rays(pix->delta_upsample, &pix->pixels[i].ray_disp, pix->upsample2, pix->inv_upsample2, &pix->beam_rays[i*pix->upsample2], tree ? &pix->pixels[i].mid_ray : 0, tree ?  &pix->pixels[i].beam : 0);
  }
  // Build the super beam
  heightfield_beam_coroutine_s super_beam_walk;
  const int use_beam = tree->manhattan_bound.bounds != 0;
  const int use_super_beam = use_beam; // Can manually disable
  
  if (use_super_beam)
  {
    // Copy initial beam
    heightfield_beam_s quad_beam = pix->pixels[0].beam;

    for (unsigned i=1; i<4; ++i)
    {
      heightfield_beam_enlarge_to_beam(&pix->pixels[i].beam, &quad_beam);
    }


    heightfield_ray_s mid_ray;
    heightfield_ray_init(pix->pixels[3].ray_disp.xyz, &mid_ray); // Top left of final pixel in quad is approx middle of beam

    // Walk this large beam
    heightfield_trace_beam_to_last_unobstructed_crossing(cursor, pix->camera_pos.xyz, &mid_ray, &quad_beam, tree, &super_beam_walk);

#ifndef NDEBUG
    assert(super_beam_walk.count > 0);
    pix->super_beam_steps += super_beam_walk.count;
#endif
  }
  
  for (unsigned i=0; i<4; ++i)  
  {
    // Walk the beams
    upsample_coroutine_start(&pix->beam_rays[i*pix->upsample2], use_beam ? &pix->pixels[i].beam : 0, use_beam ? &pix->pixels[i].mid_ray : 0, cursor, pix->camera_pos.xyz, pix->upsample2, tree,
			     use_super_beam ? &super_beam_walk : 0,
			     &pix->pixels[i].upsampler);
#ifndef NDEBUG
    pix->beam_steps += pix->pixels[i].upsampler.beam_walk.count;
#endif
  }
}

[[nodiscard]] static int pixel_ray_grid_init(pixel_ray_grid_s* pix, const uint2 shape, const vec3f_s topleft, const vec3f_s delta_i, const vec3f_s delta_j, const vec3f_s camera_pos, const int n_upsample, const float inv_upsample)
{
  pix->topleft = topleft;
  pix->delta_y = delta_i;
  pix->delta_x = delta_j;
  pix->camera_pos = camera_pos;
  
  pix->shape = shape;
  pix->upsample2 = (uint32_t)(n_upsample * n_upsample);  
  pix->delta_upsample = (vec3f_s*)malloc(pix->upsample2 * sizeof(vec3f_s));
  if (!pix->delta_upsample) return 0;
  
  pix->beam_rays = (heightfield_ray_s*)malloc(4 * pix->upsample2 * sizeof(heightfield_ray_s));  
  if (!pix->beam_rays)
  {
    free(pix->delta_upsample);
    return 0;
  }
  
  for (int iy = 0, i=0; iy < n_upsample; ++iy)
  {
    const float fy = iy * inv_upsample;
    const vec3f_s frac_y = {delta_i.x * fy, delta_i.y * fy, delta_i.z * fy};
    for (int ix = 0; ix < n_upsample; ++ix, ++i)
    {
      pix->delta_upsample[i] = vec_fma(frac_y, ix * inv_upsample, delta_j);
    }
  }
  pix->inv_upsample2 = inv_upsample * inv_upsample;
  return 1;
}

// Called implicitly when pixel_ray_grid_next fails
static void _pixel_ray_grid_stop(pixel_ray_grid_s* pix)
{
  // Free internal data
  free(pix->beam_rays);
  free(pix->delta_upsample);
}

static pixel_s* pixel_ray_grid_start(pixel_ray_grid_s* pix, const quadtree_cursor_s* cursor, const quadtree* tree)
{
  // Start pos
  pix->idx = (uint2){0,0};
  pix->i_pix = 0;
  pix->ray_disp_i = pix->topleft;
#ifndef NDEBUG
  pix->beam_steps = 0; // Total steps of all beams
  pix->super_beam_steps = 0; // Total steps of super beams  
#endif
  
  _pixel_fill_quad_upsamplers(pix->ray_disp_i, cursor, pix, tree);
  return &pix->pixels[0];
}

static pixel_s* pixel_ray_grid_next(pixel_ray_grid_s* restrict pix, const quadtree_cursor_s* cursor, const quadtree* restrict tree)
{
  // Four pixels at a time
  
  // End of row
  const uint32_t next_j = pix->idx.j + 1;
  // Next in row
  if ((next_j & 1) && next_j != pix->shape.j)
  {
    pix->idx.j = next_j;
    ++pix->i_pix;
    return &pix->pixels[(pix->idx.i & 1)<<1 | 1];
  }
  // Finished row, move to next

  const uint32_t next_i = pix->idx.i + 1;
  // Next row
  if ((next_i & 1))
  {
    pix->idx.j &= ~1u;    
    if (next_i != pix->shape.i)
    {
      pix->idx.i = next_i;
      pix->i_pix = next_i * pix->shape.j + pix->idx.j;
      return &pix->pixels[2];
    }
  }

  // Finished block of 4, move to next
  vec3f_s quad_topleft;
  if (next_j == pix->shape.j)
  {
    pix->idx.j = 0;
    pix->idx.i = next_i;
    if (next_i == pix->shape.i)
    {
      // Finished final row
      _pixel_ray_grid_stop(pix);
      return 0;
    }
    pix->ray_disp_i = vec_fma(pix->topleft, (float)next_i, pix->delta_y);
    quad_topleft = pix->ray_disp_i;
  }
  else
  {
    pix->idx.j = next_j;
    pix->idx.i &= ~1u;
    quad_topleft = vec_fma(pix->ray_disp_i, (float)next_j, pix->delta_x);
  }
  _pixel_fill_quad_upsamplers(quad_topleft, cursor, pix, tree);
  pix->i_pix = pix->idx.i * pix->shape.j + pix->idx.j;  
  return &pix->pixels[0];  
}

// Like get_shadow_lit_image, but raycasting from a camera position
int raytrace_and_shadow_light(const float* restrict camera_pos, const int num_x, const int num_y, const int upsample, const float* restrict ray_disps,
			      const float x_scale, const quadtree* restrict tree, const float* heights,
			      const float* restrict light_dir, const float* light_color, const float* sky_color, uint32_t* restrict out_colors, float* restrict hit_fracs)
{
  const float length = tree->shape.i+tree->shape.j;
  
  heightfield_ray_s shadow_ray;
  {
    const float light_disp[3] = {light_dir[0]*length, light_dir[1]*length, light_dir[2]*length};
    heightfield_ray_init(light_disp, &shadow_ray);
  }
  
  const float inv_l = 1.0f / sqrtf((light_dir[0]*light_dir[0] + light_dir[1]*light_dir[1])*x_scale*x_scale + light_dir[2]*light_dir[2]);
  const vec3f_s light_dir_scale = {light_dir[0]*x_scale*inv_l, light_dir[1]*x_scale*inv_l, light_dir[2]*inv_l};

  const int upsample2 = upsample * upsample;

  // Collection of rays
  const int use_acceleration_beam = tree->manhattan_bound.bounds != 0;

      
  pixel_ray_grid_s pixels;
  if (!pixel_ray_grid_init(&pixels, (uint2){(uint32_t)num_y, (uint32_t)num_x},
		       (vec3f_s){ray_disps[0], ray_disps[1], ray_disps[2]}, // topleft
		       (vec3f_s){ray_disps[6], ray_disps[7], ray_disps[8]}, // y im coord
		       (vec3f_s){ray_disps[3], ray_disps[4], ray_disps[5]}, // x im coord
		       (vec3f_s){camera_pos[0], camera_pos[1], camera_pos[2]}, // camera pos
			   upsample, 1.0f / upsample)) return 1;

  const float3 sky = {sky_color[0], sky_color[1], sky_color[2]};
  const float3 direct_light = {light_color[0], light_color[1],light_color[2]};

#ifndef NDEBUG
  uint64_t ray_steps=0, num_rays = (uint64_t)(num_x * num_y * upsample2);
  if (use_acceleration_beam)
  {
    const float required_manhattan_rho = light_dir[2] / (fabsf(light_dir[0]) + fabsf(light_dir[1]));
    if (required_manhattan_rho < tree->manhattan_rho)
    {
      printf("WARNING: Light direction in grid space (%f, %f, %f) requires Manhattan rho %f, but tree built with %f, shadows will be slow\n", light_dir[0], light_dir[1], light_dir[2], required_manhattan_rho, tree->manhattan_rho);
    }
    else
    {
      printf("Lighting requires Manhattan rho %f, tree is built with rho=%f, shadows will use Manhattan bounds\n", required_manhattan_rho, tree->manhattan_rho);      
    }
  }
#endif

  const quadtree_cursor_s cursor = init_cursor(tree->shape.i > tree->shape.j ? (int)tree->shape.i : (int)tree->shape.j);  
  for (pixel_s* pix = pixel_ray_grid_start(&pixels, &cursor, tree);
       pix;
       pix = pixel_ray_grid_next(&pixels, &cursor, tree))
  {
    // Exposure in RGB
    float3 exposure = {0.0f, 0.0f, 0.0f};
    upsample_coroutine_s* upsampler = &pix->upsampler;
    
    for (int i = 0; i < upsample2; i = (int)++upsampler->idx)
    {
      vec3f_s ray_pos = pixels.camera_pos; // Take a copy
      
      heightfield_hit_s hit = {{0.f, 0.f}, 0.f}; // These gradients are in the frame of the heightfield, which can be scaled    
      // warm start
      int intersect = 0;
      heightfield_ray_coroutine_s rc;

      if (use_acceleration_beam)
      {
	heightfield_beam_warm_start_ray_coroutine(&upsampler->beam_walk, &upsampler->beam_rays[i], &rc);
	intersect = heightfield_intersect_warm_start(ray_pos.xyz, &upsampler->beam_rays[i], &rc, tree, heights, &hit);
      }
      else
      {
#ifndef NDEBUG
	rc.qrc.steps = 0;
#endif
	intersect = heightfield_intersect(&cursor, ray_pos.xyz, &upsampler->beam_rays[i], &rc, tree, heights, &hit);
      }
      if (intersect)
      { 
	if (hit_fracs) hit_fracs[pixels.i_pix] = hit.time;
	// Found a facet, ray_pos has now been updated to be on the surface
	const pseudovec3f_s norm_scale = norm_from_grad(hit.grad, x_scale);
	
#ifndef NDEBUG
	assert(!use_acceleration_beam || (upsampler->beam_walk.count + pixels.super_beam_steps) >= 1);
	
	ray_steps += rc.qrc.steps;
#endif
	const float norm_dot_light = dot(norm_scale, light_dir_scale);
	
	// Facing the light direction, check for illumination
	if (norm_dot_light > 0.0f)
	{
	  // Not in shadow
	  heightfield_hit_s dummy_hit;
	  heightfield_ray_coroutine_s shadow_rc;
	  if (!heightfield_intersect(&rc.qrc.cursor, ray_pos.xyz, &shadow_ray, &shadow_rc, tree, heights, &dummy_hit))
	  {
	    // Add direct light contribution
	    exposure = float3_fma(exposure, norm_dot_light, direct_light);
	  }
	}

	// Sky color always present
	exposure = float3_fma(exposure, norm_scale.z, (float3){sky_color[0], sky_color[1], sky_color[2]});
      }
      else // If out of bounds assume a vertical surface, no shadows
      {
#ifndef NDEBUG
	assert(!use_acceleration_beam || (upsampler->beam_walk.count + pixels.super_beam_steps) >= 1);
	ray_steps += rc.qrc.steps;
#endif
	
	if (i==0)
	{
	  // Add direct light contribution
	  exposure = float3_fma(exposure, light_dir_scale.z* upsample2, // Only calculate the direct contribution on 1 sample, but averaging over upsample2, so over-weight
				direct_light);
	}
	exposure = float3_add(exposure, sky);
      }
    }
    
    // Scale by upsampling
    for (int k=0;k<3;++k)
      exposure.abc[k] *= pixels.inv_upsample2;
    
    out_colors[pixels.i_pix] = 0xFF000000;
    for (int k=0;k<3;++k)
    {
      const int f = (int)floorf(256.0f * (1.0f - exp(-exposure.abc[k])));
      const uint32_t cpt = f < 0 ? 0 : (f > 255 ? 255 : (uint32_t)f);
      out_colors[pixels.i_pix] |= cpt << (k*8);
    }
  } 

#ifndef NDEBUG
  printf("Average rays took %.3f steps/ray, beams took %.3f, (saving %.2f%% of total), super beams %.3f\n",
	 (float)ray_steps/(float)num_rays, (float)pixels.beam_steps/(float)(num_x*num_y),
	 (float)(pixels.beam_steps * (uint64_t)upsample2) * 100.f / (float)(ray_steps + (uint64_t)upsample2 * pixels.beam_steps), (float)pixels.super_beam_steps*4.f/(float)(num_x*num_y));
#endif
  
  return 0;
}

// Like raytrace_and_shadow_light but with ambient rays
int raytrace_and_shadow_and_ambient_light(const float* restrict camera_pos, const int num_x, const int num_y, const int upsample, const float* restrict ray_disps,
			      const float x_scale, const quadtree* restrict tree, const float* heights,
			      const float* restrict light_dir, const float* light_color, const float* sky_color, uint32_t* restrict out_colors, float* restrict hit_fracs)
{
  const quadtree_cursor_s cursor = init_cursor(tree->shape.i > tree->shape.j ? (int)tree->shape.i : (int)tree->shape.j);
  const float length = tree->shape.i+tree->shape.j;
  
  heightfield_ray_s shadow_ray;
  {
    const float light_disp[3] = {light_dir[0]*length, light_dir[1]*length, light_dir[2]*length};
    heightfield_ray_init(light_disp, &shadow_ray);
  }
  
  const float inv_l = 1.0f / sqrtf((light_dir[0]*light_dir[0] + light_dir[1]*light_dir[1])*x_scale*x_scale + light_dir[2]*light_dir[2]);
  const vec3f_s light_dir_scale = {light_dir[0]*x_scale*inv_l, light_dir[1]*x_scale*inv_l, light_dir[2]*inv_l};

  const int upsample2 = upsample * upsample;
  // Displacement delta for rays in y and x
  const vec3f_s topleft = {ray_disps[0], ray_disps[1], ray_disps[2]};  
  const vec3f_s delta_y = {ray_disps[6], ray_disps[7], ray_disps[8]};
  const vec3f_s delta_x = {ray_disps[3], ray_disps[4], ray_disps[5]};

  vec3f_s delta_upsample[upsample2];
  
  const float inv_upsample = 1.0f / (float)upsample;  
  for (int iy = 0, i=0; iy < upsample; ++iy)
  {
    const float fy = iy * inv_upsample;
    const vec3f_s frac_y = {delta_y.x * fy,delta_y.y * fy,delta_y.z * fy};
    for (int ix = 0; ix < upsample; ++ix, ++i)
    {
      delta_upsample[i] = vec_fma(frac_y, ix * inv_upsample, delta_x);
    }
  }
  const float inv_upsample2 = inv_upsample * inv_upsample;
  const float rootT = sqrtf(1.0f/3.0f);
  const float rootH = sqrtf(1.0f/2.0f);  
  const float sphere[26*3] = {rootT,rootT,rootT, rootT,rootT,-rootT, rootT,-rootT,rootT, rootT,-rootT,-rootT,
                              -rootT,rootT,rootT, -rootT,rootT,-rootT, -rootT,-rootT,rootT, -rootT,-rootT,-rootT,
			      0,0,1, 0,1,0, 1,0,0, 0,0,-1, 0,-1,0, -1,0,0,
			      -rootH,-rootH,0, -rootH,0,-rootH, 0,-rootH,-rootH,
			      rootH,rootH,0, rootH,0,rootH, 0,rootH,rootH,     
			      rootH,-rootH,0, rootH,0,-rootH, 0,rootH,-rootH,
			      -rootH,rootH,0, -rootH,0,rootH, 0,-rootH,rootH };

  const float3 sky = {sky_color[0], sky_color[1], sky_color[2]};
  const float3 direct_light = {light_color[0], light_color[1],light_color[2]};
  const float ao_weight = (float)upsample2/(float)(upsample2-1); // ambient occlusion inverse fraction of rays
  
  for (int i_ray=0, iy=0; iy < num_y; ++iy)
  {
    const vec3f_s ray_disp_i = vec_fma(topleft, (float)(iy), delta_y);
    
    for (int ix=0; ix < num_x; ++ix, ++i_ray)
    {
      // Exposure in RGB
      float3 exposure = {0.0f, 0.0f, 0.0f};
      // Collection of rays
      heightfield_beam_s beam;
      for (int i_upsample = 0; i_upsample < upsample2; ++i_upsample)
      {
	const vec3f_s ray_disp = vec_add(delta_upsample[i_upsample],
					 vec_fma(ray_disp_i, (float)ix, delta_x)); 
	  
	float ray_pos[3] = {camera_pos[0], camera_pos[1], camera_pos[2]};
	
	heightfield_hit_s hit = {{0.f,0.f}, 0.f}; // These gradients are in the frame of the heightfield, which can be scaled    
	heightfield_ray_s ray;	
	heightfield_ray_init(ray_disp.xyz, &ray);

	// TODO make array
	heightfield_beam_init_ray(&ray, &beam);

	heightfield_ray_coroutine_s rc;
	if (heightfield_intersect(&cursor, ray_pos, &ray, &rc, tree, heights, &hit))
	{ 
	  if (hit_fracs) hit_fracs[i_ray] = hit.time;
	  // Found a facet, ray_pos has now been updated to be on the surface
	  const pseudovec3f_s norm_scale = norm_from_grad(hit.grad, x_scale);

	  if (i_upsample==0)
	  {
	    const float norm_dot_light = dot(norm_scale, light_dir_scale);
	    
	    // Facing the light direction, check for illumination
	    if (norm_dot_light > 0.0f)
	    {
	      // Now check to see if we are in shadow
	      
	      // Not in shadow
	      heightfield_hit_s dummy_hit;
	      if (!heightfield_intersect(&rc.qrc.cursor, ray_pos, &shadow_ray, &rc, tree, heights, &dummy_hit))
	      {
		// Add direct light contribution
		exposure = float3_fma(exposure, norm_dot_light * upsample2, // Only calculate the direct contribution on 1 sample, but averaging over upsample2, so over-weight
				      direct_light);
	      }
	    }
	  }
	  else
	  {
	    // Attempt a new ray
	    // Random on sphere:
	    int k = (i_upsample - 1);
	    while (k >= 26) k-=26;
	    
	    const float3 s = {sphere[3*k],sphere[3*k+1],sphere[3*k+2]};
	    vec3f_s dir_scale;
	    const float weight = lambertian_restricted(s, (float3){0.f,0.f,1.0f}, norm_scale.f3, &dir_scale.f3);

	    if (weight > 0.f)
	    {
#ifndef NDEBUG
	      if (!(dir_scale.z >=0.f))
	      {
		printf("s %f %f %f \n", s.abc[0], s.abc[1], s.abc[2]);
		printf("n %f %f %f \n", norm_scale.x, norm_scale.y, norm_scale.z);      
		printf("dir_scale %f %f %f \n", dir_scale.x, dir_scale.y, dir_scale.z);
		printf("w %f", weight);
		assert(0);
		
	      }
	      assert(dir_scale.z >=0.f); // Should always be upward facing
#endif
	      // Back scale (should've been a pseudovector)	    
	      const vec3f_s disp = vec_mul(length, (vec3f_s){dir_scale.x, dir_scale.y, dir_scale.z*x_scale}); // Search more than 1 unit
	      // Brighter in vertical direction
//	      const float cos_sky = float3_inner(dir_scale.f3, light_dir_scale.f3);
	      const float rayleigh = 1.f;//0.5f * (1.f + cos_sky*cos_sky);
	      
	      // In shadow, remove light contribution
	      heightfield_hit_s dummy_hit;
	      heightfield_ray_s ambient_ray;
	      heightfield_ray_init(disp.xyz, &ambient_ray);
	      if (!heightfield_intersect(&rc.qrc.cursor, ray_pos, &ambient_ray, &rc, tree, heights, &dummy_hit))
	      {
		exposure = float3_fma(exposure, weight * rayleigh * ao_weight, sky);
	      }
	    }
	  }
	}
	else // If out of bounds assume a vertical surface, no shadows
	{
	  if (i_upsample==0)
	  {
	    // Add direct light contribution
	    exposure = float3_fma(exposure, light_dir_scale.z* upsample2, // Only calculate the direct contribution on 1 sample, but averaging over upsample2, so over-weight
				  direct_light);
	  }
	  exposure = float3_add(exposure, sky);

	}
      }
    
      // Scale by upsampling
      for (int k=0;k<3;++k)
	exposure.abc[k] *= inv_upsample2;
      
      out_colors[i_ray] = 0xFF000000;
      for (int k=0;k<3;++k)
      {
	const int f = (int)floorf(256.0f * (1.0f - exp(-exposure.abc[k])));
	const uint32_t cpt = f < 0 ? 0 : (f > 255 ? 255 : (uint32_t)f);
	out_colors[i_ray] |= cpt << (k*8);
      }
    }
  }
  return 0;
}
