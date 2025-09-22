#include "heightfield_ray.h" // quadtree_cursor, heightfield_ray
#include "heightfield_cursor.h"

#include <float.h> // FLT_MIN

// Unsafe collection of exact height at point (point must be in heightfield bounds)
[[maybe_unused]]
static float facet_height(const float x, const float y, const quadtree* tree, const float* heights, float* restrict grad)
{
  const triangle_index t = get_triangle_index(x,y);
  const uint32_t quad_i = (uint32_t)t.i,
    quad_j = (uint32_t)(t.v>>1);
  
  const uint32_t i0 = quad_i, i1 = i0+1, j0 = quad_j, j1 = j0+1,
    is_upper = t.v & 1;

  const uint32_t size_j = tree->shape.j;
  const float z00 = heights[i0 * size_j + j0], z01 = heights[i0 * size_j + j1],
    z10 = heights[i1 * size_j + j0], z11 = heights[i1 * size_j + j1];

  // x0 in the frame of the bottom-left quad corner
  const float x0_rel_bl[2] = {x - quad_i,
                        y - quad_j};

  assert(x0_rel_bl[0] <= 1.0f && x0_rel_bl[0] >= 0.0f);
  assert(x0_rel_bl[1] <= 1.0f && x0_rel_bl[1] >= 0.0f);
  // Triangle gradient
  grad[0] = is_upper ? z11 - z01 : z10 - z00;
  grad[1] = is_upper ? z01 - z00 : z11 - z10;
  
  const float res = grad[0]*x0_rel_bl[0] + grad[1]*x0_rel_bl[1] + z00;
  return res;
}

// If the ray from x0 to x0 + dx*(*t_exit) intersects the current facet (t) before t_exit, return 1 and set t_exit, x0 (to the accurate hit position), and height_grad
static inline int cursor_facet_intersect(const grid_size_s height_size, const int is_periodic, const triangle_index t, const float* restrict heights, 
					 float* restrict x0, const float* restrict dx, const float t_enter, float* restrict t_exit, heightfield_gradient* restrict height_grad)
{
  // Check for reflection against the current triangle
  const int32_t pos_i = t.i,
    pos_j = t.v>>1; // Note require signed right shift that keeps sign
  assert(is_periodic || ((uint32_t)pos_i +1 < height_size.i && (uint32_t)pos_j+1 < height_size.j));
  assert(is_periodic || ((uint32_t)pos_i < height_size.i && (uint32_t)pos_j < height_size.j));

  // Wrapping if necc
  const uint32_t periodic_masks[2] = {height_size.i-1, height_size.j-1};
  const uint32_t i0 = is_periodic ? (uint32_t)pos_i & periodic_masks[0] : (uint32_t)pos_i,
    j0 = is_periodic ? (uint32_t)pos_j & periodic_masks[1] : (uint32_t)pos_j;  
  
  const uint32_t i1 = is_periodic ? (i0+1) & periodic_masks[0] : i0+1;
  const uint32_t j1 = is_periodic ? (j0+1) & periodic_masks[1] : j0+1,
    is_upper = t.v & 1;
  
  const float z00 = heights[i0 * height_size.j + j0], z01 = heights[i0 * height_size.j + j1],
    z10 = heights[i1 * height_size.j + j0], z11 = heights[i1 * height_size.j + j1];

  // (x,y) components of perpendicular (facing upwards) to triangle (z cpt =1 implicit), 
  const heightfield_gradient grad = {is_upper ? z11-z01 : z10 - z00,
			     is_upper ? z01-z00 : z11 - z10};

  // dx.tri_perp
  const float dx_dot_minus_perp = grad.dh_di * dx[0] + grad.dh_dj * dx[1] - dx[2];

  // Not entering triangle, nothing to do
  if (dx_dot_minus_perp <= FLT_MIN) return 0;
  
  // Position at cursor exit
  const float t_triangle_exit = *t_exit;
  const float x_exit[3] = {x0[0] + t_triangle_exit * dx[0], x0[1] + t_triangle_exit * dx[1], x0[2] + t_triangle_exit * dx[2]};
  
  // Triangle bottom-left
  const float tri_bl[3] = {(float)pos_i, (float)pos_j, z00};
  
  // x_exit relative to triangle bottom-left
  const float x_exit_rel_bl[3] = {x_exit[0] - tri_bl[0], x_exit[1] - tri_bl[1], x_exit[2] - tri_bl[2]};

  // If the cursor exit is still above the plane, nothing to do
  // Note that we form this dot-product from x_exit_rel_bl, using dot(x0_rel_bl,perp) + t_triangle_exit*dot(dx, perp) is not accurate enough
  const float dist_exit_above = x_exit_rel_bl[2] - grad.dh_di * x_exit_rel_bl[0] - grad.dh_dj * x_exit_rel_bl[1];
  if (dist_exit_above > -FLT_MIN) return 0;

  // x0 in the frame of the bottom-left quad corner
  const float x0_rel_bl[3] = {x0[0] - tri_bl[0], x0[1] - tri_bl[1], x0[2] - tri_bl[2]};

  const float dist_x0_above = x0_rel_bl[2] - grad.dh_di * x0_rel_bl[0] - grad.dh_dj * x0_rel_bl[1];

  // Hit outside triangle
  if (dist_x0_above >= t_triangle_exit * dx_dot_minus_perp) return 0;

  // Hit before entering triangle (t_enter)
  if (dist_x0_above < dx_dot_minus_perp * t_enter)
  {

    // Should only happen if we were extremely close to triangle in the normal direction
    // or along the ray parameter
#ifndef NDEBUG
    const float t_cross = dist_x0_above / dx_dot_minus_perp;
    const float dist2_x0_bl = x0_rel_bl[0]*x0_rel_bl[0] + x0_rel_bl[1]*x0_rel_bl[1] + x0_rel_bl[2]*x0_rel_bl[2];
    if (!(dist_x0_above*dist_x0_above <= 1e-5f*1e-5f*dist2_x0_bl || t_cross + 1e-3f > t_enter))
    {
      // Suspicious ray values
      printf("t_cross = %f outside triangle t_tri = [%f,%f]\n", t_cross, t_enter, t_triangle_exit);
      print_triangle(t);
      printf("  t=%f with t_exit=%f ray_dir (%f,%f,%f) dx_dot_perp %f x_rel_perp %f \n", t_enter, t_triangle_exit,
	   dx[0],dx[1],dx[2],-dx_dot_minus_perp, dist_x0_above);

      printf("  Ray start (%f,%f,%f) dir (%f,%f,%f)\n", x0[0],x0[1],x0[2], dx[0],dx[1],dx[2]);
      printf("  Triangle exit x(t=%f) = (%f,%f,%f)\n", t_triangle_exit, x_exit[0], x_exit[1], x_exit[2]);
      printf("  Triangle entry x(t=%f) = (%f,%f,%f)\n", t_enter, x0[0] + t_enter * dx[0], x0[1] + t_enter * dx[1], x0[2] + t_enter * dx[2]);
      if (is_upper)
      {
	printf("z[%d,%d] = %f, z[%d,%d] = %f, z[%d,%d] = %f\n", (int)i0, (int)j0, z00,
	       (int)i0, (int)j1, z01,
	       (int)i1, (int)j1, z11);
      }
      else
      {
	printf("z[%d,%d] = %f, z[%d,%d] = %f, z[%d,%d] = %f\n", (int)i0, (int)j0, z00,
	       (int)i1, (int)j0, z10,
	       (int)i1, (int)j1, z11);
      }	
      printf("t=%f at index %d,%d from %f,%f \n", *t_exit, (int)i0, (int)j0, x0[0],x0[1]);
      printf("Ray (x0 - tri_corner).perp = %f, (x_exit - tri_corner).perp=%f\n", dist_x0_above, dist_exit_above);
      assert(0);
    }
#endif

    // Use the entry point
    *t_exit = t_enter;
    
  }
  else
  {
    // Solve for the hit
    *t_exit = dist_x0_above / dx_dot_minus_perp;
  }
  
  x0[0] = x0[0] + (*t_exit) * dx[0];
  x0[1] = x0[1] + (*t_exit) * dx[1];
  // Height is special, use exact facet height
  x0[2] = tri_bl[2] + (x0[0] - tri_bl[0])*grad.dh_di + (x0[1] - tri_bl[1])*grad.dh_dj;
  
  *height_grad = grad;
  return 1;
}

static inline void heightfield_ray_coroutine_start(const quadtree_cursor_s* cursor, const float* restrict ray_x0, const heightfield_ray_s* restrict ray, heightfield_ray_coroutine_s* restrict hrc)
{
  quadtree_ray_coroutine_start(&hrc->qrc, cursor, (float2){ray_x0[0], ray_x0[1]}, &ray->tq_march);
  // Heights
  hrc->z_cell = (float2){ray_x0[2],
    ray_x0[2] + hrc->qrc._next.time * ray->dz}; // Height we exit cell
  hrc->z0 = ray_x0[2];
}

void heightfield_beam_warm_start_ray_coroutine(const heightfield_beam_coroutine_s* restrict beam, const heightfield_ray_s* restrict ray, heightfield_ray_coroutine_s* restrict hrc)
{
  quadtree_beam_warm_start_ray_coroutine(&hrc->qrc, &ray->tq_march, &beam->qbc);

  // Heights
  hrc->z_cell = (float2){beam->z0 + hrc->qrc.last_time * ray->dz,
                         beam->z0 + hrc->qrc._next.time * ray->dz}; // Height we exit cell
  hrc->z0 = beam->z0;
}

static inline int heightfield_ray_coroutine_next(const enum EQRC refine, const heightfield_ray_s* restrict ray, heightfield_ray_coroutine_s* hrc)
{
  if (!quadtree_ray_coroutine_next(refine, &ray->tq_march, &hrc->qrc)) return 0;

  if (!refine) // continue
  {
    // Heights
    hrc->z_cell.v[0] = hrc->z_cell.v[1];
  }
  // TODO any checks on crossing corners?
  hrc->z_cell.v[1] = hrc->z0 + max_2f(hrc->qrc._next.time, hrc->qrc.last_time) * ray->dz; // Height we exit cell
  return 1;
}

int raytrace_shadows(const quadtree* restrict tree, const float* restrict heights, 
		     const float* restrict light_dir, int32_t* restrict is_shadow)
{
  /*
    Find those elements that are in shadow from the given direction (pointing outwards/upwards)
   */
  const int N = (int)tree->shape.i, M = (int)tree->shape.j;

  quadtree_cursor_s cursor = init_cursor(N > M ? N : M);
  const float length = N+M;

  // Baked light ray
  heightfield_ray_s light_ray;
  const float ray_dir[3] = {light_dir[0]*length, light_dir[1]*length, light_dir[2]*length};  
  heightfield_ray_init(ray_dir, &light_ray);
  
  for (int i=0, idx=0; i<N; ++i)
  {
    float ray_pos[3];
    
    for (int j=0; j<M; ++j, ++idx)
    {
      ray_pos[0] = i;      
      ray_pos[1] = j;
      ray_pos[2] = heights[i*M+j];
#ifndef NDEBUG
//      printf("Ray %d from pos (%.3f,%.3f)\n", idx, ray_pos[0], ray_pos[1]);    
#endif
      heightfield_hit_s hit = {{0.f, 0.f}, 0.f};
      heightfield_ray_coroutine_s hrc;      
      is_shadow[idx] = heightfield_intersect(&cursor, ray_pos, &light_ray, &hrc, tree, heights, &hit);
    }
  }
  return 0;
}

void heightfield_beam_init_ray(const heightfield_ray_s* restrict ray, heightfield_beam_s* restrict beam)
{
  // Horizontal cpts
  tq_beam_init_ray(&ray->tq_march, &beam->tri_qtree_beam);
  
  // z-cpt
  beam->min_dz = ray->dz;
  for (unsigned axis=0;axis<2;++axis)
    beam->d_min_z_d_xy.v[axis] = ray->dz * ray->tq_march.inv_disp[axis];
}

void heightfield_beam_enlarge_to_beam(const heightfield_beam_s* restrict other, heightfield_beam_s* restrict beam)
{
  tq_beam_add_beam(&other->tri_qtree_beam, &beam->tri_qtree_beam); // Horizontal cpts
  beam->min_dz = min_2f(beam->min_dz, other->min_dz); // z cpt

  for (unsigned axis=0;axis<2;++axis)
  {
    // If moving in positive direction (d_pos[axis]==1), update with smaller value.
    // If moving in negative (d_pos[]==0), update with the larger.
    // If moving degenerate (d_pos[]==2), it doesnt matter as this ignored
    if ((beam->d_min_z_d_xy.v[axis] > other->d_min_z_d_xy.v[axis]) == beam->tri_qtree_beam.d_pos[axis])
    {
      beam->d_min_z_d_xy.v[axis] = other->d_min_z_d_xy.v[axis];
    }
  }
}

void heightfield_beam_enlarge_to_ray(const heightfield_ray_s* ray, heightfield_beam_s* beam)
{
  heightfield_beam_s other;
  heightfield_beam_init_ray(ray, &other);
  heightfield_beam_enlarge_to_beam(&other, beam);
}

// Start the beam walk
static void heightfield_beam_coroutine_start(const quadtree_cursor_s* cursor, const float* ray_x0, const heightfield_ray_s* restrict ray, const heightfield_beam_s* beam, heightfield_beam_coroutine_s* hbc)
{
  quadtree_beam_coroutine_start(&hbc->qbc, cursor, (float2){ray_x0[0], ray_x0[1]}, &ray->tq_march, &beam->tri_qtree_beam);

#ifndef NDEBUG
  hbc->count = 0;
#endif
  
  // Heights
  // TODO
  const unsigned side = hbc->qbc.exit.side;
  hbc->z0 = ray_x0[2];
  hbc->z_cell.v[0] = ray_x0[2];
  if (side<2) // x or y axis
  {
    const float last_disp = hbc->qbc.exit.hit; // Either x or y, according to side
    hbc->z_cell.v[1] = ray_x0[2] + last_disp * beam->d_min_z_d_xy.v[side]; // Lowest point
  }
  else // (side==2) // All rays finish here
  {
    hbc->z_cell.v[1] = ray_x0[2] + beam->min_dz;
  }
}

void heightfield_beam_warm_start_fine_beam(const float* restrict ray_x0, const heightfield_ray_s* restrict canonical_ray, const heightfield_beam_s* restrict fine_beam,
					   const heightfield_beam_coroutine_s* restrict beam_walk, heightfield_beam_coroutine_s* restrict fine_beam_walk)
{
  quadtree_beam_warm_start_fine_beam(&canonical_ray->tq_march, &fine_beam->tri_qtree_beam, &beam_walk->qbc, &fine_beam_walk->qbc);

#ifndef NDEBUG
  fine_beam_walk->count = 0;
#endif
  
  // Heights
  const unsigned side = fine_beam_walk->qbc.exit.side;
  fine_beam_walk->z0 = ray_x0[2]; // TODO nicer if this was relative too
  fine_beam_walk->z_cell.v[0] = ray_x0[2];
  if (side<2) // x or y axis
  {
    const float last_disp = fine_beam_walk->qbc.exit.hit; // Either x or y, according to side
    fine_beam_walk->z_cell.v[1] = ray_x0[2] + last_disp * fine_beam->d_min_z_d_xy.v[side]; // Lowest point
  }
  else // (side==2) // All rays finish here
  {
    fine_beam_walk->z_cell.v[1] = ray_x0[2] + fine_beam->min_dz;
  }
}

static int heightfield_beam_coroutine_next(const enum EQRC refine, const heightfield_ray_s* restrict ray, const heightfield_beam_s* beam, heightfield_beam_coroutine_s* hbc)
{
  // Underlying quadtree beam would finish
  if (!quadtree_beam_coroutine_next(refine, &ray->tq_march, &beam->tri_qtree_beam, &hbc->qbc)) return 0;
  
  if (!refine) // continue
  {
    hbc->z_cell.v[0] = hbc->z_cell.v[1];
  }
#ifndef NDEBUG
    ++(hbc->count);
#endif
  
  // Set the exit (TODO func with _start)
  const unsigned side = hbc->qbc.exit.side;
  if (side<2) // x or y axis
  {
    const float last_disp = hbc->qbc.exit.hit; // Either x or y, according to side
    hbc->z_cell.v[1] = hbc->z0 + last_disp * beam->d_min_z_d_xy.v[side]; // Lowest point
  }
  else // (side==2) // All rays finish here
  {
    hbc->z_cell.v[1] = hbc->z0 + beam->min_dz;
  }
  
  return 1;
}

static inline const quadtree_cursor_s* _beam_cursor(const heightfield_beam_coroutine_s* hbc)
{
  return &hbc->qbc.qrc.cursor;
}

// Whether the heightfield beam over this quad intersects the corresponding Manhattan pyramid bound between entry and exit
static inline int _beam_overlaps_manhattan_pyramid_bounds(const quadtree_cursor_s* restrict cursor, const float2 x0, const float2 z_bound_entry_exit,
							  const _beam_side_exit* restrict beam_entry, const _beam_side_exit* restrict beam_exit, const quadtree* restrict tree, int* beam_hit)
{
  // By convexity, only need to check the entry projection and the exit projection
  // (all intermediate positions are a linear interpolation between an entry and
  // exit position, and since the acceptance shape is convex, if the extremes are
  // permissible then so are the interpolations).

  // Check entry, exit distances from quad
  const float2 dx_entry_exit = beam_max_horiz_dist_outside_quad(cursor, x0, beam_entry, beam_exit);

  const int beam_outside_cursor = dx_entry_exit.v[0] > 0.f || dx_entry_exit.v[1] > 0.f;

  *beam_hit = beam_outside_cursor;
  
  // Beam is entirely within quad => Use the heightfield bound for this quad, this is looser
  if (!beam_outside_cursor)
  {
    // No bound (outside heightfield) continue
    float z_max;    
    if (!cursor_height_bound(cursor, tree, 0, &z_max)) return 0;
    
    return min_2f(z_bound_entry_exit.v[1], z_bound_entry_exit.v[0]) <= z_max;
  }
  
  // Otherwise have to check pyramid bounds
  const float bound = cursor_manhattan_bound(cursor, tree->periodic, tree->shape, &tree->manhattan_bound);

  return float2_min(float2_fma(z_bound_entry_exit, -tree->manhattan_rho, dx_entry_exit)) <= bound;
}

[[nodiscard]] static inline int _outside_shape_and_escaping(const triangle_index t, const grid_size_s shape, const unsigned* disp_pos)
{
  if (disp_pos[1] == 1 && t.v >= (int32_t)(shape.j * 2)) return 1;
  if (disp_pos[0] == 1 && t.i >= (int32_t)shape.i) return 1;  
  // Note if triangle pos is negative, then the quad cannot cross 0, so we do not need to check width
  if (disp_pos[1] == 0 && t.v < 0) return 1;
  if (disp_pos[0] == 0 && t.i < 0) return 1;  
  
  return 0;
}

void heightfield_trace_beam_warm_start(const float* ray_x0, const heightfield_ray_s* restrict canonical_ray, const heightfield_beam_s* beam, const quadtree* tree, heightfield_beam_coroutine_s* restrict hbc)
{
  // Maximum of entire heightfield, for escape check
  const float max_heightfield_z = get_max_aggregate_bound(&tree->max_height);
  const float2 x0 = {ray_x0[0], ray_x0[1]};
  const quadtree_cursor_s* cursor = _beam_cursor(hbc);
  
  // Do not refine further than most refined bound. Whilst the smaller quads may help us increment,
  // the subsequent rays will likely have to ascend these refined levels.
  const unsigned lost_bit_refine_limit = 1 + tree->manhattan_bound.bound[0].power2;

  // Need a canonical ray that covers the whole beam
  // Actually part would be fine, as long as we can recover finished/unfinished rays if we complete
  do 
  {
    // Refine until we are sure we don't hit a triangle
    int beam_outside_cursor = 0;
    while (_beam_overlaps_manhattan_pyramid_bounds(cursor, x0, hbc->z_cell, &hbc->qbc.entry, &hbc->qbc.exit, tree, &beam_outside_cursor))
    {
      // Allow refinement if beam hits lower bounding plane rather than sides
      if (cursor->lost_bits <= lost_bit_refine_limit && beam_outside_cursor) return;
      
      // Some rays within the beam may have collided with geometry
      if (!heightfield_beam_coroutine_next(ERefine, canonical_ray, beam, hbc)) return;
    };

    if (beam_outside_cursor) continue;

    // Check if we escape entirely
    if (_outside_shape_and_escaping(cursor->t, tree->shape, beam->tri_qtree_beam.d_pos)) return;

    if (cursor_is_root(cursor))
    {
      if (hbc->z_cell.v[1] > max_heightfield_z && // above root node
	  canonical_ray->dz >= 0) return; // escaping
      // Could have extra checks for outside box for very long rays
    }
    
    // Move to next    
  } while (heightfield_beam_coroutine_next(EContinue, canonical_ray, beam, hbc)); 

  // Escaped volume
}

// Like heightfield_intersect, but for an entire beam. We never actually complete, just find the first quad (never triangle) where the entry is clear for
// all rays, but the exit may not be.
void heightfield_trace_beam_to_last_unobstructed_crossing(const quadtree_cursor_s* cursor, const float* ray_x0, const heightfield_ray_s* restrict canonical_ray, const heightfield_beam_s* beam, const quadtree* tree, heightfield_beam_coroutine_s* restrict hbc)
{
  heightfield_beam_coroutine_start(cursor, ray_x0, canonical_ray, beam, hbc);

  heightfield_trace_beam_warm_start(ray_x0, canonical_ray, beam, tree, hbc);
}

// On success (intersection), sets fraction of the displacement before the hit in hit_fraction
int heightfield_intersect_warm_start(float* restrict ray_x0, const heightfield_ray_s* restrict ray, heightfield_ray_coroutine_s* restrict hrc, 
			  const quadtree* restrict tree, const float* restrict heights, heightfield_hit_s* restrict hit)
{
  // Add a getter?
  const float ray_disp[3] = {ray->tq_march.disp[0], ray->tq_march.disp[1], ray->dz};
  
  const float max_heightfield_z = get_max_aggregate_bound(&tree->max_height);

  const int use_manhattan_escape_cone = tree->manhattan_bound.bounds &&
      tree->manhattan_rho * (fabsf(ray_disp[0]) + fabsf(ray_disp[1])) < ray_disp[2]; // Ray is moving upwards faster than Manhattan cone
  const quadtree_cursor_s* cursor = &hrc->qrc.cursor;
  do 
  {
    // Escaping
    if (_outside_shape_and_escaping(cursor->t, tree->shape, ray->tq_march.disp_pos)) return 0;
    
    // Refine until we are sure we don't hit a triangle
    while (cursor_overlaps_height(cursor, tree, heights, hrc->z_cell))
    {
      // Might hit a triangle, refine
      if (heightfield_ray_coroutine_next(ERefine, ray, hrc))
	continue; // Was not a leaf

      // Found a leaf, check if we hit
      float t_hit = hrc->qrc._next.time; // time that we leave triangle
      assert(cursor->lost_bits==0); // Should only be called at the finest level
      const int does_intersect = cursor_facet_intersect(tree->shape, (int)tree->periodic, cursor->t, heights, 
							ray_x0, ray_disp, hrc->qrc.last_time, &t_hit, &hit->grad);
      if (does_intersect)
      {
	hit->time = t_hit;
	return 1;	
      }

      break; // Continue to next facet      
    };

    // Check if we escape entirely    
    if (use_manhattan_escape_cone)
    {
      // Above the bound for this cell
      if (hrc->z_cell.v[1] >= cursor_manhattan_bound(cursor, tree->periodic, tree->shape, &tree->manhattan_bound)) return 0;
    }
    else if (cursor_is_root(cursor))
    {
      if (hrc->z_cell.v[1] > max_heightfield_z && // above root node
	  ray->dz >= 0) return 0; // escaping
      // Could have extra checks for outside box for very long rays
    }
    
    // Move to next
  } while (heightfield_ray_coroutine_next(EContinue, ray, hrc)); 

  // Escaped volume
  return 0;
}

// On success (intersection), sets fraction of the displacement before the hit in hit_fraction
int heightfield_intersect(const quadtree_cursor_s* restrict cursor, float* restrict ray_x0, const heightfield_ray_s* restrict ray, heightfield_ray_coroutine_s* restrict hrc, 
			  const quadtree* restrict tree, const float* restrict heights, heightfield_hit_s* restrict hit)
{
  heightfield_ray_coroutine_start(cursor, ray_x0, ray, hrc);

  return heightfield_intersect_warm_start(ray_x0, ray, hrc, tree, heights, hit);
}
