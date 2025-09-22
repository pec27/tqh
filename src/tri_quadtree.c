#include "tri_quadtree.h"
#include <math.h> // fabsf
#include <float.h> // FLT_MIN
  
static inline int32_t _cursor_width(const quadtree_cursor_s* cursor)
{
  if (!cursor->lost_bits) return 1;
  return 1<<(cursor->lost_bits - 1);
}

static inline int _move_cursor_to_child(quadtree_cursor_s* cursor)
{
  // Returns 1 on success, 0 on leaf (triangle)
  
  if (!cursor->lost_bits) return 0; // Already at max-depth!
  cursor->lost_bits--;
  return 1; // success
}

static inline uint32_t triangle_diff(const triangle_index a, const triangle_index b)
{
  return (((uint32_t)(a.i ^ b.i)<<1) | (uint32_t)(a.v ^ b.v));
}
#ifndef NDEBUG
void print_quadtree_cursor(const quadtree_cursor_s* cursor)
{
  printf("Cursor at lost_bits %d\n  ", cursor->lost_bits);
  print_triangle(cursor->t);
  const int w = _cursor_width(cursor);
  const int keep_mask = ~(w-1); // Keep the top bits
  
  // Position of cell bottom-left relative to ray-start
  const int bl[2] = {cursor->t.i & keep_mask, (cursor->t.v >> 1) & keep_mask};
  printf("  Quad [%d,%d)x[%d,%d)\n", bl[0],bl[0]+w, bl[1],bl[1]+w);
  
}

void print_tq_ray_coroutine(const quadtree_ray_coroutine_data* qrc)
{
  printf("Quadtree ray coroutine x0=(%f, %f)\n entered cell changing %d, at time %f\n"
	 "  exit cell %d at time %f\n", qrc->x0.v[0], qrc->x0.v[1], qrc->last_edge, qrc->last_time, qrc->_next.edge, qrc->_next.time);
  print_quadtree_cursor(&qrc->cursor);
}

void print_tq_beam_coroutine(const tri_quadtree_beam_coroutine_s* beam)
{
  printf("Quadtree beam coroutine\n entered cell changing %d, projection [%f, %f]", beam->entry.side, beam->entry.proj.min, beam->entry.proj.max);
  
  printf("\n  exit changing %d, projection [%f, %f]\n", beam->exit.side, beam->exit.proj.min, beam->exit.proj.max);  
  print_tq_ray_coroutine(&beam->qrc);
}

#endif //NDEBUG
static inline void move_cursor_to_first_common_ancestor(quadtree_cursor_s* cursor, const triangle_index t)
{
  // Move the current cursor to the first common ancestor of (i,j), set this as the cursor fine point
  uint32_t lost_bits = cursor->lost_bits;

  // TODO if any high bits different, force root node?
  
  for (const uint32_t idx_diff = triangle_diff(t, cursor->t) & cursor->ignore_repeats;
       idx_diff >> lost_bits; // Non-zero indicates wrong branch
       lost_bits++);

  cursor->t = t;
  cursor->lost_bits = lost_bits;
}

// Extent of the current cursor quad relative to x0, along a given axis
static inline intervalf_s _cursor_quad_rel_x0_axis(const quadtree_cursor_s* cursor, const float2 x0, const unsigned axis)
{
  assert(axis < 2); // Should only be x or y axis

  /// Width of the cursor
  const int width = _cursor_width(cursor);
  const int keep_mask = ~(width-1); // Keep the top bits

  const int32_t t_axis = axis ? (cursor->t.v >> 1) : cursor->t.i;
      
  // Position of cell bottom-left relative to ray-start
  const float left = (t_axis & keep_mask) - x0.v[axis];

  // Extent along axis
  return (intervalf_s){left, left + (float)width};
}

static inline float _beam_max_horiz_dist_outside_cursor(const quadtree_cursor_s* restrict cursor, const float2 x0, const _beam_side_exit* restrict entry)
{
  const unsigned axis = entry->side;
  if (axis >= 2) return 0.f;
  
  const intervalf_s ortho_proj = _cursor_quad_rel_x0_axis(cursor, x0, 1-axis);
  const float max_dist_outside_quad = interval_max_outside(entry->proj, ortho_proj);
  return max_dist_outside_quad;
}

float2 beam_max_horiz_dist_outside_quad(const quadtree_cursor_s* restrict cursor, const float2 x0, const _beam_side_exit* entry, const _beam_side_exit* beam_exit)
{
  const float2 res = {_beam_max_horiz_dist_outside_cursor(cursor, x0, entry),
    _beam_max_horiz_dist_outside_cursor(cursor, x0, beam_exit)};
  return res;
}

static inline void _set_quadtree_ray_cell_exit(const float2 start, const quadtree_cursor_s* restrict cursor, const _raymarch_data* restrict ray, _cell_exit* restrict next)
{
  /* Update the quadtree_ray_coroutine_data with details of the exit from the
   current cell (quad or triangle)
   qrc->time     - time to exit (from x0,y0)
   qrc->_next_edge - 
     0: change x,
     1: change y,
     2: cross diagonal
     3: finish (reach t_max =1)
  */
  
  const int w = _cursor_width(cursor);
  const int keep_mask = ~(w-1); // Keep the top bits
  
  // Position of cell bottom-left relative to ray-start
  const float bl[2] = {(cursor->t.i & keep_mask) - start.v[0],
		       ((cursor->t.v >> 1) & keep_mask) - start.v[1]};

  const unsigned int not_leaf = cursor->lost_bits; // NB careful, not just 0/1

  // Only used if a leaf
  const unsigned int is_upper = (cursor->t.v & 1);

  // Can we hit the diagonal?
  const unsigned int incl_diag = (ray->dy_m_dx_pos ^ is_upper) & (!not_leaf);

  float t_exit = 1.0f;
  int edge = 3; // Default case finish in cell

  const float dist_diag = bl[1] - bl[0];
  const float t_diag = dist_diag * ray->inv_dy_m_dx;  
  if (incl_diag && t_diag < t_exit)
  {
    t_exit = t_diag;
    edge   = 2; // Crosses the diagonal
  }

  const float delta[2] = {ray->disp_pos[0] ? bl[0] + w : bl[0],
			  ray->disp_pos[1] ? bl[1] + w : bl[1]};
  
  const unsigned int incl_x = (ray->disp_pos[0] ^ is_upper) | not_leaf; // NB not_leaf can be >1, but this is ok
  const unsigned int incl_y = (ray->disp_pos[1] == is_upper) | not_leaf; // NB not_leaf can be >1, but this is ok

  const float t_walls[2] = {delta[0] * ray->inv_disp[0], delta[1] * ray->inv_disp[1]};
  
  const unsigned int hit_i_first = incl_x && (t_walls[0] < t_exit);
  t_exit = hit_i_first ? t_walls[0] : t_exit;
  edge   = hit_i_first ? 0 : edge; 
    
  const unsigned int hit_j_first = incl_y && (t_walls[1] < t_exit);
  t_exit  = hit_j_first ? t_walls[1] : t_exit;
  edge = hit_j_first ? 1 : edge; 

#ifndef NDEBUG
/*  printf("Cursor w %d at index (%d,%d).%d t_exit %.4f exit_dir %d dir %.3f %.3f\n",(int)w,
	 (int)(cursor->t.i), (int)(cursor->t.v >> 1), (int)(cursor->t.v & 1),
	 t_exit, (int)edge, ray->dx, ray->dy); */
#endif

  next->time = t_exit;
  next->edge = edge;
}

void tri_quadtree_init_ray(const float* restrict dx, _raymarch_data* restrict ray)
{
  // Initialise the ray start point and direction, precomputing some cached values
  ray->disp[0] = dx[0];
  ray->disp[1] = dx[1];
  const float dy_m_dx = dx[1] - dx[0];
  ray->dy_m_dx = dy_m_dx;
  
  // Safe inverses, with signs consistent with d*_pos
  ray->inv_disp[0] = fabsf(dx[0]) > FLT_MIN ? 1.0f / dx[0] : copysignf(1e9f, dx[0]);
  ray->inv_disp[1] = fabsf(dx[1]) > FLT_MIN ? 1.0f / dx[1] : copysignf(1e9f, dx[1]);
  ray->inv_dy_m_dx = fabsf(dy_m_dx) > FLT_MIN ? 1.0f / dy_m_dx : copysignf(1e9f, dy_m_dx);    

  ray->disp_pos[0] = ray->inv_disp[0] > 0;
  ray->disp_pos[1] = ray->inv_disp[1] > 0;
  ray->dy_m_dx_pos = ray->inv_dy_m_dx > 0;
}

triangle_index get_triangle_index(const float x, const float y)
{
  const int32_t i = floorf(x);
  const int32_t v0 = (int32_t)floorf(2.0f * y) & (~1);
  const int32_t is_upper = (x - (float)i) < (y - 0.5f * (float)v0);
  const triangle_index t = {i, v0 | is_upper};
  return t;
}

void quadtree_ray_coroutine_start(quadtree_ray_coroutine_data* restrict qrc, 
				  const quadtree_cursor_s* restrict cursor, const float2 x0, const _raymarch_data* restrict ray)
{
  // Initialise the quadtree_ray_coroutine_data with the ray start point and
  // direction, precomputing some cached values. Set the cursor to the first
  // ancestor of this point and find the time to exit it
  qrc->cursor = *cursor;
  
  // Move the cursor to the common ancestor (set the fine index to (i_new, j_new))
  const triangle_index t = get_triangle_index(x0.v[0], x0.v[1]);
  qrc->x0 = x0;

  // TODO single structure
  qrc->last_time = 0.0f;
  qrc->last_edge = 3; // Started in current cell
  
  move_cursor_to_first_common_ancestor(&qrc->cursor, t);

  // Set time to exit current cell  
  _set_quadtree_ray_cell_exit(qrc->x0, &qrc->cursor, ray, &qrc->_next);
}

static inline int32_t clamp_node(const float x, const int32_t x_high, const int32_t low_bits)
{
  const int32_t xi = (int32_t)floorf(x);
  const int32_t x_max = x_high + low_bits;
  if (xi < x_high) return x_high;
  if (xi > x_max) return x_max;
  return xi;
}

int quadtree_ray_coroutine_next(const enum EQRC refine, const _raymarch_data* restrict ray, quadtree_ray_coroutine_data* restrict qrc)
{
  /*
    refine - whether to refine (=1) or continue(=0)
    qrc    - ray cursor data

    Returns
      1 - Successfully refined/continued
      0 - Failed to refine (leaf) or continue (end of ray)

   */
#ifndef NDEBUG
//  if (refine) printf("Refine\n");
//  else printf("Next %d\n", qrc->_next.edge);
#endif

  if (!refine)
  {
    // Continue to the next cell
    triangle_index t = qrc->cursor.t;
    
    const int32_t w = _cursor_width(&qrc->cursor);
    const int32_t v_drop_bits = (w<<1)-1;
    const int32_t i_drop_bits = v_drop_bits >> 1;
    t.i &= ~i_drop_bits; // Keep the top bits

    const int32_t v_quad_old = qrc->cursor.t.v & ~v_drop_bits;

    if (qrc->_next.time > qrc->last_time)
    {
      qrc->last_time = qrc->_next.time;
    }
    
    qrc->last_edge = qrc->_next.edge;
    
    // Which face was hit?
    switch (qrc->_next.edge)
    {
    case 0:
    {
      // i-first
      t.i += ray->disp_pos[0] ? w : -1;
      
      // Check for overflow in j due to numerical error
      const int32_t v_new = clamp_node(2.0f * (qrc->x0.v[1] + qrc->_next.time * ray->disp[1]),
			 v_quad_old, v_drop_bits);
      t.v = (v_new & (~(int32_t)1)) | (int32_t)ray->disp_pos[0];
      break;
    }
    case 1:
    {
      // j-first
      const int32_t v_new = v_quad_old + (ray->disp_pos[1] ? 2*w : -2);
			     
      t.i = clamp_node(qrc->x0.v[0] + qrc->_next.time * ray->disp[0], t.i, i_drop_bits);
      t.v = v_new | (!ray->disp_pos[1]);
      break;
    }
    case 2:
      // Crosses diagonal
      // NB always leaf, so i_old==i_old_high
      t.v = qrc->cursor.t.v ^ 1;
      break;
    default:
      return 0; // ray-end
    }

    // Move the cursor to the common ancestor (set the fine index to t)    
    move_cursor_to_first_common_ancestor(&qrc->cursor, t);
  }

  const int success = _move_cursor_to_child(&qrc->cursor);
  if (refine && !success) return 0; // At leaf level  
  assert((int)refine | success); // After Continue (last common ancestor) we should always be able to refine

  // Set time to exit current cell  
  _set_quadtree_ray_cell_exit(qrc->x0, &qrc->cursor, ray, &qrc->_next);

#ifndef NDEBUG
  ++(qrc->steps);
#endif

  return 1;
}

// When warm-starting a ray in a later quad, need to find exact entry triangle from axis crossing
static triangle_index _re_find_entry_triangle_along_side(const float2 x0, const triangle_index t0, const _raymarch_data* restrict ray, const unsigned side, const float hit)
{
  // Ray and beam started in this quad, nothing to change
  if (side==2) return t0;

  const unsigned ortho_side = 1-side;
  const int enter_pos_ortho = (int)floorf(x0.v[ortho_side] + (hit * ray->inv_disp[side] * ray->disp[ortho_side]));

  triangle_index t = t0;
  
  const int32_t u = (int32_t)(enter_pos_ortho<<ortho_side) |
    (int32_t)(ray->disp_pos[0] & ortho_side); // If the side==0 (changing j), then the lower/upper is set based on whether the displacement is positive.
  t.iv[ortho_side] = u; // side==0 => i unchanged, side==1 => v unchanged, even the lower/upper bit

  return t;
}

void quadtree_beam_warm_start_ray_coroutine(quadtree_ray_coroutine_data* restrict qrc, const _raymarch_data* ray, const tri_quadtree_beam_coroutine_s* restrict qbc)
{
  *qrc = qbc->qrc;
  qrc->last_time = 0.f;
  qrc->last_edge = 3;
#ifndef NDEBUG
  qrc->steps = 0;
#endif

  // Now the beams are just checking the distance fields (pyramids), the cursor may not even
  // be in the correct quad, so we should find the exact triangle now.
  qrc->cursor.t = _re_find_entry_triangle_along_side(qrc->x0, qrc->cursor.t, ray, qbc->entry.side, qbc->entry.hit);

#ifndef NDEBUG
  {
    // Check that the triangle is along the ray

    // Position of cell bottom-left relative to ray-start
    const float bl[2] = {(float)qrc->cursor.t.i - qrc->x0.v[0],
      (float)(qrc->cursor.t.v >> 1) - qrc->x0.v[1]};
    const int is_upper = qrc->cursor.t.v & 1;
    const intervalf_s extents[2] = {{bl[0], bl[0] + 1}, {bl[1], bl[1]+1}};
    // Dot product of triangle corners with perpendicular to ray displacement
    const float perp[2] = {-ray->disp[1], ray->disp[0]};
    const float perp2 = perp[0]*perp[0] + perp[1]*perp[1];    
    const float d00 = extents[0].min * perp[0] + extents[1].min * perp[1],
      d01 = extents[0].v[1-is_upper] * perp[0] + extents[1].v[is_upper] * perp[1],
      d11 = extents[0].max * perp[0] + extents[1].max * perp[1];
    // Very small values could be classed as either side of line, so dont assert
    const float crit = 1e-2f * 1e-2f * perp2;
    const int marginal = (d00*d00 < crit) | (d01*d01 < crit) | (d11*d11 < crit);
    const int res = (d00 > 0.f)<<2 | (d01 > 0.f)<<1 | (d11 > 0.f);
    if ((res==0 || res == 7) && !marginal) 
    {
      const float inv_perp = 1.0f / sqrtf(perp2);
      
      printf("Cursor does not lie along ray %.3f %.3f %.3f\n", d00, d01, d11);
      printf("Ray start (%.3f, %.3f)\n", qrc->x0.v[0], qrc->x0.v[1]);
      printf("Direction (%.3f, %.3f)\n", ray->disp[0], ray->disp[1]);
      printf("Distances %f %f %f\n", d00 * inv_perp, d01*inv_perp, d11 * inv_perp);
      print_quadtree_cursor(&qrc->cursor);      
      assert(0);
    }
  }
#endif
  
  _set_quadtree_ray_cell_exit(qrc->x0, &qrc->cursor, ray, &qrc->_next);  
}

quadtree_cursor_s init_cursor(const int width)
{
//  uint32_t lost_bits; // Number of ignored bits (1=which triangle, 2 = finest quad etc.)
//  uint32_t ignore_repeats; // Mask (width<<1)-1, used for ignoring the high-bits  
//  triangle_index t;

  unsigned int depth = 0;
  while (width >> depth > 0) ++depth;
  const uint32_t lost_bits = depth + 1; // Includes triangle level
  
  quadtree_cursor_s res = {lost_bits, (1u<<lost_bits)-1u, {0,0}};
  return res;
}

void tq_beam_init_ray(const _raymarch_data* restrict ray, tri_quadtree_beam_s* restrict beam)
{
  for (unsigned axis=0; axis<2; ++axis)
  {
    const float disp = ray->disp[axis];
    beam->disp[axis] = (intervalf_s){disp, disp};
    beam->d_pos[axis] = ray->disp_pos[axis];
    
    const float ratio = disp * ray->inv_disp[1-axis];
    beam->dxy_dyx[axis] = (intervalf_s){ratio, ratio};
  }
}

// Expand the beam to cover a new ray
void tq_beam_add_beam(const tri_quadtree_beam_s* restrict other_beam, tri_quadtree_beam_s* restrict beam)
{
  for (unsigned axis=0; axis<2; ++axis)
  {
    beam->disp[axis] = interval_union(beam->disp[axis], other_beam->disp[axis]);
    
    if (beam->d_pos[axis] != other_beam->d_pos[axis])
    {
      beam->d_pos[axis] = 2; // Degenerate
    }

    beam->dxy_dyx[axis] = interval_union(beam->dxy_dyx[axis], other_beam->dxy_dyx[axis]);
  }
}

static inline void _set_beam_cell_exit(const int ray_edge, const float2 x0, const _raymarch_data* restrict ray, const tri_quadtree_beam_s* restrict beam, const quadtree_cursor_s* restrict cursor, _beam_side_exit* restrict next)
{
//  printf("Setting beam cell exit for side %d\n", ray_edge);
  const int axis = ray_edge;
  // If the axis of the next transition would be the diagonal, we are done (beams don't go to final triangulation)
  if (axis > 1 || beam->d_pos[axis]==2)
  {
    next->side = 2; // Diagonal/ending
    next->hit = 0.f; // Ignore
    return;
  }

  // Here we check that we have not finished any rays
  const unsigned exit_right = ray->disp_pos[axis];

  // Position along the axis we move next
  const float next_pos = _cursor_quad_rel_x0_axis(cursor, x0, (unsigned)axis).v[exit_right];

  next->hit = next_pos;
  next->side = (uint32_t)axis;

  // Started overlap, some rays will finish before we reach here
  if ((next_pos < beam->disp[axis].v[1-exit_right]) ^ exit_right)
  {
//    printf("Some rays finish at %.3f which is hit before %.3f\n", beam->disp[axis].v[1-exit_right], next_pos);
    next->side = 2; // some rays finish in this cell
    return; // Possibly we could check if *all* rays finish
  }

  // What is our extent along the orthogonal axis?
  const intervalf_s ortho_disp = beam->dxy_dyx[1-axis];
  next->proj = interval_scale(ortho_disp, next_pos);
}

void quadtree_beam_coroutine_start(tri_quadtree_beam_coroutine_s* restrict qbc, const quadtree_cursor_s* restrict cursor, const float2 x0, const _raymarch_data* restrict ray, const tri_quadtree_beam_s* restrict beam)
{
  quadtree_ray_coroutine_start(&qbc->qrc, cursor, x0, ray);

  _set_beam_cell_exit(qbc->qrc._next.edge, qbc->qrc.x0, ray, beam, &qbc->qrc.cursor, &qbc->exit);

  qbc->entry = (_beam_side_exit){{0.f, 0.f}, 2 /* enter */, 0.f};
}

// Try to refine/continue the beam, failing if that would either split the beam (exit different edges), refine to triangles, or some rays finish
int quadtree_beam_coroutine_next(const enum EQRC is_refine, const _raymarch_data* restrict ray, const tri_quadtree_beam_s* restrict beam, tri_quadtree_beam_coroutine_s* restrict qbc)
{
  // Cannot refine from triangle, or continue (may hit diagonal wall and not pre-calculated beam projection for this)
  if (!qbc->qrc.cursor.lost_bits) return 0;

  if (!is_refine) // EContinue
  {
    // If the axis of the next transition would be the diagonal, we are done (beams don't go to final triangulation),
    // or some rays would finish
    if (qbc->exit.side > 1) return 0;
    
    // Store the ortho_beam_projection
    qbc->entry = qbc->exit;
    
    // Move the cursor in x or y
  }
  [[maybe_unused]] const int success = quadtree_ray_coroutine_next(is_refine, ray, &qbc->qrc);
  assert(success);

  _set_beam_cell_exit(qbc->qrc._next.edge, qbc->qrc.x0, ray, beam, &qbc->qrc.cursor, &qbc->exit);  
  return 1;
}

static inline void _set_sub_beam_cell_entry(const tri_quadtree_beam_s* restrict sub_beam, const _beam_side_exit* restrict beam_entry, _beam_side_exit* restrict sub_beam_entry)
{
  const intervalf_s proj = beam_entry->side < 2 ? interval_scale(sub_beam->dxy_dyx[1-beam_entry->side], beam_entry->hit) : beam_entry->proj;
  *sub_beam_entry = (_beam_side_exit){proj, beam_entry->side, beam_entry->hit};
}

void quadtree_beam_warm_start_fine_beam(const _raymarch_data* canonical_ray, const tri_quadtree_beam_s* restrict fine_beam, const tri_quadtree_beam_coroutine_s* restrict beam_co, tri_quadtree_beam_coroutine_s* restrict fine_beam_co)
{
  // Warm start the canonical ray to its first triangle in the same quad as the beam
  quadtree_beam_warm_start_ray_coroutine(&fine_beam_co->qrc, canonical_ray, beam_co);

  // Unlike the ray, we actually set the entry projection (in case the fine_beam needs to start a ray)
  _set_sub_beam_cell_entry(fine_beam, &beam_co->entry, &fine_beam_co->entry);

  if (beam_co->entry.side < 2)
  {
    const intervalf_s ortho_disp = fine_beam->dxy_dyx[1-beam_co->entry.side];
    fine_beam_co->entry.proj = interval_scale(ortho_disp, beam_co->entry.hit);
  }
  
  // Set the fine_beam exit
  _set_beam_cell_exit(fine_beam_co->qrc._next.edge, fine_beam_co->qrc.x0, canonical_ray, fine_beam, &fine_beam_co->qrc.cursor, &fine_beam_co->exit);
}
