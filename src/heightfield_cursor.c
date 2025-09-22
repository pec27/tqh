// Copyright 2025 Peter Edward Creasey
#include "heightfield_cursor.h"
#include "tri_quadtree.h" // Raymarch through triangulation

#include <assert.h>

static inline float _quad_max(const float z00, const float z01, const float z10, const float z11)
{
  // Maximum over the quad
  return max_2f(max_2f(z00, z01), max_2f(z10,z11));
}

// If is_periodic not set, then should have pre-checked that we are in quadtree bounds
static inline float cursor_aggregate_bound(const unsigned is_periodic, const unsigned cursor_quad_shift, const uint2 pos, const aggregate_bounds_s* restrict quadtree_bounds)
{
  uint32_t level_idx = 0;
  const aggregate_bound* bound = &quadtree_bounds->bound[0];  
  while(bound->power2 < cursor_quad_shift
	&& (level_idx+1) < quadtree_bounds->num)
  {
    ++level_idx;
    ++bound;
  }

  // Fine index. Even if this is outside the heightfield bounds, we may still be in the quadtree
  // bounds in which case we still check the overlap.
  // This is important since it forces any ray-marches that enter the bounds from outside to refine
  // to get the correct entry point.

  const uint32_t quad_shift = bound->power2;
  const uint2 quad = {pos.i >> quad_shift, pos.j >> quad_shift};

  // Outside the heightfield, should have triggered earlier-out
#ifndef NDEBUG
  if (!(is_periodic || (quad.i < bound->shape.i && quad.j < bound->shape.j)))
  {
    printf("Attempting to access quad (%d,%d) in grid of size (%d,%d), quad shift %d cursor shift %d\n", quad.i, quad.j, bound->shape.i, bound->shape.j, quad_shift, cursor_quad_shift);
  }
//  printf(" Quad (%d-%d,%d-%d) bound[%d]=%.4f ", quad.i << quad_shift, ((quad.i+1)<<quad_shift)-1, quad.j<<quad_shift, ((quad.j+1)<<quad_shift)-1, quad_idx, bound);
  assert(is_periodic || (quad.i < bound->shape.i && quad.j < bound->shape.j));
#endif
  
  const uint32_t quad_idx = is_periodic ? ((quad.i & (bound->shape.i-1))*bound->shape.j + (quad.j & (bound->shape.j-1))) :
    quad.i * bound->shape.j + quad.j;
  
  assert(quadtree_bounds->level_starts[level_idx] + quad_idx < quadtree_bounds->level_starts[level_idx+1]);

#ifndef NDEBUG
//  printf(" Quad (%d-%d,%d-%d) bound[%d]=%.4f ", quad.i << quad_shift, ((quad.i+1)<<quad_shift)-1, quad.j<<quad_shift, ((quad.j+1)<<quad_shift)-1, quad_idx, bound);
#endif

  return quadtree_bounds->bounds[quadtree_bounds->level_starts[level_idx] + quad_idx];
}

int cursor_height_bound(const quadtree_cursor_s* cursor, const quadtree* tree, const float* restrict heights, float* z_max)
{
  const grid_size_s shape = tree->shape;

  // Ignore triangle
  const uint2 lattice_pos = {(uint32_t)cursor->t.i, (uint32_t)(cursor->t.v >> 1)};

  // At the leaf-level, we actually just check the containing quad for this
  const unsigned int cursor_quad_shift = cursor->lost_bits ? cursor->lost_bits - 1 : 0;

  // Check if outside heightfield
  if (!tree->periodic)
  {
    // Keep relevant bits for heightfield index
    const uint32_t upper_mask = ~((1u<<cursor_quad_shift) - 1);
    // Index of base pos in heightfield
    const uint32_t height_i = (uint32_t)lattice_pos.i & upper_mask,
      height_j = (uint32_t)lattice_pos.j & upper_mask;

    // Outside the heightfield.
    if (height_i+1 >= shape.i || height_j+1 >= shape.j) return 0;
  }

  // At leaf level the containing quad, and not pre-calculated
  if (heights && cursor_quad_shift==0 && tree->max_height.bound[0].power2 != 0)
  {
    const uint32_t row_left = lattice_pos.i * shape.j;
    
    if (!tree->periodic)
    {
      const unsigned int idx0 = row_left + lattice_pos.j, idx1 = idx0 + shape.j;
      if (cursor->lost_bits)  
      {
	*z_max =_quad_max(heights[idx0], heights[idx0+1], heights[idx1], heights[idx1+1]);
//      printf("Manual bound from (%d-%d,%d-%d)=%f\n", (int)lattice_pos.i, (int)lattice_pos.i+1, (int)lattice_pos.j, (int)lattice_pos.j+1, *z_max);	
      }
      else
      {
	// Single triangle
	const int is_upper = cursor->t.v & 1;
	*z_max = max_2f(max_2f(heights[idx0], heights[idx1+1]), heights[is_upper ? idx0+1 : idx1]);
      }
    }
    else
    {
      const uint32_t periodic_j_mask = shape.j - 1;
      const uint32_t shift_periodic_i = (shape.i - 1) * shape.j;
      
      const uint32_t idx_j0 = lattice_pos.j & periodic_j_mask;
      const uint32_t idx_i0 = row_left & shift_periodic_i;
      const uint32_t idx_j1 = (lattice_pos.j + 1) & periodic_j_mask;
      const uint32_t idx_i1 = (row_left + shape.j) & shift_periodic_i;	
      
      *z_max =_quad_max(heights[idx_i0 | idx_j0], heights[idx_i0 | idx_j1],
			heights[idx_i1 | idx_j0], heights[idx_i1 | idx_j1]);	
    }
  }
  else
  {
    *z_max = cursor_aggregate_bound(tree->periodic, cursor_quad_shift, lattice_pos, &tree->max_height);
  }
  return 1;
}

int cursor_overlaps_height(const quadtree_cursor_s* cursor, const quadtree* tree, const float* heights, const float2 z_node)
{
  // Outside heightfield
  float z_max;  
  if (!cursor_height_bound(cursor, tree, heights, &z_max)) return 0;

  const int overlap = (z_max > z_node.v[0]) | (z_max > z_node.v[1]);
  return overlap;
}

// Find the upper bound at the cursor quad, i.e. smallest b(quad) that satisfies b  <= height[i+p,j+q] - rho(|p|+|q|) for all i,j in quad.
// Outside the quad we return the maximum bound (could do more complicated things but probably not worth it)
float cursor_manhattan_bound(const quadtree_cursor_s* restrict cursor, const unsigned is_periodic, const grid_size_s shape, const aggregate_bounds_s* restrict manhattan_bounds)
{
  // Ignore triangle
  const uint2 lattice_pos = {(uint32_t)cursor->t.i, (uint32_t)(cursor->t.v >> 1)};

  // At the leaf-level, we actually just check the containing quad for this
  const unsigned int cursor_quad_shift = cursor->lost_bits ? cursor->lost_bits - 1 : 0;

  assert(!is_periodic); // Dont support periodic pyramids
  
  // Check if outside heightfield    
  {
    // Keep relevant bits for heightfield index
    const uint32_t upper_mask = ~((1u<<cursor_quad_shift) - 1);
    // Index of base pos in heightfield
    const uint2 height = {lattice_pos.i & upper_mask, lattice_pos.j & upper_mask};

    // Outside the heightfield.
    if (height.i+1 >= shape.i || height.j+1 >= shape.j) return get_max_aggregate_bound(manhattan_bounds);
  }

  return cursor_aggregate_bound(is_periodic, cursor_quad_shift, lattice_pos, manhattan_bounds);
}


