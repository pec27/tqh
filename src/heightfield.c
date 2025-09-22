#include "interval.h"
#include "heightfield.h"

#include <assert.h>
#include <stdlib.h> // malloc, calloc, free

static inline uint32_t num_grid_elements(const grid_size_s shape) { return shape.i*shape.j;}

#ifndef NDEBUG
static inline intervalf_s _get_minmax(const grid_size_s shape, const float* restrict arr)
{
  intervalf_s bound = (intervalf_s){1e12f, -1e12f}; // TODO better bounds
  
  for (size_t i=0, i_end = num_grid_elements(shape); i<i_end; ++i)
  {
    if (arr[i] > bound.max) bound.max = arr[i];
    if (arr[i] < bound.min) bound.min = arr[i];      
  }
  return bound;
}
#endif

// Returns a non-zero value if the non-zero argument is not a power of 2
static inline uint32_t is_not_power2(const uint32_t nonzero)
{
  return (nonzero & (nonzero - 1));
}

// Find Manhattan cone bounds for every point in the heightfield, that is to say
// at each (i,j), find b[i,j] s.t.
//   h[p,q] <= b[i,j] + rho * (|p-i| + |q-j|) \forall p,q .
// That is to say at (i,j) there is an empty inverted square-based pyramid rising
// from b[i,j] that is empty of geometry.
// Both heights and pyramid should be shape (N*M) arrays
void fill_manhattan_pyramid(const grid_size_s shape, const float* const restrict heights, const float rho, float* restrict pyramid)
{
  // TODO support is_periodic
  
  assert(rho > 0); // Should be strictly positive
  const uint32_t size = num_grid_elements(shape);
    
  // Forward sweep, i.e. make sure the bound in pyramid[i,j] satisfies heights at previous indices (idx=i*shape.j + j)
  {
    pyramid[0] = heights[0];
    // First row (no prev. row)
    uint32_t ij = 1; 
    for (; ij < shape.j; ++ij)
    {
      pyramid[ij] = max_2f(heights[ij], pyramid[ij-1] - rho);
    }
    // All later rows 
    while (ij < size)
    {
      // First point in row
      pyramid[ij] = max_2f(heights[ij], pyramid[ij - shape.j] - rho);
      const uint32_t ij_end = ij + shape.j;
      while ((++ij) < ij_end)
      {
	pyramid[ij] = max_2f(heights[ij], max_2f(pyramid[ij - shape.j], pyramid[ij -1]) - rho);
      }
    }
  }

  // Backward sweep, i.e. make sure bound consistent with later indices
  {
    uint32_t ij = size - 1;
    // Last row (no next row)
    for (const uint32_t row_start = size-shape.j; (ij--) > row_start;)
    {
      pyramid[ij] = max_2f(pyramid[ij], pyramid[ij + 1] - rho);
    }
    
    // Prev rows
    while (ij < size)
    {
      pyramid[ij] = max_2f(pyramid[ij], pyramid[ij + shape.j] - rho); // Last point in row
      
      // Fill row
      for (const uint32_t prev_row_last = ij - shape.j; (--ij) != prev_row_last;)
      {
	pyramid[ij] = max_2f(pyramid[ij], max_2f(pyramid[ij+1], pyramid[ij + shape.j]) - rho);      
      }
    };
  }
  // Done
}
static void fill_periodic_bounds_from_heights(float* restrict bounds, const float* restrict heights,
					      const grid_size_s heights_shape, const unsigned int step_size)
{
  // Fill in the bounds from a periodic heightfield, s.t.
  //   bounds[i,j] = max(heights[u,v]) 
  // with
  //   u in [i*s % N, i*s+1 % N, i*s+2 % N, ... % N, (i*s + s) % N]
  //   v in [j*s % M, j*s+1 % M, j*s+2 % M, ... % M, (j*s + s) & M]
  // where N,M = heights_shape which must be powers of 2


#ifndef NDEBUG
  printf("Filling bounds from [0,%d)x[0,%d)\n", heights_shape.i, heights_shape.j);    
#endif
  assert(!is_not_power2(step_size * num_grid_elements(heights_shape)));
  
  const uint32_t j_mask = heights_shape.j - 1;
  const uint32_t i_mask = (heights_shape.i-1)*heights_shape.j;
  const uint32_t num_heights = num_grid_elements(heights_shape);

  for (uint32_t idx_i0=0; idx_i0 < num_heights;
       idx_i0 += step_size * heights_shape.j)
  {
    const uint32_t idx_i_end = idx_i0 + (step_size + 1) * heights_shape.j;
    
    for (uint32_t j0=0; j0 < heights_shape.j; j0 += step_size)
    {
      assert(j0 + idx_i0 < num_grid_elements(heights_shape)+step_size); // Can touch (periodic repeats)
      float bound = heights[j0 + idx_i0]; // z00

      // May be beyond the final element, 
      const uint32_t j_end = j0 + step_size + 1;

      for (uint32_t j = j0+1; j < j_end; ++j)
      {
	bound = max_2f(bound, heights[j&j_mask | idx_i0]); // z[0,j]
      }
      for (uint32_t idx_i = idx_i0 + heights_shape.j;
	   idx_i != idx_i_end;
	   idx_i += heights_shape.j) // next row
      {
	// Do the periodic wrapping. This cannot be done earlier in case the (step_size+1)
	// is larger than the original dimension in u
	const uint32_t idx = idx_i & i_mask; 
	for (uint32_t j = j0; j < j_end; ++j)
	{
	  assert((j&j_mask | idx) < num_grid_elements(heights_shape));
	  bound = max_2f(bound, heights[j&j_mask | idx]); // z[i,j]	  
	}
      }
      *bounds++ = bound;
    }
  }
}

static void fill_bounds_level(float* restrict bounds, const grid_size_s size, const float* restrict child_bounds,
			      const grid_size_s child_size, const unsigned int step_size, const unsigned int block_size)
{
  // Fill in the bounds from that of the children
  // step_size  - Step for walking through the children
  // block_size - Bound for a block_size*block_size quadrant of children, i.e.
  //   bounds[i,j] = max(child_bounds[step_size*i:step_size*i + block_size, step_size*j:step_size*j + block_size])

#ifndef NDEBUG
  {
    const uint32_t large_threshold = 1u<<20;
    const int size_large = num_grid_elements(size) > large_threshold;
    const int child_large = num_grid_elements(child_size) > large_threshold;
    if (size_large && child_large)
    {
      printf("Filling bounds [0,%d)x[0,%d) from [0,%d)x[0,%d)\n", size.i, size.j, child_size.i, child_size.j);
    }
    else if (child_large)
    {
      printf("Filling bounds %dx%d -> %dx%d", child_size.i, child_size.j, size.i, size.j);
    }
    else 
    {
      printf(" -> %dx%d%s", size.i, size.j, (size.i*size.j==1) ? "\n" :  "");
    }
  }    
#endif
  assert(step_size*(size.i-1) + block_size >= child_size.i);
  assert(step_size*(size.j-1) + block_size >= child_size.j);
  
  for (uint32_t i=0, idx=0; i<size.i; ++i)
  {
    const uint32_t ci0 = step_size*i;
    const uint32_t ci0_0 = ci0 * child_size.j;

    const uint32_t cij_end = (ci0 + block_size < child_size.i ? ci0 + block_size : child_size.i) * child_size.j;
    
    for (uint32_t j=0;j<size.j; ++j, ++idx)
    {
      const uint32_t cj0 = j*step_size;
      assert(cj0 + ci0_0 < child_size.i*child_size.j);
      float bound = child_bounds[cj0 + ci0_0]; // z00

      const uint32_t cj_end = cj0 + block_size > child_size.j ? child_size.j : cj0 + block_size;

      for (uint32_t cj = cj0+1; cj < cj_end; ++cj)
      {
	bound = max_2f(child_bounds[cj + ci0_0], bound); // z[0,j]
      }
      for (uint32_t ci = ci0_0 + child_size.j; ci < cij_end; ci += child_size.j)
      {
	for (uint32_t cj = cj0; cj < cj_end; ++cj)
	{
	  assert(cj+ci < num_grid_elements(child_size));
	  bound = max_2f(child_bounds[cj + ci], bound); // z[i,j]	  
	}
      }
      bounds[idx] = bound;
    }
  }
}

// Find smallest x s.t. x<<power2 >= val
static inline uint32_t _get_downscale(const uint32_t val, const uint32_t power2)
{
  const uint32_t val_shift = val >> power2;
  const uint32_t remainder = (val & ((1u<<power2)-1)) ? 1 : 0;
  return val_shift + remainder;
}

// Find a downsampled shape s.t. shape_i<<power2 >= parent_i for i in {i,j}
static inline grid_size_s _get_downsample(const grid_size_s parent_shape, const uint32_t power2)
{
  return (grid_size_s){_get_downscale(parent_shape.i, power2), _get_downscale(parent_shape.j, power2)};
}

// Returns the number of levels. Sets max_height->level_starts[0... max_height->num]
static uint32_t _fill_downsample_sizes(const grid_size_s first_shape, const uint32_t first_power2, aggregate_bounds_s* levels)
{
  levels->level_starts[0] = 0;
  uint32_t num_sizes = 1;

  {
    aggregate_bound* child = levels->bound;

    child->power2 = first_power2;

    child->shape = first_shape;
    levels->level_starts[1] = num_grid_elements(child->shape);
    
    for (;num_grid_elements(child->shape) > 1; ++num_sizes)
    {
      // Values for parent
      aggregate_bound* parent = child+1;
      const uint32_t shift_to_parent = 1;
      parent->power2 = child->power2 + shift_to_parent;
      parent->shape = _get_downsample(child->shape, shift_to_parent);
      
      levels->level_starts[num_sizes+1] = levels->level_starts[num_sizes] + num_grid_elements(parent->shape);
      // Move to parent      
      child = parent;
    }
  }
  levels->num = num_sizes;

#ifndef NDEBUG
  printf("Allocating %d bounds levels with %d elements (takes %.2f MB)\n",
	 levels->num, levels->level_starts[levels->num], (float)((levels->level_starts[levels->num] * sizeof(float))/(1024.0f * 1024.f)));
#endif

  levels->bounds = malloc(levels->level_starts[num_sizes] * sizeof(float));

  // Out-of-mem
  if (!levels->bounds) return 0;
  
  return num_sizes;
}

// Fill downscaled values after 0th level done
static void _fill_subsequent_downscaling(aggregate_bounds_s* max_height)
{

  aggregate_bound* parent = &max_height->bound[0];
  
  for (uint32_t i=1;i<max_height->num;++i)
  {
    aggregate_bound* child = parent++;
    // Fill parent bounds from child
    const uint32_t downscale = 1 << (parent->power2 - child->power2);
#ifndef NDEBUG
    if (i==1)
    {
      printf("Downscaling with %dx%d blocks\n", downscale, downscale);
    }
#endif
    
    fill_bounds_level(max_height->bounds + max_height->level_starts[i], parent->shape,
		      max_height->bounds + max_height->level_starts[i-1], child->shape, downscale, downscale);
#ifndef NDEBUG
    if (i==1)
    {
      const intervalf_s minmax = _get_minmax(parent->shape, max_height->bounds + max_height->level_starts[i]);
      printf("  Min %.6f Maximum %.6f\n", minmax.min, minmax.max);
    }
#endif
  }
}
static int _init_max_heights(const int is_periodic, const grid_size_s shape, const float* restrict heights, aggregate_bounds_s* restrict max_height)
{
  // Non-periodic (N,M) heights are converted to (N-1,M-1) quads, with periodic (N,M)
  const grid_size_s height_quads_shape = is_periodic ? shape : (grid_size_s){shape.i-1, shape.j-1};
  const uint32_t first_power2 = 1; // Skip the zero-th level to avoid allocating as much as the heightfield itself
  // Out-of-mem
  if (!_fill_downsample_sizes(_get_downsample(height_quads_shape, first_power2), first_power2, max_height))
  {
    return 0;
  }

#ifndef NDEBUG
  printf("  (cf heightfield is %.2f MB)\n", (float)(num_grid_elements(shape)*(int)sizeof(float))/(1024.f * 1024.f));
#endif
  
  // Fill in current bounds from heights
  {
    aggregate_bound* child = max_height->bound;    
    const uint32_t downscale = 1 << child->power2;
    if (is_periodic)
    {
      fill_periodic_bounds_from_heights(max_height->bounds, heights, shape, downscale);
    }
    else
    {
      fill_bounds_level(max_height->bounds + max_height->level_starts[0], child->shape,
			heights, shape, downscale, downscale + 1);
    }
  }
  // Fill in all the subsequent bounds
  _fill_subsequent_downscaling(max_height);
  return 1;
}

static int _init_manhattan_bounds(const grid_size_s shape, const float* const restrict heights, const float rho, aggregate_bounds_s* restrict bound)
{
  // First find manhattan over same points as grid
#ifndef NDEBUG
  printf("Allocating temporary %dx%d array for pointwise Manhattan bounds (%.f MB)\n", shape.i, shape.j, (float)(num_grid_elements(shape)*sizeof(float))/(1024.f * 1024.f));
#endif

  float* pointwise = (float*)malloc(num_grid_elements(shape)*sizeof(float));
  if (!pointwise) return 0;

  fill_manhattan_pyramid(shape, heights, rho, pointwise);

  const grid_size_s height_quads_shape = {shape.i-1, shape.j-1};
  const uint32_t first_power2 = 1; // Skip the zero-th level to avoid allocating as much as the heightfield itself

#ifndef NDEBUG
  printf("Allocating Manhattan bounds\n");
#endif
  
  // Out-of-mem
  if (!_fill_downsample_sizes(_get_downsample(height_quads_shape, first_power2), first_power2, bound))
  {
    free(pointwise);
    return 0;
  }
  
  // Fill in current bounds from heights
  {
    const uint32_t downscale = 1 << first_power2;
    fill_bounds_level(bound->bounds + bound->level_starts[0], bound->bound->shape,
		      pointwise, shape, downscale, downscale + 1);
  }
  
  _fill_subsequent_downscaling(bound); // Fill in all the subsequent bounds
  
  free(pointwise); // Release temporary pointwise bounds
  return 1;
}

// Returns 0 on out-of-memory, or is_periodic set but N,M not powers of 2.
quadtree* init_tree(const int N, const int M, const int is_periodic, const int include_manhattan, const float* restrict heights)
{
  const grid_size_s shape = (grid_size_s){(uint32_t)N,(uint32_t)M};
  
  // Must have at least one element and if periodic, those should be a power of 2
  {
    const uint32_t num_heights = num_grid_elements(shape);
    if (num_heights < 1 || (is_periodic && is_not_power2(num_heights))) return 0;
  }
  
  quadtree* tree = (quadtree*)malloc(sizeof(quadtree));
  if (!tree) return 0;
  
  tree->shape = shape;
  tree->periodic = is_periodic ? 1 : 0;
  
  uint32_t height_size_shift = 0;
  for (uint32_t size=1; shape.i>=size || shape.j>=size; ++height_size_shift)
    size <<= 1;
  tree->max_depth = height_size_shift;
  tree->width = 1<<tree->max_depth;

  // Out-of-mem
  if (!_init_max_heights(is_periodic, shape, heights, &tree->max_height))
  {
    free(tree);
    return 0;
  }

  if (include_manhattan && !is_periodic)
  {
    tree->manhattan_rho = 0.1f; // TODO make a parameter (probably to be based on xy-scaling, but perhaps surface gradients)
    if (!_init_manhattan_bounds(shape, heights, tree->manhattan_rho, &tree->manhattan_bound))
    {
      free(tree->max_height.bounds);
      free(tree);
      return 0;
    }
  }
  else
  {
    tree->manhattan_bound.bounds = 0;
  }
  return tree;
}

void free_tree(quadtree* tree)
{
  if (tree->manhattan_bound.bounds)
  {
    free(tree->manhattan_bound.bounds);
  }
  free(tree->max_height.bounds);
  tree->max_height.bounds = 0;
}
