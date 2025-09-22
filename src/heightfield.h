// Copyright 2025 Peter Edward Creasey
#pragma once
// Heightfield quadtree
#include <stdint.h>

// Forward declaration
struct heightfield_ray_coroutine;
typedef struct heightfield_ray_coroutine heightfield_ray_coroutine_s;
struct quadtree_cursor;
typedef struct quadtree_cursor quadtree_cursor_s;
struct heightfield_ray;
typedef struct heightfield_ray heightfield_ray_s;
struct heightfield_beam;
typedef struct heightfield_beam heightfield_beam_s;
struct heightfield_beam_coroutine;
typedef struct heightfield_beam_coroutine heightfield_beam_coroutine_s;

// Gradient w.r.t. indices i,j
typedef struct
{
  float dh_di;
  float dh_dj;
} heightfield_gradient;

typedef struct 
{
  uint32_t i,j; // Sizes in i (slowest rolling index) and j (fastest rolling)
} grid_size_s;

typedef struct
{
  uint32_t i,j;
} uint2;

// Shape of an aggregate bound, along with shift
typedef struct
{
  grid_size_s shape; // Shape of the rasterisation
  uint32_t power2; // Bounds are v[i,j] = max(v[i<<power2:1+(i+1)<<power2, j<<power2:1+(j+1)<<power2]) in the original image
} aggregate_bound;

typedef struct
{
  unsigned int num; // Number of levels of bounds (>=1 and <32)
  aggregate_bound bound[32];
  uint32_t level_starts[33];
  float* bounds;  
} aggregate_bounds_s;

// Maximum value for most de-refined bound
static inline float get_max_aggregate_bound(const aggregate_bounds_s* bound)
{
  return bound->bounds[bound->level_starts[bound->num] - 1];
}

typedef struct
{
  grid_size_s shape; // Size in i,j 
  uint32_t max_depth;   // max(shape.i, shape.j) <= 1<<max_depth
  unsigned int width; // 1<<max_depth

  // For the bounds
  unsigned int periodic; // 0= non-periodic, 1=periodic

  // Bounds for the maxima of the heights. These are used for individual rays
  aggregate_bounds_s max_height;

  // Bounds for the manhattan pyramid, b[i,j] >= height[i+p,j+q] - rho*(|p|+|q|) for-all p,q
  float manhattan_rho;
  aggregate_bounds_s manhattan_bound;  

} quadtree;

// Returns 0 on out-of-memory. If is_periodic is set, then N and M must be powers of 2
// If include_manhattan is set, we build the manhattan bounds
quadtree* init_tree(const int N, const int M, const int is_periodic, const int include_manhattan, const float* heights);

void free_tree(quadtree* tree);
