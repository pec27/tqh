// Copyright 2025 Peter Edward Creasey
#include "tri_quadtree.h" // Raymarch through triangulation
#include "heightfield.h"

// Cached data for ray-marching through the heightfield
struct heightfield_ray
{
  // Horizontal cpts tracked in tri_quadtree_ray
  _raymarch_data tq_march; // Ray marching through triangulated quadtree

  float dz; // Displacement in z
};

typedef struct heightfield_ray heightfield_ray_s;

static void heightfield_ray_init(const float* ray_disp, heightfield_ray_s* ray)
{
  ray->dz = ray_disp[2];
  tri_quadtree_init_ray(ray_disp, &ray->tq_march);
}

struct heightfield_ray_coroutine
{
  float2 z_cell; // z at cell entry,exit
  quadtree_ray_coroutine_data qrc;
  // Fixed over lifetime of ray
  float z0;
};

typedef struct heightfield_ray_coroutine heightfield_ray_coroutine_s;

//////////////////////////////////////////////////////////////////
// Beams

// Precomputed values when tracking a beam starting from a single point traversing the heightfield
struct heightfield_beam
{
  // Constant over lifetime
  tri_quadtree_beam_s tri_qtree_beam;
  float min_dz; // Minimum disp z of all rays in this beam
  // d(min(z))/dx and d(min(z))/dy, i.e. finding the lowest point at a cell exit.
  // These are only used iff the signs of the dx (or dy) are unique
  float2 d_min_z_d_xy; 
};

// Coroutine for a beam walk through the heightfield
struct heightfield_beam_coroutine
{
  tri_quadtree_beam_coroutine_s qbc;
  // Updated as we walk through cells
  float2 z_cell; // Lower bound for z at the entry and exit projection of the beam
  float z0; // value we started at

#ifndef NDEBUG
  uint32_t count; // Number of beam steps
#endif
};

typedef struct heightfield_beam heightfield_beam_s;
typedef struct heightfield_beam_coroutine heightfield_beam_coroutine_s;

// Initialize the beam data from a single ray data 
void heightfield_beam_init_ray(const heightfield_ray_s* ray, heightfield_beam_s* beam);

// Enlarge the beam to include another ray
void heightfield_beam_enlarge_to_ray(const heightfield_ray_s* ray, heightfield_beam_s* beam);

// Enlarge beam to include other beam
void heightfield_beam_enlarge_to_beam(const heightfield_beam_s* other, heightfield_beam_s* beam);

#ifndef NDEBUG
static inline void print_heightfield_beam(const struct heightfield_beam* beam)
{
  printf("min disp.z %.3f d_min(z))/dx %.3f d_min(z)/dy %.3f\n", beam->min_dz, beam->d_min_z_d_xy.v[0],beam->d_min_z_d_xy.v[1]);
  printf("Quadtree beam\n");
  print_tq_beam(&beam->tri_qtree_beam);
}
static inline void print_heightfield_beam_coroutine(const struct heightfield_beam_coroutine* beam_walk)
{
  printf("Heightfield beam coroutine at step %d, z0=%f z cell %f -> %f\n", beam_walk->count, beam_walk->z0, beam_walk->z_cell.v[0], beam_walk->z_cell.v[1]);
  print_tq_beam_coroutine(&beam_walk->qbc);
}
#endif

typedef struct
{
  heightfield_gradient grad;
  float time;
} heightfield_hit_s;

// Returns 1 if the heightfield is hit, and sets
//  ray_pos      - Outputs the position of the intersection
//  hit_fraction - Fraction f s.t. ray_x0 + f * ray_displacement hits the triangle
//  hit_grad     - Gradient of the triangle at which the hit occurs
int heightfield_intersect(const quadtree_cursor_s* cursor, float* ray_pos, const heightfield_ray_s* ray, heightfield_ray_coroutine_s* hrc, 
			  const quadtree* tree, const float* heights, heightfield_hit_s* hit);



void heightfield_beam_warm_start_ray_coroutine(const heightfield_beam_coroutine_s* beam_walk, const heightfield_ray_s* ray, heightfield_ray_coroutine_s* ray_walk);

void heightfield_beam_warm_start_fine_beam(const float* ray_x0, const heightfield_ray_s* canonical_ray, const heightfield_beam_s* fine_beam, const heightfield_beam_coroutine_s* beam_walk, heightfield_beam_coroutine_s* fine_beam_walk);

// Like heightfield_intersect but with an additional cursor
int heightfield_intersect_warm_start(float* ray_x0, const heightfield_ray_s* ray, heightfield_ray_coroutine_s* hrc, 
				     const quadtree* tree, const float* heights, heightfield_hit_s* hit);

// Step the beam through the heightfield bounds until the last unobstructed crossing of all rays is found.
// (The beam coroutine can then be used with a heightfield_intersect_warm_start)
void heightfield_trace_beam_to_last_unobstructed_crossing(const quadtree_cursor_s* cursor, const float* ray_x0, const heightfield_ray_s* ray, const heightfield_beam_s* beam, const quadtree* tree, heightfield_beam_coroutine_s* hbc);

// Like above but with hbc already initialised
void heightfield_trace_beam_warm_start(const float* ray_x0, const heightfield_ray_s* ray, const heightfield_beam_s* beam, const quadtree* tree, heightfield_beam_coroutine_s* hbc);
