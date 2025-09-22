// Copyright 2025 Peter Edward Creasey
#ifndef NDEBUG
  #include <stdio.h>
#endif
#include <assert.h>

#include "interval.h"

typedef struct
{
  union
  {
    // The index of a triangle in the quadtree-triangulation
    // Stores (i,j,upper_bit), where (i,j) refers to the quad
    // upper_bit indicates the top-left (rather than bottom-right)
    // triangle. v stores j in the top 31 bits, upper bit is the least-significant
    struct
    {
      int32_t i,v;
    };
    int32_t iv[2];
    uint64_t _u64; // alignment
  };
  
} triangle_index;

struct quadtree_cursor
{
  uint32_t lost_bits; // Number of ignored bits (1=which triangle, 2 = finest quad etc.)
  uint32_t ignore_repeats; // Mask (width<<1)-1, used for ignoring the high-bits  
  triangle_index t;
};

typedef struct quadtree_cursor quadtree_cursor_s;

quadtree_cursor_s init_cursor(const int width);

static inline int cursor_is_root(const quadtree_cursor_s* cursor) {return (cursor->ignore_repeats >> cursor->lost_bits)==0;}

// Current state of the quadtree_ray_coroutine. You shouldn't need
// to fill this yourself, just call quadtree_ray_coroutine_start
typedef struct
{
  float disp[2]; // Direction
  float inv_disp[2]; // 1/dx, 1/dy  
  float dy_m_dx; // dy - dx (used for testing crossing the diagonal)
  float inv_dy_m_dx; // 1/(dy-dx)
  unsigned int disp_pos[2]; // Sign of dx,dy
  unsigned int dy_m_dx_pos; 
} _raymarch_data;

typedef struct 
{
  float time; // Exit time of current cell
  int edge; // 0=changes i, 1=changes j, 2=crosses diagonal, 3=finishes in current cell (t=1), 
} _cell_exit;

// Coroutine data for a ray through the quadtree
struct quadtree_ray_coroutine
{
  quadtree_cursor_s cursor;
  
  float last_time;
  int last_edge; // 0=crossed i, 1=crossed j, 2=crossed diagonal, 3=started in current cell
  _cell_exit _next;
  
  // Internal data, constant over the lifetime of the ray
  float2 x0; // Ray start

#ifndef NDEBUG
  uint32_t steps; // Number of steps taken
#endif
};

typedef struct quadtree_ray_coroutine quadtree_ray_coroutine_data;

triangle_index get_triangle_index(const float x, const float y);

enum EQRC {EContinue=0,ERefine=1};

void tri_quadtree_init_ray(const float* dx, _raymarch_data* ray);

// Initialises the quadtree ray coroutine - qrc->cursor set to last common ancestor with cursor
void quadtree_ray_coroutine_start(quadtree_ray_coroutine_data* qrc, const quadtree_cursor_s* cursor, 
				  const float2 x0, const _raymarch_data* ray);

int quadtree_ray_coroutine_next(const enum EQRC next, const _raymarch_data* ray, quadtree_ray_coroutine_data* qrc_);

// A "Beam" version of the regular cursor, for a beam of rays that start at the same point and have
// similar directions (TODO later consider not starting at same point). We can then find the last
// common ancestor of cursor iterations for the constituent rays

typedef struct 
{
  // Minimum and maximum change in x per unit change in y, i.e. if we have crossed into
  // a node at y=y0+delta_y, the range of x positions will be x0 + delta_y * [dx_dy.min, dx_dy.max]
  intervalf_s dxy_dyx[2]; // dx/dy and dy/dx ranges
  // Bounds for displacements along the two axis
  intervalf_s disp[2]; 

  // Used to check degeneracy
  unsigned d_pos[2]; // Whether displacement x,y are positive for every displacement (1), negative for every displacement (0), or some positive, some negative (2)
} tri_quadtree_beam_s;


typedef struct
{
  intervalf_s proj; // The projection of the beam along the axis orthogonal to the last entry. E.g. if the side is at x=5, this is the y extent of the beam displacement projected onto x=5
  uint32_t side; // =0 (i), 1 (j) or 2 (enter, some exit, or degenerate sides)
  float hit; // The displacement along the crossing axis (qrc->_next.edge) where we enter the next cell (undefined for finish)
} _beam_side_exit;

typedef struct
{
  quadtree_ray_coroutine_data qrc;
  
  // Cursor stuff
  // Projections for cell entry and exit
  _beam_side_exit entry, exit;

} tri_quadtree_beam_coroutine_s;

// Initialize the beam from a single ray
void tq_beam_init_ray(const _raymarch_data* ray, tri_quadtree_beam_s* beam);
// Expand the beam to include other_beam
void tq_beam_add_beam(const tri_quadtree_beam_s* other_beam, tri_quadtree_beam_s* beam);

// Initialise quadtree beam coroutine
void quadtree_beam_coroutine_start(tri_quadtree_beam_coroutine_s* qbc, const quadtree_cursor_s* cursor, const float2 x0, const _raymarch_data* ray, const tri_quadtree_beam_s* beam);

// Can the beam refine/continue coherently, i.e. would all constituent rays would be in the same node?
// On success the qrc is updated, the cursor is moved/refined. On failure the cursor is left in
// the old position, i.e. the last common ancestor of all rays in the beam.
int quadtree_beam_coroutine_next(const enum EQRC is_refine, const _raymarch_data* ray, const tri_quadtree_beam_s* beam, tri_quadtree_beam_coroutine_s* qbc);

// Initialise the quadtree_ray_coroutine with a beam, whose cursor points to a quad along the ray
// _raymarch_data precomputed. Note the qrc->last_time will be set to 0.f, and qrc->last_edge to 3, even
// though we did not enter in this cell
void quadtree_beam_warm_start_ray_coroutine(quadtree_ray_coroutine_data* qrc, const _raymarch_data* ray, const tri_quadtree_beam_coroutine_s* beam);

// Warm start fine_beam from a containing beam
void quadtree_beam_warm_start_fine_beam(const _raymarch_data* canonical_ray, const tri_quadtree_beam_s* fine_beam, const tri_quadtree_beam_coroutine_s* beam_co, tri_quadtree_beam_coroutine_s* fine_beam_co);

// Maximum distance of the beam projection outside the current cursor quad
float2 beam_max_horiz_dist_outside_quad(const quadtree_cursor_s* cursor, const float2 x0, const _beam_side_exit* entry, const _beam_side_exit* beam_exit);

#ifndef NDEBUG
static inline void print_triangle(triangle_index t)
{
  printf("triangle (%d,%d %s)\n", (int)t.i, (int)(t.v>>1), (t.v & 1) ? "upper" : "lower");  
}
static inline void print_tq_beam(const tri_quadtree_beam_s* beam)
{
  printf("Displacement in ");
  print_interval(beam->disp[0]);
  printf(" (positivity %d)\n", beam->d_pos[0]);  
  print_interval(beam->disp[1]);
  printf(" (positivity %d)\ndx/dy in ", beam->d_pos[1]);
  print_interval(beam->dxy_dyx[0]);  
  printf("\ndy/dx in ");
  print_interval(beam->dxy_dyx[1]);
  printf("\n");
}

void print_quadtree_cursor(const quadtree_cursor_s* cursor);

void print_tq_ray_coroutine(const quadtree_ray_coroutine_data* qrc);

void print_tq_beam_coroutine(const tri_quadtree_beam_coroutine_s* beam);
#endif

