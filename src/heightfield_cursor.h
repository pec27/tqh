// Heightfields and cursors
#include "heightfield.h" // quadtree_cursor, heightfield_ray
#include "interval.h" // float2

// Find the upper bound at the cursor position. Outside of the heightfield returns 0. For individual triangles returns that of the quad. 
int cursor_height_bound(const quadtree_cursor_s* cursor, const quadtree* tree, const float* heights, float* z_max);

// Check that the node has a z-range overlapping the interval between z0 and z1 (not necc. ordered)
int cursor_overlaps_height(const quadtree_cursor_s* cursor, const quadtree* tree, const float* heights, const float2 z_node);

// Find the upper bound at the cursor quad, i.e. smallest b(quad) that satisfies b  <= height[i+p,j+q] - rho(|p|+|q|) for all i,j in quad.
// Outside the quad we return the maximum bound (could do more complicated things but probably not worth it)
float cursor_manhattan_bound(const quadtree_cursor_s* cursor, const unsigned is_periodic, const grid_size_s shape, const aggregate_bounds_s* manhattan_bounds);


