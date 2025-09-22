#pragma once // Can be included from both 2d and 3d rays

// Floating point intervals
#include <math.h> // floorf, ceilf
#include <stdint.h> // uint64_t

// Maximum of two floats
static inline float max_2f(const float a, const float b) { return a > b ? a : b;}

// Minimum of two floats
static inline float min_2f(const float a, const float b) { return a < b ? a : b;}

typedef struct 
{
  union
  {
    float v[2];
    uint64_t _u64; // for alignment
  };
} float2;

// Minimum of a float2
static inline float float2_min(const float2 a)
{
  return min_2f(a.v[0], a.v[1]);
}

// a + t*b
static inline float2 float2_fma(const float2 a, const float t, const float2 b)
{
  return (float2){a.v[0] + t*b.v[0], a.v[1] + t*b.v[1]};
}
  
typedef struct 
{
  union
  {
    struct
    {
      float min, max; // Minimum and maximum values
    };
    float v[2];
    float2 f2;
  };
} intervalf_s;

// Does interval a contain b?
static inline int interval_contains(const intervalf_s a, const intervalf_s b)
{
  return a.min <= b.min && a.max >= b.max;
}

static inline intervalf_s interval_union(const intervalf_s a, const intervalf_s b)
{
  return (intervalf_s){a.min <= b.min ? a.min : b.min,
                       max_2f(a.max, b.max)};
}

static inline intervalf_s interval_add(const intervalf_s a, const float v)
{
  return (intervalf_s){a.min <= v ? a.min : v,
                       max_2f(a.max, v)};
}

static inline intervalf_s interval_scale(const intervalf_s a, const float scale)
{
  const float v[2] = {a.min * scale, a.max * scale};
  const int is_pos = scale >= 0.f;
  return (intervalf_s){v[1-is_pos], v[is_pos]};
}

// Furthest distance of any point in interval a from b
static inline float interval_max_outside(const intervalf_s a, const intervalf_s b)
{
  return max_2f(max_2f(b.min - a.min, a.max - b.max), 0.f);
}

#ifndef NDEBUG
#include <stdio.h> // printf
static inline void print_interval(const intervalf_s a)
{
  printf("[%.3f,%.3f]", a.min, a.max);
}
#endif //NDEBUG

// Right-open interval from [left, right)
typedef struct
{
  int32_t left, right;
} ROI;

static inline void roi_add(const int32_t pos, ROI* restrict roi)
{
  if (roi->left > pos) roi->left = pos;
  if (roi->right <= pos) roi->right = pos + 1;
}

static inline ROI get_roi_from_interval(const intervalf_s interval)
{
  return (ROI){(int32_t)floorf(interval.min), (int32_t)ceilf(interval.max)};
}
