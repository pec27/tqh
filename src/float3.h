// Copyright 2025 Peter Edward Creasey

// float3 (aligned)
#include <stdint.h> // uint32_t
#include "interval.h" // intervalf_s

typedef struct
{
  union
  {
    float abc[3];
    uint64_t _u64; // Force alignment
  };
} float3;

static inline float3 float_mul_float3(const float a, const float3 b)
{
  return (float3){a*b.abc[0], a*b.abc[1], a*b.abc[2]}; 
}

static inline float3 float3_mul(const float3 a, const float3 b)
{
  return (float3){a.abc[0]*b.abc[0], a.abc[1]*b.abc[1], a.abc[2]*b.abc[2]}; 
}

static inline float3 float3_add(const float3 a, const float3 b)
{
  return (float3){a.abc[0] + b.abc[0], a.abc[1] + b.abc[1], a.abc[2] + b.abc[2]}; 
}

static inline float float3_L2(const float3 a)
{
  return a.abc[0]*a.abc[0] + a.abc[1]*a.abc[1] + a.abc[2]*a.abc[2];   
}

static inline float float3_inner(const float3 a, const float3 b)
{
  return a.abc[0]*b.abc[0] + a.abc[1]*b.abc[1] + a.abc[2]*b.abc[2]; 
}

static inline float3 float3_fma(const float3 a, const float t, const float3 b)
{
  return (float3){a.abc[0] + t*b.abc[0], a.abc[1] + t*b.abc[1], a.abc[2] + t*b.abc[2]}; 
}

typedef struct
{
  union
  {
    struct {
      float x,y,z;
    };
    float xyz[3];
    float3 f3;
  };
} vec3f_s;

typedef struct 
{
  union
  {
    struct {
      float x,y,z;
    };
    float xyz[3];
    float3 f3;
    uint64_t _u64; // Force alignment
  };
} pseudovec3f_s;

// x0 + time * disp
static inline vec3f_s vec_fma(const vec3f_s x0, const float time, const vec3f_s disp)
{
  return (vec3f_s){x0.x + disp.x*time, x0.y + disp.y*time, x0.z + disp.z*time};
}

static inline vec3f_s vec_add(const vec3f_s a, const vec3f_s b)
{
  return (vec3f_s){a.x + b.x, a.y + b.y, a.z + b.z};
}
static inline vec3f_s vec_mul(const float a, const vec3f_s b)
{
  const float3 f = float_mul_float3(a, b.f3);
  return (vec3f_s){f.abc[0],f.abc[1],f.abc[2]};
}

