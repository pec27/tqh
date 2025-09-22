# -fPIC process independent code (for shared libraries)
# -std=c99 complex numbers

# Find the cpython suffix for the library, e.g. ".cpython-39-darwin.so"
CPYTHON_SUFFIX := $(shell python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

LIB_PREFIX = build/libtqh
LIB = $(LIB_PREFIX)$(CPYTHON_SUFFIX)
# -Weverything -Wsign-conversion -Wcovered-switch-default -Wshorten-64-to-32
CC = cc -Wextra -Wunused-variable -Wunused-function -Wsign-conversion -Wunused-but-set-variable -Wpedantic -Wswitch-enum  -Wshadow -Wimplicit-fallthrough -Werror -fPIC -shared -std=c2x -O3

HDR = src/float3.h src/tri_quadtree.h src/heightfield.h src/interval.h src/heightfield_cursor.h src/heightfield_ray.h

# Absolute sources (not auto-generated)
SRC_ABS = src/tri_quadtree.c src/heightfield.c src/lidar.c src/heightfield_cursor.c src/heightfield_ray.c
LIB_DEPENDS = ${SRC_ABS} ${HDR} Makefile 

LIB = $(LIB_PREFIX)$(CPYTHON_SUFFIX)
LIB_DEBUG = $(LIB_PREFIX)_debug$(CPYTHON_SUFFIX)

all: build ${LIB} debug


build:
	mkdir -p build

debug: ${LIB_DEBUG}

${LIB}: ${LIB_DEPENDS}
	${CC} -DNDEBUG -o ${LIB} ${SRC_ABS}

${LIB_DEBUG}: ${LIB_DEPENDS}
	${CC} -o ${LIB_DEBUG} ${SRC_ABS}

clean:
	rm -rf build



