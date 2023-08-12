#pragma once

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
#define cextern extern "C"
#else
#define cextern extern
#endif

typedef struct VegaImplInstPair {
    void * impl;
    void * inst;
} VegaImplInstPair;

#define VEGAIMPLINST(_impl, _inst) ((VegaImplInstPair){.impl=_impl,.inst=_inst})

typedef char byte;
typedef unsigned char ubyte;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

/*
    A structure to store allocation functions to override the default ones.
*/
typedef struct VegaAllocFuncs {
    // void * malloc(size_t size);
    void * (*malloc_fn)(size_t);
    // void * calloc(size_t blocksize, size_t nblocks);
    void * (*calloc_fn)(size_t, size_t);
    // void free(void *);
    void (*free_fn)(void *);
} VegaAllocFuncs;

// Sets the allocation functions internally. You must call this before vegaInit().
cextern void vegaSetAllocFuncs(VegaAllocFuncs aFuncs);

cextern void * vegaAlloc(size_t size);
cextern void * vegaReAlloc(void * ptr, size_t size);
cextern void * vegaCalloc(size_t size, size_t nmemb);
cextern void vegaFree(void * ptr);

#define VEGA_VIDMODE_FULLSCREEN (1 << 0)

typedef enum VegaGraphicsResource {
    GFX_PROGRAM,
    GFX_MODEL,
    GFX_MATERIAL,
    GFX_TEXTURE,
} VegaGraphicsResource;

typedef enum VegaShaderType {
    SHADERTYPE_COMPUTE,
    SHADERTYPE_VERTEX,
    SHADERTYPE_FRAGMENT,
    SHADERTYPE_GEOMETRY,
    SHADERTYPE_COMPOSITE,
} VegaShaderType;

typedef struct VegaGraphicsBackend {
    size_t type_size;

    void * (*init)();
    void (*run)(void * self);
    void (*deinit)(void * self);

    uint (*get_window_width)(void * self);
    uint (*get_window_height)(void * self);
    uint (*get_window_mode)(void * self);

    void (*set_window_aspect)(void * self, uint asp_x, uint asp_y);
    void (*set_window_width)(void * self, uint width);
    void (*set_window_height)(void * self, uint width);
    void (*set_window_title)(void * self, const char * title);
    void (*set_window_mode)(void * self, uint mode);

    uint (*push_resource)(void * self, VegaGraphicsResource type, const VegaImplInstPair resource);
    void * (*pop_resource)(void * self);
    void * (*peek_resource)(void * self, VegaGraphicsResource type, uint id);
    VegaGraphicsResource (*peek_resource_type)(void * self, uint id);

    void (*program_init)(void * self, uint progid);
    void (*program_add_shader)(void * self, uint progid, const VegaImplInstPair shader);
    void (*program_link)(void * self, uint progid);
    void (*program_deinit)(void * self, uint progid);

    void (*model_init)(void * self, uint mdlid, const VegaImplInstPair model);
    void (*model_deinit)(void * self, uint mdlid);

    void (*material_init)(void * self, uint mtlid, const VegaImplInstPair material);
    void (*material_deinit)(void * self, uint mtlid);

    void (*texture_init)(void * self, uint texid, const VegaImplInstPair texture);
    void (*texture_deinit)(void * self, uint texid);
} VegaGraphicsBackend;

// Initializes Vega3D. Call this function FIRST before you do ANYTHING related to the game engine.
cextern int vegaInit(const VegaGraphicsBackend * gfxBackend);

// Shuts down Vega3D, freeing all objects allocated by Vega3D.
cextern int vegaDeinit();

// Runs the game engine, blocking until the window is closed.
cextern int vegaRun();

cextern VegaImplInstPair * vegaGetGraphicsBackend();

typedef struct VegaResource {
    size_t type_size;

    void * (*open)(void * arg);
    void (*close)(void * self);

    size_t (*read)(void * self, size_t size, size_t nmemb, void * buf);
    size_t (*write)(void * self, size_t size, size_t nmemb, void * buf);
    long (*seek)(void * self, long off, int whence);
} VegaResource;

typedef struct VegaScript {
    size_t type_size;

    void * (*load)(void * arg);
    void (*unload)(void * self);

    void (*init_fn)(void * self);
    void (*update_fn)(void * self);
} VegaScript;

typedef enum VegaDataTypes {
    DATATYPE_NULL,
    DATATYPE_POINTER,
    DATATYPE_UBYTE,
    DATATYPE_BYTE,
    DATATYPE_USHORT,
    DATATYPE_SHORT,
    DATATYPE_UINT,
    DATATYPE_INT,
    DATATYPE_FLOAT,
    DATATYPE_SINGLE = DATATYPE_FLOAT,
    DATATYPE_DOUBLE,
    DATATYPE_LONGDOUBLE,
} VegaDataTypes;

typedef enum VegaPolyTypes {
    POLYTYPE_POINTS,
    POLYTYPE_LINESTRIPS,
    POLYTYPE_LINELOOPS,
    POLYTYPE_LINES,
    POLYTYPE_TRISTRIPS,
    POLYTYPE_TRIFANS,
    POLYTYPE_TRIANGLES,
} VegaPolyTypes;

typedef enum VegaUniformTypes {
    UNIFTYPE_INT,
    UNIFTYPE_UINT,
    UNIFTYPE_FLOAT,
    UNIFTYPE_IVEC2,
    UNIFTYPE_UVEC2,
    UNIFTYPE_VEC2,
    UNIFTYPE_IVEC3,
    UNIFTYPE_UVEC3,
    UNIFTYPE_VEC3,
    UNIFTYPE_IVEC4,
    UNIFTYPE_UVEC4,
    UNIFTYPE_VEC4,
    UNIFTYPE_MAT2,
    UNIFTYPE_MAT3,
    UNIFTYPE_MAT4,
} VegaUniformTypes;

typedef enum VegaTextureTypes {
    TEXTYPE_TEX1D,
    TEXTYPE_TEX2D,
    TEXTYPE_TEX3D,
} VegaTextureTypes;

typedef enum VegaTextureFormats {
    TEXFORMAT_RED,
    TEXFORMAT_RG,
    TEXFORMAT_RGB,
    TEXFORMAT_BGR,
    TEXFORMAT_RGBA,
    TEXFORMAT_BGRA,
    TEXFORMAT_DEPTH,
    TEXFORMAT_DEPTHSTENCIL,
} VegaTextureFormats;

typedef struct VegaRenderStandard {
    size_t vattr_count;
    struct {
        int size;
        VegaDataTypes type;
    } * vattrs;
    size_t unif_count;
    struct {
        const char * name;
        VegaUniformTypes type;
    } * unifs;
    size_t tex_count;
    struct {
        VegaTextureTypes textype;
        VegaTextureFormats format;
        VegaDataTypes type;
    } * texs;
    VegaDataTypes mesh_data_type;
    VegaPolyTypes mesh_poly_type;
} VegaRenderStandard;

typedef struct VegaModel {
    size_t type_size;

    void * (*load)(void * arg);
    void (*unload)(void * self);

    void * (*vattr)(void * self, uint index);
    uint (*mesh_count)(void * self);
    uint (*mesh_poly_count)(void * self, uint index);
    void * (*mesh_polys)(void * self, uint index);
} VegaModel;

typedef struct VegaMaterial {
    size_t type_size;

    void * (*load)(void * arg);
    void (*unload)(void * self);

    void * (*unif)(void * self, uint index);
    uint (*tex)(void * self, uint index);
} VegaMaterial;

typedef struct VegaTexture {
    size_t type_size;

    void * (*load)(void * arg);
    void (*unload)(void * self);

    uint (*dimension)(void * self, uint dimindex);
    void * (*pixels)(void * self);
} VegaTexture;

typedef struct VegaShader {
    size_t type_size;

    void * (*load)(void * arg);
    void (*unload)(void * self);

    ubyte (*is_binary)(void * self);
    char * (*source)(void * self);
    void * (*binary)(void * self);
} VegaShader;

#include "vegaimpls.h"
