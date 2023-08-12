#pragma once

#include "vega3d.h"

typedef struct 

typedef struct VegaExt_FSResourceArgs {
    const char * path;
    const char * mode;
} VegaExt_FSResourceArgs;

#define VEGAEXT_FSRESOURCEARGS(_path, _mode) (&(VegaExt_FSResourceArgs){.path=_path, .mode=_mode})

extern const VegaResource VegaExt_FSResource;

typedef struct VegaExt_CFuncScriptArgs {
    void (*init_fn)();
    void (*update_fn)();
} VegaExt_CFuncScriptArgs;

#define VEGAEXT_CFUNCSCRIPTARGS(_init, _update) (&(VegaExt_CFuncScriptArgs){.init_fn=_init, .update_fn=_update});

extern const VegaScript VegaExt_CFuncScript;

#define VEGAEXT_LUASCRIPTARGS(_init, _update) (&(VegaExt_CFuncScriptArgs){.init_fn=_init, .update_fn=_update});

extern const VegaScript VegaExt_LuaScript;