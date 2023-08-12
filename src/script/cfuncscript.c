#include "vega3d.h"
#include "vegaimpls.h"

typedef struct CFuncScript {
    void (*init_fn)();
    void (*update_fn)();
} CFuncScript;

void * CFuncScript_load(void * user) {
    VegaExt_CFuncScriptArgs * arg = (VegaExt_CFuncScriptArgs *)user;
    CFuncScript * self = vegaAlloc(sizeof(CFuncScript));
    self->init_fn = arg->init_fn;
    self->update_fn = arg->update_fn;
    return self;
}

void CFuncScript_unload(void * self) {
    vegaFree(self);
}

void CFuncScript_init_fn(void * self) {
    ((CFuncScript *)self)->init_fn();
}

void CFuncScript_update_fn(void * self) {
    ((CFuncScript *)self)->update_fn();
}

const VegaScript VegaExt_CFuncScript = (VegaScript) {
    .type_size = sizeof(CFuncScript),
    .load = &CFuncScript_load,
    .unload = &CFuncScript_unload,
    .init_fn = &CFuncScript_init_fn,
    .update_fn = &CFuncScript_update_fn,
};