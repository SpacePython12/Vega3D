#include <lua5.4/lua.h>
#include <lua5.4/lualib.h>
#include <lua5.4/lauxlib.h>

#include "vega3d.h"
#include "vegaimpls.h"

int luaGetGfxInst(lua_State * L) {
    VegaImplInstPair * pair = lua_newuserdata(L, sizeof(VegaImplInstPair));
    VegaImplInstPair * gfx = vegaGetGraphicsBackend();
    luaL_getmetatable(L, "Vega3D.graphics");
    lua_setmetatable(L, -2);
    pair->impl = gfx->impl;
    pair->inst = gfx->inst;
    return 1;
}

VegaImplInstPair * luaGetSelf(lua_State * L, const char * name, const char * err) {
    void *ud = luaL_checkudata(L, 1, name);
    luaL_argcheck(L, ud != NULL, 1, err);
    return (VegaImplInstPair *)ud;
}

int luaGfxGetWindowWidth(lua_State * L) {
    VegaImplInstPair * self = luaGetSelf(L, "Vega3D.graphics", "'graphics' expected");
    lua_pushinteger(L, ((VegaGraphicsBackend *)self->impl)->get_window_width(self->inst));
    return 1;
}

int luaGfxGetWindowHeight(lua_State * L) {
    VegaImplInstPair * self = luaGetSelf(L, "Vega3D.graphics", "'graphics' expected");
    lua_pushinteger(L, ((VegaGraphicsBackend *)self->impl)->get_window_height(self->inst));
    return 1;
}

int luaGfxGetWindowMode(lua_State * L) {
    VegaImplInstPair * self = luaGetSelf(L, "Vega3D.graphics", "'graphics' expected");
    lua_pushinteger(L, ((VegaGraphicsBackend *)self->impl)->get_window_mode(self->inst));
    return 1;
}

int luaGfxSetWindowAspect(lua_State * L) {
    VegaImplInstPair * self = luaGetSelf(L, "Vega3D.graphics", "'graphics' expected");
    int asp_x = luaL_checkinteger(L, 2);
    int asp_y = luaL_checkinteger(L, 3);
    ((VegaGraphicsBackend *)self->impl)->set_window_aspect(self->inst, asp_x, asp_y);
    return 0;
}

int luaGfxSetWindowWidth(lua_State * L) {
    VegaImplInstPair * self = luaGetSelf(L, "Vega3D.graphics", "'graphics' expected");
    int width = luaL_checkinteger(L, 2);
    ((VegaGraphicsBackend *)self->impl)->set_window_width(self->inst, width);
    return 0;
}

int luaGfxSetWindowHeight(lua_State * L) {
    VegaImplInstPair * self = luaGetSelf(L, "Vega3D.graphics", "'graphics' expected");
    int height = luaL_checkinteger(L, 2);
    ((VegaGraphicsBackend *)self->impl)->set_window_height(self->inst, height);
    return 0;
}

int luaGfxSetWindowTitle(lua_State * L) {
    VegaImplInstPair * self = luaGetSelf(L, "Vega3D.graphics", "'graphics' expected");
    size_t l;
    const char * title = luaL_checklstring(L, 2, &l);
    ((VegaGraphicsBackend *)self->impl)->set_window_title(self->inst, title);
    return 0;
}

int luaGfxSetWindowMode(lua_State * L) {
    VegaImplInstPair * self = luaGetSelf(L, "Vega3D.graphics", "'graphics' expected");
    int mode = luaL_checkinteger(L, 2);
    ((VegaGraphicsBackend *)self->impl)->set_window_mode(self->inst, mode);
    return 0;
}

const luaL_Reg luaGraphics[] = {
    {"getGraphics", &luaGetGfxInst},

    {"getWindowWidth", &luaGfxGetWindowWidth},
    {"getWindowHeight", &luaGfxGetWindowHeight},
    {"getWindowMode", &luaGfxGetWindowMode},
    {"setWindowAspect", &luaGfxSetWindowAspect},
    {"setWindowWidth", &luaGfxSetWindowWidth},
    {"setWindowHeight", &luaGfxSetWindowHeight},
    {"setWindowTitle", &luaGfxSetWindowTitle},
    {"setWindowMode", &luaGfxSetWindowMode},

    {NULL, NULL}
};

typedef struct LuaScript {
    lua_State * lua;
} LuaScript;

void * LuaScript_load(void * user) {
    LuaScript * self = vegaAlloc(sizeof(LuaScript));
    self->lua = luaL_newstate();
    luaL_openlibs(self->lua);

    luaL_newmetatable(self->lua, "Vega3D.graphics");
    luaL_setfuncs(self->lua, luaGraphics, 0);
    lua_pushvalue(self->lua, -1);
    lua_setfield(self->lua, -2, "__index");

    VegaImplInstPair * pair = (VegaImplInstPair *)user;
    // ((VegaResource *)pair->impl)->read(pair->inst, )
    return self;
}


const VegaScript VegaExt_LuaScript = (VegaScript) {

};