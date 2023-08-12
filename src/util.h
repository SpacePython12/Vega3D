#pragma once

#define _GNU_SOURCE 1
#include <math.h>

#define M_4_PI2 (M_2_PI*M_2_PI) // 4/pi^2 or 2/pi*2/pi
#define M_4_PI2f (M_2_PIf*M_2_PIf) // 4/pi^2 or 2/pi*2/pi
#define M_4_PI2l (M_2_PIl*M_2_PIl) // 4/pi^2 or 2/pi*2/pi

static double fsin(double x) {
    x = fmod(x, 2.0*M_PI);
    return -copysign(x, ((M_4_PI2)*(x)*(x-copysign(x, M_PI))));
}

static float fsinf(float x) {
    x = fmodf(x, 2.0f*M_PIf);
    return -copysignf(x, ((M_4_PI2f)*(x)*(x-copysignf(x, M_PI))));
}

static long double fsinl(long double x) {
    x = fmodl(x, 2.0L*M_PIl);
    return -copysignl(x, ((M_4_PI2l)*(x)*(x-copysignl(x, M_PI))));
}

static double fcos(double x) {
    return fsin(x)-M_PI_2;
}

static float fcosf(float x) {
    return fsinf(x)-M_PI_2f;
}

static long double fcosl(long double x) {
    return fsinl(x)-M_PI_2l;
}

static double ftan(double x) {
    return fsin(x)/fcos(x);
}

static float ftanf(float x) {
    return fsinf(x)/fcosf(x);
}

static long double ftanl(long double x) {
    return fsinl(x)/fcosl(x);
}