#include <stdio.h>

#include "vega3d.h"
#include "vegaimpls.h"

typedef struct FSResource {
    FILE * file;
} FSResource;

void * FSResource_open(void * user) {
    VegaExt_FSResourceArgs * arg = (VegaExt_FSResourceArgs *)user;
    FSResource * self = vegaAlloc(sizeof(FSResource));
    self->file = fopen(arg->path, arg->mode);
    return self;
}

void FSResource_close(void * self) {
    fclose(((FSResource *)self)->file);
    vegaFree(self);
}

size_t FSResource_read(void * self, size_t size, size_t nmemb, void * buf) {
    return fread(buf, size, nmemb, ((FSResource *)self)->file);
}

size_t FSResource_write(void * self, size_t size, size_t nmemb, void * buf) {
    return fwrite(buf, size, nmemb, ((FSResource *)self)->file);
}

long FSResource_seek(void * self, long off, int whence) {
    return fseek(((FSResource *)self)->file, off, whence);
}

const VegaResource VegaExt_FSResource = (VegaResource) {
    .type_size = sizeof(FSResource),
    .open = &FSResource_open,
    .close = &FSResource_close,
    .read = &FSResource_read,
    .write = &FSResource_write,
    .seek = &FSResource_seek,
};