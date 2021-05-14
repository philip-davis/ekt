#ifndef _EKT_H_
#define _EKT_H_

#include <margo.h>
#include <mpi.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef int (*serdes_fn)(void *, void *, void **);
typedef int (*watch_fn)(void *, void *);

typedef struct ekt_id *ekt_id;
typedef struct ekt_type *ekt_type;
typedef struct ekt_peer *ekt_peer;

int ekt_init(struct ekt_id **ekt_handle, const char *app_name, MPI_Comm comm,
             margo_instance_id mid);

int ekt_fini(struct ekt_id **ekt_handle);

int ekt_connect(struct ekt_id *ekt_handle, const char *peer);

int ekt_watch(struct ekt_id *ekt_handle, struct ekt_type *type, watch_fn cb);

int ekt_tell(struct ekt_id *ekt_handle, struct ekt_type *type, void *data);

int ekt_register(uint32_t type_id, serdes_fn ser, serdes_fn des, void *arg, struct ekt_type **type);

int ekt_deregister(struct ekt_type **type);

#if defined(__cplusplus)
}
#endif

#endif
