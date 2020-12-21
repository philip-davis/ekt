#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <margo.h>

#include <mpi.h>

#include "ekt.h"
#include "ekt_types.h"

#define EKT_FILE_SUFFIX ".ekt"

static char *get_address(margo_instance_id mid)
{
    hg_addr_t my_addr;
    hg_size_t my_addr_size;
    char *my_addr_str;

    margo_addr_self(mid, &my_addr);
    margo_addr_to_string(mid, NULL, &my_addr_size, my_addr);
    my_addr_str = malloc(my_addr_size);
    margo_addr_to_string(mid, my_addr_str, &my_addr_size, my_addr);
    margo_addr_free(mid, my_addr);

    return (my_addr_str);
}

DECLARE_MARGO_RPC_HANDLER(hello_rpc);
static void hello_rpc(hg_handle_t h);

static int gather_addresses(struct ekt_id *ekth)
{
    MPI_Comm gather_comm;
    MPI_Comm collector_comm;
    int per_collector = 1;
    int grank, gsize, crank, csize;
    int *addr_sizes = NULL;
    int *size_psum = NULL;
    int my_addr_size;
    int addr_buf_size = 0;
    char *addr_str_buf;
    int i;

    while(per_collector * per_collector < ekth->app_size) {
        per_collector *= 1;
    }

    MPI_Comm_split(ekth->comm, ekth->rank / per_collector, ekth->rank,
                   &gather_comm);
    MPI_Comm_rank(gather_comm, &grank);
    MPI_Comm_size(gather_comm, &gsize);

    if(grank == 0) {
        addr_sizes = malloc(sizeof(*addr_sizes) * gsize);
        size_psum = malloc(sizeof(*size_psum) * gsize);
    }
    my_addr_size = strlen(ekth->my_addr_str) + 1;
    MPI_Gather(&my_addr_size, 1, MPI_INT, addr_sizes, 1, MPI_INT, 0,
               gather_comm);
    if(grank == 0) {
        addr_buf_size = 0;
        for(i = 0; i < gsize; i++) {
            size_psum[i] = addr_buf_size;
            addr_buf_size += addr_sizes[i];
        }
        addr_str_buf = malloc(addr_buf_size);
    }
    MPI_Gatherv(ekth->my_addr_str, my_addr_size, MPI_CHAR, addr_str_buf,
                addr_sizes, size_psum, MPI_CHAR, 0, gather_comm);
    if(grank == 0) {
        ekth->gather_addrs = malloc(sizeof(*ekth->gather_addrs) * gsize);
        for(i = 0; i < gsize; i++) {
            ekth->gather_addrs[i] = &addr_str_buf[size_psum[i]];
        }
        ekth->gather_count = gsize;
        free(addr_sizes);
        free(size_psum);
    } else {
        ekth->gather_addrs = NULL;
        ekth->gather_count = 0;
    }

    ekth->collector_addrs = NULL;
    ekth->collector_count = 0;
    ekth->collector_addrs_len = 0;

    // Maybe a better MPI operation for this...we only care about the ranks that
    // are gathering.
    MPI_Comm_split(ekth->comm, ekth->rank % per_collector, ekth->rank,
                   &collector_comm);
    if(grank == 0) {
        MPI_Comm_rank(collector_comm, &crank);
        MPI_Comm_size(collector_comm, &csize);

        if(crank == 0) {
            addr_sizes = malloc(sizeof(*addr_sizes * csize));
            size_psum = malloc(sizeof(*size_psum * csize));
        }
        MPI_Gather(&my_addr_size, 1, MPI_INT, addr_sizes, 1, MPI_INT, 0,
                   collector_comm);
        if(crank == 0) {
            addr_buf_size = 0;
            for(i = 0; i < csize; i++) {
                size_psum[i] = my_addr_size;
                addr_buf_size += addr_sizes[i];
            }
            addr_str_buf = malloc(addr_buf_size);
        }
        MPI_Gatherv(ekth->my_addr_str, my_addr_size, MPI_CHAR, addr_str_buf,
                    addr_sizes, size_psum, MPI_CHAR, 0, collector_comm);
        if(crank == 0) {
            ekth->collector_addrs = addr_str_buf;
            ekth->collector_count = csize;
            ekth->collector_addrs_len = addr_buf_size;
        }
    }

    MPI_Comm_free(&gather_comm);
    MPI_Comm_free(&collector_comm);
}

static int write_bootstrap_conf(struct ekt_id *ekth)
{
    const char *suffix = EKT_FILE_SUFFIX;
    char *fname;
    size_t fname_len;
    int file;
    int file_locked = 0;
    struct flock lock = {
        .l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 0, .l_len = 0};
    char eofc = 0x0a;
    int ret;

    fname_len = strlen(ekth->app_name) + strlen(suffix) + 1;
    fname = malloc(fname_len);
    strcpy(fname, ekth->app_name);
    strcat(fname, suffix);

    file = open(fname, O_CREAT | O_EXCL | O_WRONLY, S_IRUSR | S_IWUSR);
    if(file < 0) {
        fprintf(stderr,
                "ERROR: %s: unable to open '%s' for writing: ", __func__,
                fname);
        perror(NULL);
        return (-1);
    }

    if(fcntl(file, F_SETLK, &lock) < 0) {
        fprintf(stderr, "ERROR: %s: unable to lock '%s': ", __func__, fname);
        perror(NULL);
        close(file);
        return (-1);
    }

    write(file, ekth->my_addr_str, strlen(ekth->my_addr_str) + 1);
    fsync(file);

    lock.l_type = F_UNLCK;
    if(fcntl(file, F_SETLK, &lock) < 0) {
        fprintf(stderr,
                "WARNING: %s: could not unlock '%s'. This will cause problems "
                "with other apps connecting. ",
                __func__, fname);
    }

    close(file);
    free(fname);
}

static void delete_bootstrap_conf(struct ekt_id *ekth)
{
    const char *suffix = EKT_FILE_SUFFIX;
    size_t fname_len;
    char *fname;

    fname_len = strlen(ekth->app_name) + strlen(suffix) + 1;
    fname = malloc(fname_len);
    strcpy(fname, ekth->app_name);
    strcat(fname, suffix);

    unlink(fname);
    free(fname);
}

int ekt_init(struct ekt_id **ekt_handle, const char *app_name, MPI_Comm comm,
             margo_instance_id mid)
{
    struct ekt_id *ekth;
    hg_bool_t flag;
    hg_id_t id;
    int i;

    ekth = malloc(sizeof(**ekt_handle));

    for(i = 0; i < EKT_WATCH_HASH_SIZE; i++) {
        ekth->watch_cbs[i] = NULL;
    }

    ekth->app_name = strdup(app_name);

    MPI_Comm_dup(comm, &ekth->comm);
    MPI_Comm_rank(comm, &ekth->rank);
    MPI_Comm_size(comm, &ekth->app_size);

    ekth->mid = mid;

    margo_registered_name(mid, "hello_rpc", &id, &flag);

    if(flag == HG_TRUE) { /* Provider already registered */
        margo_registered_name(mid, "hello_rpc", &ekth->hello_id, &flag);
    } else {
        //ekth->hello_id = MARGO_REGISTER(mid, "hello_rpc", hello_in_t, hello_out_t, hello_rpc);
        margo_register_data(mid, ekth->hello_id, (void *)ekth, NULL);
    }

    ekth->my_addr_str = get_address(mid);
    gather_addresses(ekth);

    if(ekth->rank == 0) {
        write_bootstrap_conf(ekth);
    }

    *ekt_handle = ekth;

    return (0);
}

int ekt_fini(struct ekt_id **ekt_handle)
{
    struct ekt_id *ekth = *ekt_handle;
    struct watch_list_node **cb_node, *cb_node_next;
    int i;

    if(ekth->rank == 0) {
        delete_bootstrap_conf(ekth);
    }

    for(i = 0; i < EKT_WATCH_HASH_SIZE; i++) {
        cb_node = &ekth->watch_cbs[i];
        while(*cb_node) {
            cb_node_next = (*cb_node)->next;
            free(*cb_node);
            cb_node = &cb_node_next;
        }
    }

    free(ekth->my_addr_str);
    free(ekth->app_name);
    MPI_Comm_free(&ekth->comm);

    free(ekth);
    *ekt_handle = NULL;
}

int ekt_connect(struct ekt_id *ekt_handle, const char *peer)
{
    // rank 0 reads bootstrapping conf

    /* rank 0 contacts rank 0 of peer:
            send greeting rpc
            response is peer's collector_addrs and app size
    */

    // rank 0 broadcasts peer app size

    // calculate mapping

    // rank 0 scatters collector addresses as necessary

    /* all contact peer ranks to buildup mesh
        ask peer collector nodes for outgoing peer rank addresses
        local collector nodes provide addresses to incoming peer ranks
    */
}

int ekt_watch(struct ekt_id *ekt_handle, struct ekt_type *type, watch_fn cb)
{
    int type_id = type->type_id;
    int type_hash = type_id % EKT_WATCH_HASH_SIZE;
    struct watch_list_node **cb_node = &ekt_handle->watch_cbs[type_hash];

    while(*cb_node) {
        if((*cb_node)->type_id) {
            fprintf(stderr, "ERROR: %s: already watching type_id %i.\n",
                    __func__, type_id);
            return (-1);
        }
        cb_node = &((*cb_node)->next);
    }
    *cb_node = malloc(sizeof(**cb_node));

    (*cb_node)->type_id = type_id;
    (*cb_node)->cb = cb;
    (*cb_node)->next = NULL;

    return (0);
}

int ekt_handle_announce(struct ekt_id *ekt_handle, struct ekt_type *type,
                        void *data)
{
    int type_id = type->type_id;
    int type_hash = type_id % EKT_WATCH_HASH_SIZE;
    struct watch_list_node *cb_node = ekt_handle->watch_cbs[type_hash];

    while(cb_node) {
        if(cb_node->type_id == type_id) {
            break;
        }
        cb_node = cb_node->next;
    };

    if(!cb_node) {
        fprintf(stderr,
                "WARNING: received announcement with unknown type %d.\n",
                type_id);
        return (0);
    }

    return (cb_node->cb(data, type->arg));
}

int ekt_tell(struct ekt_id *ekt_handle, struct ekt_type *type, void *data)
{

    // transmit to all other ekt clients
    // handle locally
    ekt_handle_announce(ekt_handle, type, data);

    return (0);
}

int ekt_register(uint32_t type_id, serdes_fn ser, serdes_fn des, void *arg,
                 struct ekt_type **type)
{
    struct ekt_type *ektt;

    *type = malloc(sizeof(**type));
    ektt = *type;

    ektt->ser = ser;
    ektt->des = des;
    ektt->arg = arg;
    ektt->type_id = type_id;

    return (0);
}

int ekt_deregister(struct ekt_type **type)
{
    free(*type);
    *type = NULL;
}
