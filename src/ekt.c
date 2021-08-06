#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <abt.h>
#include <margo.h>

#include <mpi.h>

#include "ekt.h"
#include "ekt_types.h"

#define EKT_FILE_SUFFIX ".ekt"

int ekt_handle_announce(struct ekt_id *ekt_handle, int type_id, void *data);

static void map_from_collector(int collector, int size, int *start, int *end);

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

DECLARE_MARGO_RPC_HANDLER(query_addrs_rpc);
static void query_addrs_rpc(hg_handle_t h);

DECLARE_MARGO_RPC_HANDLER(tell_rpc);
static void tell_rpc(hg_handle_t h);

static int map_to_collector(int rank, int size);

static int gather_addresses(struct ekt_id *ekth)
{
    MPI_Comm gather_comm;
    MPI_Comm collector_comm;
    int collector_count, per_collector;
    int grank, gsize, crank, csize;
    int *addr_sizes = NULL;
    int *size_psum = NULL;
    int my_addr_size;
    int addr_buf_size = 0;
    char *addr_str_buf;
    int i;

    collector_count = 1;
    while(collector_count * collector_count < ekth->app_size) {
        collector_count++;
    }
    per_collector = ekth->app_size / collector_count;

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
        ekth->gather_addrs_len = addr_buf_size;
        ekth->gather_count = gsize;
        free(addr_sizes);
        free(size_psum);
    } else {
        ekth->gather_addrs = NULL;
        ekth->gather_count = 0;
        ekth->gather_addrs_len = 0;
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
            addr_sizes = malloc(sizeof(*addr_sizes) * csize);
            size_psum = malloc(sizeof(*size_psum) * csize);
        }
        MPI_Gather(&my_addr_size, 1, MPI_INT, addr_sizes, 1, MPI_INT, 0,
                   collector_comm);
        if(crank == 0) {
            addr_buf_size = 0;
            for(i = 0; i < csize; i++) {
                size_psum[i] = addr_buf_size;
                addr_buf_size += addr_sizes[i];
            }
            addr_str_buf = malloc(addr_buf_size);
        }

        MPI_Gatherv(ekth->my_addr_str, my_addr_size, MPI_CHAR, addr_str_buf,
                    addr_sizes, size_psum, MPI_CHAR, 0, collector_comm);
        ekth->collector_count = csize;
        if(crank == 0) {
            ekth->collector_addrs = addr_str_buf;
            ekth->collector_addrs_len = addr_buf_size;
        }
    }

    MPI_Comm_dup(gather_comm, &ekth->gather_comm);
    MPI_Comm_dup(collector_comm, &ekth->collector_comm);
    MPI_Comm_free(&gather_comm);
    MPI_Comm_free(&collector_comm);
}

char *get_conf_name(const char *app_name)
{
    const char *suffix = EKT_FILE_SUFFIX;
    char *fname;
    size_t fname_len;

    fname_len = strlen(app_name) + strlen(suffix) + 1;
    fname = malloc(fname_len);
    strcpy(fname, app_name);
    strcat(fname, suffix);

    return (fname);
}

static int write_bootstrap_conf(struct ekt_id *ekth)
{
    char *fname;
    int file;
    int file_locked = 0;
    struct flock lock = {
        .l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 0, .l_len = 0};
    char eofc = 0x0a;
    int ret;

    fname = get_conf_name(ekth->app_name);

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

static int read_bootstrap_conf(struct ekt_id *ekth, const char *peer,
                               char **peer_addr)
{
    char *fname;
    struct stat st;
    struct flock lock = {
        .l_type = F_RDLCK, .l_whence = SEEK_SET, .l_start = 0, .l_len = 0};
    int file;

    fname = get_conf_name(peer);

    while(1) {
        if(stat(fname, &st) < 0) {
            if(errno == ENOENT) {
                sleep(1);
                continue;
            } else {
                fprintf(
                    stderr,
                    "ERROR: could not open bootstrapping file '%s': ", fname);
                perror(NULL);
                return (-1);
            }
        } else {
            // there is a brief period between the writer opening the file and
            // locking it...
            while(st.st_size == 0) {
                sleep(1);
                stat(fname, &st);
            }
            break;
        }
    }

    file = open(fname, O_RDONLY);
    if(file < 0) {
        fprintf(stderr,
                "ERROR: %s: unable to open '%s' for reading: ", __func__,
                fname);
        perror(NULL);
        return (-1);
    }

    if(fcntl(file, F_SETLKW, &lock) < 0) {
        fprintf(stderr, "ERROR: %s: unable to lock '%s': ", __func__, fname);
        perror(NULL);
        close(file);
        return (-1);
    }

    // Now we can trust that the writer has finished writing
    stat(fname, &st);
    *peer_addr = malloc(st.st_size);
    read(file, *peer_addr, st.st_size);

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
    int mybegin, myend;
    char *fname;

    fname_len = strlen(ekth->app_name) + strlen(suffix) + 1;
    fname = malloc(fname_len);
    strcpy(fname, ekth->app_name);
    strcat(fname, suffix);

    unlink(fname);
    free(fname);
}

static void hello_rpc(hg_handle_t handle)
{
    margo_instance_id mid;
    const struct hg_info *info;
    struct ekt_id *ekth;
    hello_in_t in;
    hello_out_t out;
    hg_return_t hret;

    mid = margo_hg_handle_get_instance(handle);
    info = margo_get_info(handle);
    ekth = (struct ekt_id *)margo_registered_data(mid, info->id);

    hret = margo_get_input(handle, &in);

    out.size = ekth->app_size;
    out.addrs.len = ekth->collector_addrs_len;
    out.addrs.buf = (void *)ekth->collector_addrs;

    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(hello_rpc)

static void query_addrs_rpc(hg_handle_t handle)
{
    margo_instance_id mid;
    const struct hg_info *info;
    struct ekt_id *ekth;
    query_addrs_in_t in;
    ekt_buf_t out;
    int crank;
    int mystart, myend;
    hg_return_t hret;

    mid = margo_hg_handle_get_instance(handle);
    info = margo_get_info(handle);
    ekth = (struct ekt_id *)margo_registered_data(mid, info->id);

    if(ekth->gather_count == 0) {
        fprintf(stderr,
                "ERROR: (%s): received an an address query at a non-collector "
                "rank.\n",
                __func__);
    }

    hret = margo_get_input(handle, &in);
    MPI_Comm_rank(ekth->collector_comm, &crank);

    if(map_to_collector(in.start, ekth->app_size) != crank ||
       map_to_collector(in.end, ekth->app_size) != crank) {
        fprintf(stderr,
                "ERROR: (%s): received an address query for ranks %d through "
                "%d, which is outside the range managed by rank %d.\n",
                __func__, in.start, in.end, ekth->rank);
    }

    map_from_collector(crank, ekth->app_size, &mystart, &myend);
    if(map_to_collector(in.end + 1, ekth->app_size) == crank) {
        out.len = ekth->gather_addrs[(in.end + 1) - mystart] -
                  ekth->gather_addrs[in.start - mystart];
    } else {
        out.len =
            ekth->gather_addrs_len -
            (ekth->gather_addrs[in.start - mystart] - ekth->gather_addrs[0]);
    }

    out.buf = malloc(out.len);
    memcpy(out.buf, ekth->gather_addrs[in.start - mystart], out.len);
    margo_respond(handle, &out);

    margo_free_input(handle, &in);
    margo_destroy(handle);
    free(out.buf);
}
DEFINE_MARGO_RPC_HANDLER(query_addrs_rpc)

static int deser_tell_data(struct ekt_id *ekth, int type_id, void *ser_data,
                           void **deser_data)
{
    int type_hash = type_id % EKT_WATCH_HASH_SIZE;
    struct watch_list_node *cb_node = ekth->watch_cbs[type_hash];
    struct ekt_type *type;

    while(cb_node) {
        if(cb_node->type_id == type_id) {
            break;
        }
        cb_node = cb_node->next;
    }

    if(!cb_node) {
        fprintf(stderr,
                "WARNING: received announcement with unknown type %d.\n",
                type_id);
        return (-1);
    }

    type = cb_node->type;
    type->des(ser_data, type->arg, deser_data);

    return (0);
}

static void tell_rpc(hg_handle_t handle)
{
    margo_instance_id mid;
    const struct hg_info *info;
    struct ekt_id *ekth;
    tell_in_t in;
    void *data;
    hg_return_t hret;

    mid = margo_hg_handle_get_instance(handle);
    info = margo_get_info(handle);
    ekth = (struct ekt_id *)margo_registered_data(mid, info->id);

    hret = margo_get_input(handle, &in);

    deser_tell_data(ekth, in.type_id, in.data.buf, &data);
    ekt_handle_announce(ekth, in.type_id, data);

    free(data);

    margo_free_input(handle, &in);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(tell_rpc)

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
        margo_registered_name(mid, "query_addr_rpc", &ekth->query_addrs_id,
                              &flag);
        margo_registered_name(mid, "tell_rpc", &ekth->tell_id, &flag);
    } else {
        ekth->hello_id = MARGO_REGISTER(mid, "hello_rpc", hello_in_t,
                                        hello_out_t, hello_rpc);
        margo_register_data(mid, ekth->hello_id, (void *)ekth, NULL);
        ekth->query_addrs_id =
            MARGO_REGISTER(mid, "query_addr_rpc", query_addrs_in_t, ekt_buf_t,
                           query_addrs_rpc);
        margo_register_data(mid, ekth->query_addrs_id, (void *)ekth, NULL);
        ekth->tell_id =
            MARGO_REGISTER(mid, "tell_id", tell_in_t, void, tell_rpc);
        margo_register_data(mid, ekth->tell_id, (void *)ekth, NULL);
        margo_registered_disable_response(mid, ekth->tell_id, HG_TRUE);
    }

    ekth->my_addr_str = get_address(mid);
    gather_addresses(ekth);

    ABT_mutex_create(&ekth->peer_mutex);
    ekth->peers = NULL;

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
    MPI_Comm_free(&ekth->gather_comm);
    MPI_Comm_free(&ekth->collector_comm);

    free(ekth);
    *ekt_handle = NULL;
}

static int get_collector_count(int app_size)
{
    int count = 1;

    while((count * count) < app_size) {
        count++;
    }

    return (count);
}

static int map_to_collector(int rank, int size)
{
    int count = get_collector_count(size);

    return (rank / (size / count));
}

static void map_from_collector(int collector, int size, int *start, int *end)
{
    int count = get_collector_count(size);
    int stride = size / count;

    *start = stride * collector;
    *end = *start + (stride - 1);
    if(*end >= size) {
        *end = size - 1;
    }
}

static int *partition_seq(int seq_size, int part_count)
{
    int div, rem, str, i;
    int *parts;

    div = seq_size / part_count;
    rem = seq_size % part_count;

    parts = malloc(sizeof(*parts) * part_count);

    for(i = 0; i < part_count; i++) {
        if(part_count < seq_size) {
            parts[i] = div;
            if(i < rem) {
                parts[i]++;
            }
        } else if((i < (part_count - rem)) &&
                  (i % (part_count / seq_size)) == 0) {
            parts[i] = 1;
        } else {
            parts[i] = 0;
        }
    }

    return (parts);
}

static void distribute_g2c(struct ekt_id *ekth, int peer_size,
                           char **peer_caddrs, int *addrs_count, char **caddr)
{
    int *cmap, *gmap;
    int crank;
    int first, last, firstc, lastc;
    int *addr_list_lens;
    char **srcs;
    MPI_Request req;
    int addrs_len;
    int i, j;

    MPI_Comm_rank(ekth->collector_comm, &crank);
    cmap = partition_seq(peer_size, ekth->collector_count);
    if(ekth->rank == 0) {
        addr_list_lens =
            malloc(sizeof(*addr_list_lens) * ekth->collector_count);
        srcs = malloc(sizeof(*srcs) * ekth->collector_count);
    }
    *addrs_count = 0;
    first = 0;
    for(i = 0; i < ekth->collector_count; i++) {
        if(cmap[i]) {
            firstc = map_to_collector(first, peer_size);
            last = first + (cmap[i] - 1);
            lastc = map_to_collector(last, peer_size);
            first = first + cmap[i];
            if(ekth->rank == 0) {
                addr_list_lens[i] = 0;
                for(j = firstc; j <= lastc; j++) {
                    addr_list_lens[i] += strlen(peer_caddrs[j]) + 1;
                }
                srcs[i] = peer_caddrs[firstc];
            }
            if(i == crank) {
                *addrs_count = (lastc - firstc) + 1;
            }
        }
    }
    MPI_Scatter(addr_list_lens, 1, MPI_INT, &addrs_len, 1, MPI_INT, 0,
                ekth->collector_comm);

    *caddr = malloc(addrs_len);
    if(ekth->rank == 0) {
        memcpy(*caddr, peer_caddrs[0], addrs_len);
        for(i = 1; i < ekth->collector_count; i++) {
            MPI_Isend(srcs[i], addr_list_lens[i], MPI_BYTE, i, 0,
                      ekth->collector_comm, &req);
        }
    } else {
        MPI_Recv(*caddr, addrs_len, MPI_BYTE, 0, 0, ekth->collector_comm,
                 MPI_STATUS_IGNORE);
    }

    free(cmap);
    if(ekth->rank == 0) {
        free(srcs);
        free(addr_list_lens);
    }
}

static int query_addrs(struct ekt_id *ekth, char *tgt_addr, int lower,
                       int upper, int *addrs_len, char **addrs)
{
    hg_addr_t addr;
    hg_handle_t handle;
    query_addrs_in_t in;
    ekt_buf_t out;
    hg_return_t hret;

    in.start = lower;
    in.end = upper;

    margo_addr_lookup(ekth->mid, tgt_addr, &addr);
    margo_create(ekth->mid, addr, ekth->query_addrs_id, &handle);
    hret = margo_forward(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%d): margo_forward() failed.\n", hret);
        margo_destroy(handle);
        return (-1);
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%d): margo_get_output() failed.\n", hret);
        margo_destroy(handle);
        return (-1);
    }

    *addrs_len = out.len;
    *addrs = malloc(out.len);
    memcpy(*addrs, out.buf, out.len);

    margo_free_output(handle, &out);
    margo_destroy(handle);
    margo_addr_free(ekth->mid, addr);

    return (0);
}

static void get_gather_peer_range(struct ekt_id *ekth, int peer_size,
                                  int *start, int *end)
{
    int crank;
    int *cmap;
    int i;

    MPI_Comm_rank(ekth->collector_comm, &crank);
    cmap = partition_seq(peer_size, ekth->collector_count);

    *start = 0;
    for(i = 0; i < crank; i++) {
        *start += cmap[i];
    }

    *end = *start + (cmap[crank] - 1);

    free(cmap);
}

static int get_peer_caddr_count(struct ekt_id *ekth, int peer_size)
{
    int collector_count = get_collector_count(ekth->app_size);
    int my_collector = ekth->rank / collector_count;
    int *cmap;
    int count;

    cmap = partition_seq(peer_size, collector_count);
    count = cmap[my_collector];
    free(cmap);

    return (count);
}

static void get_peer_addrs(struct ekt_id *ekth, int peer_size,
                           int pcoll_addr_count, char **pcoll_addrs,
                           int *peer_buf_len, char **addr_list_buf)
{
    int start, end;
    int coffset;
    int qstart, qend;
    char **peer_addrs;
    char *res_addrs[pcoll_addr_count];
    int res_addr_len[pcoll_addr_count];
    int res_addr_count[pcoll_addr_count];
    int cpeer;
    int i;

    get_gather_peer_range(ekth, peer_size, &start, &end);

    *peer_buf_len = 0;
    coffset = map_to_collector(start, peer_size);
    for(i = 0; i < pcoll_addr_count; i++) {
        cpeer = coffset + i;
        map_from_collector(cpeer, peer_size, &qstart, &qend);
        if(qstart < start) {
            qstart = start;
        }
        if(qend > end) {
            qend = end;
        }
        query_addrs(ekth, pcoll_addrs[i], qstart, qend, &res_addr_len[i],
                    &res_addrs[i]);
        *peer_buf_len += res_addr_len[i];
    }

    *addr_list_buf = malloc(*peer_buf_len);
    for(i = 0; i < pcoll_addr_count; i++) {
        memcpy(*addr_list_buf, res_addrs[i], res_addr_len[i]);
        addr_list_buf += res_addr_len[i];
        free(res_addrs[i]);
    }
}

/*
static void distribute_collectors(struct ekt_id *ekth, int peer_size,
                                  char **collector_addrs,
                                  int *addrs_len,
                                  char **addr_list)
{
    int *gather_map;
    int *collector_map;
    int peer_rank = 0;
    int *addr_list_lens;
    char **collector_src;
    int peer_collector_count;
    int first_collector, last_collector;
    MPI_Request req;
    int i, j;

    peer_collector_count = get_collector_count(peer_size);

    collector_src = malloc(sizeof(*collector_src) * ekth->collector_count);
    gather_map = malloc(sizeof(*gather_map) * ekth->collector_count);
    addr_list_lens = malloc(sizeof(*addr_list_lens) * ekth->collector_count);

    calc_gather_map(gather_map, ekth->collector_count, peer_size);
    if(ekth->rank == 0) {
        for(i = 0; i < ekth->collector_count; i++) {
            if(gather_map[i] > 0) {
                first_collector = peer_rank / peer_collector_count;
                last_collector =
                    (peer_rank + (gather_map[i] - 1)) / peer_collector_count;
                addr_list_lens[i] = 0;
                for(j = first_collector; j <= last_collector; j++) {
                    addr_list_lens[i] += strlen(collector_addrs[i]) + 1;
                }
                collector_src[i] = collector_addrs[first_collector];
            }
            peer_rank += gather_map[i];
        }
    }
    MPI_Scatter(addr_list_lens, 1, MPI_INT, addrs_len, 1, MPI_INT, 0,
                ekth->collector_comm);

    if(ekth->rank == 0) {
        *addr_list = collector_addrs[0];
        for(i = 1; i < ekth->collector_count; i++) {
            if(gather_map[i] > 0) {
                MPI_Isend(collector_src[i], addr_list_lens[i], MPI_BYTE, i, 0,
                          ekth->collector_comm, &req);
            }
        }
    } else {
        *addr_list = malloc(*addrs_len);
        MPI_Recv(*addr_list, *addrs_len, MPI_BYTE, 0, 0, ekth->collector_comm,
                 MPI_STATUS_IGNORE);
    }

    printf("address: %s, addr_len: %d\n", *addr_list, *addrs_len);
}
*/

static void deserialize_str_list(char *str, int count, char ***listp)
{
    char **list;
    int i;

    *listp = malloc(count * sizeof(*listp));
    list = *listp;

    for(i = 0; i < count; i++) {
        list[i] = str;
        while(*(str++) != '\0')
            ;
    }
}

static void distribute_peers(struct ekt_id *ekth, char **peer_addrs,
                             int peer_caddr_count, int *local_pcount,
                             char ***my_peers)
{
    int grank, gsize;
    int offset;
    int *sendcounts;
    int *displs;
    int local_psize;
    char *my_peers_buf;
    int *pmap;
    int first = 0;
    void *scatter_buf;
    int i, j;

    MPI_Comm_rank(ekth->gather_comm, &grank);
    MPI_Comm_size(ekth->gather_comm, &gsize);

    pmap = partition_seq(peer_caddr_count, gsize);
    *local_pcount = pmap[grank];
    if(grank == 0) {
        scatter_buf = *peer_addrs;
        offset = ekth->rank;
        pmap = partition_seq(peer_caddr_count, ekth->gather_count);
        displs = malloc(sizeof(*displs) * ekth->gather_count);
        sendcounts = calloc(sizeof(*sendcounts), ekth->gather_count);
        for(i = 0; i < ekth->gather_count; i++) {
            displs[i] = peer_addrs[first] - *peer_addrs;
            for(j = first; j < first + pmap[i]; j++) {
                sendcounts[i] += strlen(peer_addrs[j]) + 1;
            }
            first += pmap[i];
        }
    }
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_psize, 1, MPI_INT, 0,
                ekth->gather_comm);
    my_peers_buf = malloc(local_psize);
    MPI_Scatterv(scatter_buf, sendcounts, displs, MPI_CHAR, my_peers_buf,
                 local_psize, MPI_CHAR, 0, ekth->gather_comm);
    deserialize_str_list(my_peers_buf, *local_pcount, my_peers);
    int detected_count = 0;
    for(i = 0; i < local_psize; i++) {
        if(my_peers_buf[i] == '\0')
            detected_count++;
    }
    if(detected_count != *local_pcount) {
        fprintf(stderr,
                "ERROR: rank %d received %d addrs, expected %d addrs.\n",
                ekth->rank, detected_count, *local_pcount);
    }
}

void add_peer(struct ekt_id *ekth, const char *name, int size, int addr_count,
              char **addrs)
{
    struct ekt_peer *peer, **peerp;

    ABT_mutex_lock(ekth->peer_mutex);
    peerp = &ekth->peers;
    while(*peerp) {
        if(strcmp((*peerp)->name, name) == 0) {
            // we already know about this peer.
            break;
        }
        peerp = &(*peerp)->next;
    }
    if(!(*peerp)) {
        // we exhausted the known peers looking for the new one. Safe to add
        peer = malloc(sizeof(*peer));
        peer->next = NULL;
        peer->name = strdup(name);
        peer->size = size;
        peer->rank_count = addr_count;
        peer->peer_addrs = addrs;
        *peerp = peer;
    } else if((*peerp)->size == 0) {
        // we found a placeholder entry
        peer = *peerp;
        peer->size = size;
        peer->rank_count = addr_count;
        peer->peer_addrs = addrs;
    }
    ABT_mutex_unlock(ekth->peer_mutex);
}

int ekt_connect(struct ekt_id *ekth, const char *peer)
{
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;
    hello_in_t hin;
    hello_out_t hout;
    int peer_size;
    char *boot_addr;
    int peer_collector_count;
    char **peer_collector_addrs;
    int pcoll_addr_count;
    int pcoll_size;
    int peer_addr_buf_len;
    char *peer_addr_buf;
    char *pcoll_addrs_buf;
    char **pcoll_addrs;
    char **peer_addrs;
    int peer_caddr_count;
    int my_pcount;
    char **my_peers;

    if(ekth->rank == 0) {
        // rank 0 reads bootstrapping conf
        if(read_bootstrap_conf(ekth, peer, &boot_addr) < 0) {
            return (-1);
        }

        /* rank 0 contacts rank 0 of peer:
                send greeting rpc
                response is peer's collector_addrs and app size
        */
        hin.name = ekth->app_name;
        hin.size = ekth->app_size;
        margo_addr_lookup(ekth->mid, boot_addr, &server_addr);
        hret = margo_create(ekth->mid, server_addr, ekth->hello_id, &handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create hello rpc (%d)\n", hret);
            return (-1);
        }
        hret = margo_forward(handle, &hin);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: failed to send hello rpc (%d)\n", hret);
            margo_destroy(handle);
            return (-1);
        }

        hret = margo_get_output(handle, &hout);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: failed to get status of hello rpc (%d)\n",
                    hret);
            margo_destroy(handle);
            return (-1);
        }
        peer_size = hout.size;

        free(boot_addr);
    }

    // rank 0 broadcasts peer app size
    peer_size = hout.size;
    MPI_Bcast(&peer_size, 1, MPI_INT, 0, ekth->comm);

    peer_collector_count = get_collector_count(peer_size);
    peer_caddr_count = get_peer_caddr_count(ekth, peer_size);

    if(ekth->gather_count > 0) {
        if(ekth->rank == 0) {
            deserialize_str_list(hout.addrs.buf, peer_collector_count,
                                 &peer_collector_addrs);
        }
        // rank 0 scatters collector addresses as necessary
        distribute_g2c(ekth, peer_size, peer_collector_addrs, &pcoll_addr_count,
                       &pcoll_addrs_buf);
        deserialize_str_list(pcoll_addrs_buf, pcoll_addr_count, &pcoll_addrs);
        get_peer_addrs(ekth, peer_size, pcoll_addr_count, pcoll_addrs,
                       &peer_addr_buf_len, &peer_addr_buf);
        peer_caddr_count = get_peer_caddr_count(ekth, peer_size);
        deserialize_str_list(peer_addr_buf, peer_caddr_count, &peer_addrs);
    }

    /* all contact peer ranks to buildup mesh
        ask peer collector nodes for outgoing peer rank addresses
        local collector nodes provide addresses to incoming peer ranks
    */
    distribute_peers(ekth, peer_addrs, peer_caddr_count, &my_pcount, &my_peers);
    add_peer(ekth, peer, peer_size, my_pcount, my_peers);

    return (0);
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
    (*cb_node)->type = type;
    (*cb_node)->next = NULL;

    return (0);
}

int ekt_handle_announce(struct ekt_id *ekt_handle, int type_id, void *data)
{
    int type_hash = type_id % EKT_WATCH_HASH_SIZE;
    struct watch_list_node *cb_node = ekt_handle->watch_cbs[type_hash];

    while(cb_node) {
        if(cb_node->type_id == type_id) {
            break;
        }
        cb_node = cb_node->next;
    }

    if(!cb_node) {
        fprintf(stderr,
                "WARNING: received announcement with unknown type %d.\n",
                type_id);
        return (0);
    }

    return (cb_node->cb(data, cb_node->type->arg));
}

static int ekt_tell_peer(struct ekt_id *ekth, struct ekt_peer *peer,
                         struct ekt_type *type, void *data, int data_size)
{
    int type_id;
    tell_in_t in;
    hg_handle_t handle;
    hg_addr_t peer_addr;
    hg_return_t hret;
    int i;

    if(!type) {
        fprintf(stderr, "ERROR: bad type passed to ekt_tell.\n");
        return (-1);
    }
    in.type_id = type->type_id;
    in.data.len = data_size;
    in.data.buf = data;

    for(i = 0; i < peer->rank_count; i++) {
        margo_addr_lookup(ekth->mid, peer->peer_addrs[i], &peer_addr);
        hret = margo_create(ekth->mid, peer_addr, ekth->tell_id, &handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create hello rpc (%d)\n", hret);
            return (-1);
        }
        hret = margo_forward(handle, &in);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: failed to send hello rpc (%d)\n", hret);
            margo_destroy(handle);
            return (-1);
        }
    }
}

int ekt_tell(struct ekt_id *ekth, const char *target, struct ekt_type *type, void *data)
{
    struct ekt_peer *peer;
    int data_size;
    void *ser_data = NULL;

    // transmit to all other ekt clients
    peer = ekth->peers;
    if(peer) {
        data_size = type->ser(data, type->arg, &ser_data);
    }

    while(peer) {
        if(!target || strcmp(target, peer->name) == 0) {
            ekt_tell_peer(ekth, peer, type, ser_data, data_size);
        }
        peer = peer->next;
    }

    if(ser_data) {
        free(ser_data);
    }

    // handle locally
    ekt_handle_announce(ekth, type->type_id, data);

    return (0);
}

int ekt_register(struct ekt_id *ekth, uint32_t type_id, serdes_fn ser,
                 serdes_fn des, void *arg, struct ekt_type **type)
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
