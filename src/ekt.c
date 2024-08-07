#define _GNU_SOURCE
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
#include "tell_hash.h"

#define EKT_FILE_SUFFIX ".ekt"

#ifdef USE_APEX
#include <apex.h>
#define APEX_FUNC_TIMER_START(fn)                                              \
    apex_profiler_handle profiler0 = apex_start(APEX_FUNCTION_ADDRESS, &fn);
#define APEX_NAME_TIMER_START(num, name)                                       \
    apex_profiler_handle profiler##num = apex_start(APEX_NAME_STRING, name);
#define APEX_TIMER_STOP(num) apex_stop(profiler##num);
#else
#define APEX_FUNC_TIMER_START(fn) (void)0;
#define APEX_NAME_TIMER_START(num, name) (void)0;
#define APEX_TIMER_STOP(num) (void)0;
#endif

#define DEBUG_OUT(dstr, ...)                                                   \
    do {                                                                       \
        if(ekth->f_debug) {                                                    \
            ABT_unit_id tid = 0;                                               \
            if(ABT_initialized())                                              \
                ABT_thread_self_id(&tid);                                      \
            fprintf(stderr, "Rank %i: TID: %" PRIu64 " %s:%i (%s): " dstr,     \
                    ekth->rank, tid, __FILE__, __LINE__, __func__,             \
                    ##__VA_ARGS__);                                            \
        }                                                                      \
    } while(0);

#define WARN_OUT(wstr, ...)                                                    \
    do {                                                                       \
        ABT_unit_id tid = 0;                                                   \
        if(ABT_initialized())                                                  \
            ABT_thread_self_id(&tid);                                          \
        fprintf(stderr, "WARN: Rank %i: TID: %" PRIu64 " %s:%i (%s): " wstr,   \
                ekth->rank, tid, __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    } while(0);

#define ERR_OUT(estr, ...)                                                     \
    do {                                                                       \
        ABT_unit_id tid;                                                       \
        ABT_thread_self_id(&tid);                                              \
        fprintf(stderr, "ERROR: Rank: %i: TID: %" PRIu64 " %s:%i (%s): " estr, \
                ekth->rank, tid, __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    } while(0);

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

DECLARE_MARGO_RPC_HANDLER(query_status_rpc);
static void query_status_rpc(hg_handle_t h);

static int map_to_collector(int rank, int size);

int gather_addresses(struct ekt_id *ekth)
{
    MPI_Comm gather_comm;
    MPI_Comm collector_comm;
    int collector_count;
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

    MPI_Comm_split(ekth->comm, ekth->rank / collector_count, ekth->rank,
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
    MPI_Comm_split(ekth->comm, ekth->rank % collector_count, ekth->rank,
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

    DEBUG_OUT("gather_count = %i, collector_count = %i\n", ekth->gather_count,
              ekth->collector_count);

    MPI_Comm_dup(gather_comm, &ekth->gather_comm);
    MPI_Comm_dup(collector_comm, &ekth->collector_comm);
    MPI_Comm_free(&gather_comm);
    MPI_Comm_free(&collector_comm);

    return (0);
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
        ERR_OUT("unable to open '%s' for writing: ", fname);
        perror(NULL);
        return (-1);
    }

    if(fcntl(file, F_SETLK, &lock) < 0) {
        ERR_OUT("unable to lock '%s': ", fname);
        perror(NULL);
        close(file);
        return (-1);
    }

    write(file, ekth->my_addr_str, strlen(ekth->my_addr_str) + 1);
    fsync(file);

    lock.l_type = F_UNLCK;
    if(fcntl(file, F_SETLK, &lock) < 0) {
        WARN_OUT("could not unlock '%s'. This will cause problems with other "
                 "apps connecting.\n",
                 fname);
    }

    close(file);
    free(fname);

    return (0);
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
                ERR_OUT("could not open bootstrapping file '%s': ", fname);
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
        ERR_OUT("unable to open '%s' for reading: ", fname);
        perror(NULL);
        return (-1);
    }

    if(fcntl(file, F_SETLKW, &lock) < 0) {
        ERR_OUT("unable to lock '%s': ", fname);
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
        WARN_OUT("could not unlock '%s'. This will cause problems with other "
                 "apps connecting.\n",
                 fname);
    }

    close(file);
    free(fname);

    return (0);
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
    out.pid = __sync_fetch_and_add(&ekth->next_id, 1);
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
        ERR_OUT("received an an address query at a non-collector rank.\n")
    }

    hret = margo_get_input(handle, &in);
    MPI_Comm_rank(ekth->collector_comm, &crank);

    DEBUG_OUT("received address request for ranks %i through %i\n", in.start,
              in.end);

    if(map_to_collector(in.start, ekth->app_size) != crank ||
       map_to_collector(in.end, ekth->app_size) != crank) {
        ERR_OUT("received an address query for ranks %d through %d, which is "
                "outside the range managed by rank %d.\n",
                in.start, in.end, ekth->rank);
    }

    map_from_collector(crank, ekth->app_size, &mystart, &myend);
    DEBUG_OUT("my zone of responsibility is from rank %i to %i, and my total "
              "gather buffer is %i bytes\n",
              mystart, myend, ekth->gather_addrs_len);

    if(in.end == myend) {
        out.len =
            ekth->gather_addrs_len -
            (ekth->gather_addrs[in.start - mystart] - ekth->gather_addrs[0]);
    } else {
        out.len = ekth->gather_addrs[(in.end + 1) - mystart] -
                  ekth->gather_addrs[in.start - mystart];
    }

    DEBUG_OUT("length of response is %" PRIu64 "\n", out.len);

    out.buf = malloc(out.len);
    memcpy(out.buf, ekth->gather_addrs[in.start - mystart], out.len);
    DEBUG_OUT("response prepared\n");
    margo_respond(handle, &out);
    DEBUG_OUT("sent response\n");

    margo_free_input(handle, &in);
    margo_destroy(handle);
    DEBUG_OUT("completed address query\n");

    free(out.buf);
}
DEFINE_MARGO_RPC_HANDLER(query_addrs_rpc)

static int deser_tell_data(struct ekt_id *ekth, int type_id, void *ser_data,
                           void **deser_data)
{
    int type_hash = type_id % EKT_WATCH_HASH_SIZE;
    struct watch_list_node *cb_node = ekth->watch_cbs[type_hash];
    struct ekt_type *type;
    size_t res_size;

    while(cb_node) {
        if(cb_node->type_id == type_id) {
            break;
        }
        cb_node = cb_node->next;
    }

    if(!cb_node) {
        WARN_OUT("received announcement with unknown type %d.\n", type_id);
        return (-1);
    }

    type = cb_node->type;

    return(type->des(ser_data, type->arg, deser_data));
}

static void tell_rpc(hg_handle_t handle)
{
    margo_instance_id mid;
    const struct hg_info *info;
    struct ekt_id *ekth;
    tell_in_t in;
    void *data;
    hg_return_t hret;
    int recv_count;

    APEX_NAME_TIMER_START(0, "tell_rpc");
    mid = margo_hg_handle_get_instance(handle);
    info = margo_get_info(handle);
    ekth = (struct ekt_id *)margo_registered_data(mid, info->id);

    hret = margo_get_input(handle, &in);

    if(in.dst != ekth->rank) {
        ERR_OUT("received tell rpc with dst %i, but I am rank %i\n", in.dst,
                ekth->rank);
    }

    ABT_mutex_lock(ekth->peer_mutex);
    recv_count = incr_thash_entry(ekth->thash, in.data.buf, in.data.len, in.pid,
                                  in.type_id);
    ABT_mutex_unlock(ekth->peer_mutex);
    if(recv_count == in.degree) {
        DEBUG_OUT("received copy %i of identical type %i tell from %i with len "
                  "%" PRIu64 ", doing cb.\n",
                  recv_count, in.type_id, in.pid, in.data.len);
        if(deser_tell_data(ekth, in.type_id, in.data.buf, &data) == 0) {
            ekt_handle_announce(ekth, in.type_id, data);
            free(data);
        } else {
            ERR_OUT("deserialization failed. Skipped handling.\n");
        } 
        ABT_mutex_lock(ekth->peer_mutex);
        delete_thash_entry(ekth->thash, in.data.buf, in.data.len, in.pid,
                           in.type_id);
        ABT_mutex_unlock(ekth->peer_mutex);
    } else {
        DEBUG_OUT("received copy %i of identical type %i tell from %i, waiting "
                  "for %i, with len %" PRIu64 ".\n",
                  recv_count, in.type_id, in.pid, in.degree, in.data.len);
    }

    margo_free_input(handle, &in);
    margo_destroy(handle);
    APEX_TIMER_STOP(0);
}
DEFINE_MARGO_RPC_HANDLER(tell_rpc)

static void query_status_rpc(hg_handle_t handle)
{
    margo_instance_id mid;
    const struct hg_info *info;
    struct ekt_id *ekth;
    query_status_in_t in;
    struct ekt_peer *peer;
    uint32_t out = 0;
    uint32_t wait;
    hg_return_t hret;

    mid = margo_hg_handle_get_instance(handle);
    info = margo_get_info(handle);
    ekth = (struct ekt_id *)margo_registered_data(mid, info->id);
    hret = margo_get_input(handle, &in);
    wait = in.flag;

    DEBUG_OUT("received inquiry about peer group '%s' with wait = %i\n",
              in.name, wait);

    ABT_mutex_lock(ekth->peer_mutex);
    do {
        DEBUG_OUT("checking whether we have peer already registered...\n");
        for(peer = ekth->peers; peer; peer = peer->next) {
            DEBUG_OUT("found '%s'\n", peer->name);
            if(strcmp(in.name, peer->name) == 0) {
                DEBUG_OUT("matched!\n");
                out = 1;
                break;
            }
        }
        if(ekth->f_debug && !out) {
            DEBUG_OUT("didn't find existing mathing peer group.\n");
        }
        if(!out && wait) {
            DEBUG_OUT("wait flag is set...waiting for peer condition.\n");
            ABT_cond_wait(ekth->peer_cond, ekth->peer_mutex);
        }
    } while(!out && wait);
    ABT_mutex_unlock(ekth->peer_mutex);

    DEBUG_OUT(
        "we have the peer registered...check if we are ready for messages\n");
    ABT_mutex_lock(ekth->ready_mutex);
    if(wait) {
        while(!ekth->f_ready) {
            ABT_cond_wait(ekth->ready_cond, ekth->ready_mutex);
        }
    }
    ABT_mutex_unlock(ekth->ready_mutex);
    DEBUG_OUT("responding\n");
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(query_status_rpc)

int ekt_init(struct ekt_id **ekt_handle, const char *app_name, MPI_Comm comm,
             margo_instance_id mid)
{
    const char *envdebug = getenv("EKT_DEBUG");
    struct ekt_id *ekth;
    hg_bool_t flag;
    hg_id_t id;
    int i;

    ekth = calloc(1, sizeof(*ekth));

    for(i = 0; i < EKT_WATCH_HASH_SIZE; i++) {
        ekth->watch_cbs[i] = NULL;
    }

    ekth->app_name = strdup(app_name);

    MPI_Comm_dup(comm, &ekth->comm);
    MPI_Comm_rank(comm, &ekth->rank);
    MPI_Comm_size(comm, &ekth->app_size);

    if(envdebug) {
        ekth->f_debug = 1;
        if(ekth->rank == 0) {
            DEBUG_OUT("enabled EKT debugging.\n");
        }
    }

    ekth->mid = mid;

    margo_registered_name(mid, "hello_rpc", &id, &flag);

    if(flag == HG_TRUE) { /* Provider already registered */
        margo_registered_name(mid, "hello_rpc", &ekth->hello_id, &flag);
        margo_registered_name(mid, "query_addr_rpc", &ekth->query_addrs_id,
                              &flag);
        margo_registered_name(mid, "tell_rpc", &ekth->tell_id, &flag);
        margo_registered_name(mid, "query_status_rpc", &ekth->query_status_id,
                              &flag);
    } else {
        ekth->hello_id = MARGO_REGISTER(mid, "hello_rpc", hello_in_t,
                                        hello_out_t, hello_rpc);
        margo_register_data(mid, ekth->hello_id, (void *)ekth, NULL);
        ekth->query_addrs_id =
            MARGO_REGISTER(mid, "query_addr_rpc", query_addrs_in_t, ekt_buf_t,
                           query_addrs_rpc);
        margo_register_data(mid, ekth->query_addrs_id, (void *)ekth, NULL);
        ekth->tell_id =
            MARGO_REGISTER(mid, "tell_rpc", tell_in_t, void, tell_rpc);
        margo_register_data(mid, ekth->tell_id, (void *)ekth, NULL);
        margo_registered_disable_response(mid, ekth->tell_id, HG_TRUE);
        ekth->query_status_id =
            MARGO_REGISTER(mid, "query_status_rpc", query_status_in_t, uint32_t,
                           query_status_rpc);
        margo_register_data(mid, ekth->query_status_id, (void *)ekth, NULL);
    }

    ekth->my_addr_str = get_address(mid);
    gather_addresses(ekth);

    ekth->thash = create_thash(16, 541);

    ABT_mutex_create(&ekth->peer_mutex);
    ABT_cond_create(&ekth->peer_cond);
    ekth->peers = NULL;

    ABT_mutex_create(&ekth->ready_mutex);
    ABT_cond_create(&ekth->ready_cond);

    if(ekth->rank == 0) {
        write_bootstrap_conf(ekth);
    }

    *ekt_handle = ekth;

    return (0);
}

void ekt_enable(struct ekt_id *ekth)
{
    ABT_mutex_lock(ekth->ready_mutex);
    ekth->f_ready = 1;
    ABT_cond_broadcast(ekth->ready_cond);
    ABT_mutex_unlock(ekth->ready_mutex);
    DEBUG_OUT("ready for messages\n");
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

    return (0);
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

    return (rank / count);
}

static void map_from_collector(int collector, int size, int *start, int *end)
{
    int count = get_collector_count(size);

    *start = count * collector;
    *end = *start + (count - 1);
    if(*end >= size) {
        *end = size - 1;
    }
}

static void part_range(int len, int parts, int part, int *start, int *end)
{
    int div = len / parts;
    int rem = len % parts;

    if(part < rem) {
        *start = (div + 1) * part;
        *end = *start + div;
    } else {
        *start = ((div + 1) * rem) + (part - rem) * div;
        *end = *start + div - 1;
    }
}

static int *part_map(int len, int parts)
{
    int *map = malloc(sizeof(*map) * parts);
    int div = len / parts;
    int rem = len % parts;
    int i;

    for(i = 0; i < parts; i++) {
        map[i] = div;
        if(i < rem)
            map[i]++;
    }

    return (map);
}

// bugged
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
        ERR_OUT("margo_forward() failed with %d\n", hret);
        margo_destroy(handle);
        return (-1);
    }

    hret = margo_get_output(handle, &out);
    if(hret != HG_SUCCESS) {
        ERR_OUT("margo_get_output() failed with %d.\n", hret);
        margo_destroy(handle);
        return (-1);
    }

    *addrs_len = out.len;
    DEBUG_OUT(
        "length of addresses of peer %i through %i, asked of %s, is %" PRIu64
        " bytes.\n",
        lower, upper, tgt_addr, out.len);
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

static void add_peer(struct ekt_id *ekth, const char *name, int size,
                     int rank_start, int rank_end, char **addrs, int peer_id,
                     int degree)
{
    struct ekt_peer *peer, **peerp;
    int addr_count = (rank_end - rank_start) + 1;

    DEBUG_OUT("adding peer group '%s', with peer size %i\n", name, size);
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
        peer->size = 0;
        *peerp = peer;
    } else {
        peer = *peerp;
    }
    if(peer->size == 0) {
        // either new or a placeholder
        peer->size = size;
        peer->rank_start = rank_start;
        peer->rank_count = addr_count;
        peer->peer_addrs = addrs;
        peer->id = peer_id;
        peer->degree = degree;
    }
}

static int peer_hello(struct ekt_id *ekth, const char *boot_addr, char **addrs,
                      int *addrs_len, int *peer_id)
{
    hello_in_t hin;
    hello_out_t hout;
    int peer_size;
    hg_addr_t server_addr;
    hg_handle_t handle;
    hg_return_t hret;

    hin.name = ekth->app_name;
    hin.size = ekth->app_size;
    margo_addr_lookup(ekth->mid, boot_addr, &server_addr);

    hret = margo_create(ekth->mid, server_addr, ekth->hello_id, &handle);
    if(hret != HG_SUCCESS) {
        ERR_OUT("margo_create() failed with %d\n", hret);
        return (-1);
    }
    hret = margo_forward(handle, &hin);
    if(hret != HG_SUCCESS) {
        ERR_OUT("margo_forward() failed with %d.\n", hret);
        margo_destroy(handle);
        return (-1);
    }

    hret = margo_get_output(handle, &hout);
    if(hret != HG_SUCCESS) {
        ERR_OUT("margo_get_output() failed with %d.\n", hret);
        margo_destroy(handle);
        return (-1);
    }

    *addrs = malloc(hout.addrs.len);
    *addrs_len = hout.addrs.len;
    memcpy(*addrs, hout.addrs.buf, hout.addrs.len);
    peer_size = hout.size;
    *peer_id = hout.pid;
    margo_free_output(handle, &hout);
    margo_destroy(handle);

    return (peer_size);
}

static int get_peer_rank_info(struct ekt_id *ekth, const char *peer,
                              char ***pcoll_addrs, int *pcoll_count,
                              int *peer_id)
{
    int peer_size;
    char *boot_addr;
    char *pcoll_addrs_buf;
    int pcoll_addrs_buf_len;

    if(ekth->rank == 0) {
        // rank 0 reads bootstrapping conf
        if(read_bootstrap_conf(ekth, peer, &boot_addr) < 0) {
            ERR_OUT(" could not read peer bootstrap file.\n");
            peer_size = -1;
        } else {
            /* rank 0 contacts rank 0 of peer:
                send greeting rpc
                response is peer's collector_addrs and app size
            */
            peer_size = peer_hello(ekth, boot_addr, &pcoll_addrs_buf,
                                   &pcoll_addrs_buf_len, peer_id);
            free(boot_addr);
        }
    }
    MPI_Bcast(peer_id, 1, MPI_INT, 0, ekth->comm);
    MPI_Bcast(&peer_size, 1, MPI_INT, 0, ekth->comm);
    *pcoll_count = get_collector_count(peer_size);
    if(ekth->gather_count > 0) {
        MPI_Bcast(&pcoll_addrs_buf_len, 1, MPI_INT, 0, ekth->collector_comm);
        if(ekth->rank != 0) {
            pcoll_addrs_buf = malloc(pcoll_addrs_buf_len);
        }
        MPI_Bcast(pcoll_addrs_buf, pcoll_addrs_buf_len, MPI_BYTE, 0,
                  ekth->collector_comm);
        deserialize_str_list(pcoll_addrs_buf, *pcoll_count, pcoll_addrs);
    }

    return (peer_size);
}

static void get_peer_ranks(struct ekt_id *ekth, int peer_size, int *pstart,
                           int *pend, int *degree)
{
    int *degrees;
    int sum = 0;
    int i;

    *pstart = *pend = -1;
    if(peer_size < ekth->app_size) {
        degrees = part_map(ekth->app_size, peer_size);
        for(i = 0; i < peer_size; i++) {
            sum += degrees[i];
            if(sum > ekth->rank) {
                *pstart = *pend = i;
                break;
            }
        }
        if(*pstart == -1) {
            ERR_OUT(
                "could not find a fan-in node for rank %i with peer size %i.\n",
                ekth->rank, peer_size);
        }
        *degree = degrees[*pstart];
    } else {
        degrees = part_map(peer_size, ekth->app_size);
        *pstart = 0;
        for(i = 0; i < ekth->rank; i++) {
            *pstart += degrees[i];
        }
        *pend = *pstart + (degrees[ekth->rank] - 1);
        *degree = 1;
    }

    DEBUG_OUT("start peer is %i and end peer is %i\n", *pstart, *pend);

    free(degrees);
}

static int gather_peer_ranges(struct ekt_id *ekth, int pstart, int pend,
                              int **pstarts, int **pends)
{
    int gstart, gend;
    int i;

    if(ekth->gather_count > 0) {
        *pstarts = malloc(sizeof(*pstarts) * ekth->gather_count);
        *pends = malloc(sizeof(*pends) * ekth->gather_count);
    }

    MPI_Gather(&pstart, 1, MPI_INT, *pstarts, 1, MPI_INT, 0, ekth->gather_comm);
    MPI_Gather(&pend, 1, MPI_INT, *pends, 1, MPI_INT, 0, ekth->gather_comm);

    if(ekth->gather_count > 0) {
        gstart = (*pstarts)[0];
        gend = (*pends)[ekth->gather_count - 1];
        for(i = 0; i < ekth->gather_count; i++) {
            if((*pends)[i] < (*pstarts)[i]) {
                ERR_OUT("got a peer pair where start is %i and end is %i.\n",
                        (*pstarts)[i], (*pends)[i]);
                return (-1);
            }
            if((*pstarts)[i] != (*pends)[i + 1] && i < ekth->gather_count - 1 &&
               (*pstarts)[i + 1] != (*pends)[i] + 1) {
                ERR_OUT("peer pair has end %i, but next peer starts at %i.\n",
                        (*pends)[i], (*pstarts)[i + 1]);
                return (-1);
            }
            if((*pends)[i] > (*pends)[ekth->gather_count - 1]) {
                ERR_OUT("peer pair ends at %i, but last peer pair end at %i.\n",
                        (*pends)[i], gend);
                return (-1);
            }
            if((*pstarts)[i] < (*pstarts)[0]) {
                ERR_OUT("peer pair starts at %i, but first peer pair start at "
                        "%i.\n",
                        (*pstarts)[i], gstart);
                return (-1);
            }
        }
    }

    return (0);
}

static int query_collectors(struct ekt_id *ekth, char **pcoll_addrs,
                            int pcoll_count, int peer_size, int *pstarts,
                            int *pends, char ***peer_addrs)
{
    int gstart, gend;   // gather range
    int qstart, qend;   // query range
    int pcstart, pcend; // peer collector range
    char **query_res;
    int *query_res_len;
    char *paddrs_buf;
    char *paddrs_buf_p;
    int paddrs_buf_len = 0;
    int i;

    gstart = pstarts[0];
    gend = pends[ekth->gather_count - 1];
    DEBUG_OUT("need peer addresses for peers %i through %i\n", gstart, gend);
    pcstart = map_to_collector(gstart, peer_size);
    pcend = map_to_collector(gend, peer_size);
    DEBUG_OUT("need to query collectors %i through %i\n", pcstart, pcend);

    query_res = malloc(sizeof(*query_res) * ((pcend - pcstart) + 1));
    query_res_len = malloc(sizeof(*query_res_len) * ((pcend - pcstart) + 1));
    for(i = pcstart; i <= pcend; i++) {
        map_from_collector(i, peer_size, &qstart, &qend);
        DEBUG_OUT("peer collector %i responsible for peer %i through %i\n", i,
                  qstart, qend);
        if(qstart < gstart)
            qstart = gstart;
        if(qend > gend)
            qend = gend;
        query_addrs(ekth, pcoll_addrs[i], qstart, qend,
                    &query_res_len[i - pcstart], &query_res[i - pcstart]);
        paddrs_buf_len += query_res_len[i - pcstart];
    }
    DEBUG_OUT("total peer address buffer length is %i\n", paddrs_buf_len);

    paddrs_buf = malloc(paddrs_buf_len + 1);
    paddrs_buf_p = paddrs_buf;
    for(i = pcstart; i <= pcend; i++) {
        memcpy(paddrs_buf_p, query_res[i - pcstart],
               query_res_len[i - pcstart]);
        paddrs_buf_p += query_res_len[i - pcstart];
        free(query_res[i - pcstart]);
    }
    paddrs_buf[paddrs_buf_len] = '\0'; // for finding total length of buffer
    free(query_res_len);
    free(query_res);

    deserialize_str_list(paddrs_buf, (gend - gstart) + 2, peer_addrs);
    for(i = 0; i < (gend - gstart) + 2; i++) {
        DEBUG_OUT("(*peer_addrs)[%i] = %p\n", i, (*peer_addrs)[i]);
    }

    return (0);
}

static int distribute_peers(struct ekt_id *ekth, char **pcoll_addrs,
                            int pcoll_count, int peer_size, int *pstarts,
                            int *pends, char **peers_buf)
{
    char **gpeer_addrs;
    int len;
    int *displs, *lens;
    int gstart, gend;
    int pstart, pend;
    int i;

    if(ekth->gather_count > 0) {
        gstart = pstarts[0];
        gend = pends[ekth->gather_count - 1];
        query_collectors(ekth, pcoll_addrs, pcoll_count, peer_size, pstarts,
                         pends, &gpeer_addrs);
        displs = malloc(sizeof(*displs) * ekth->gather_count);
        lens = malloc(sizeof(*lens) * ekth->gather_count);
        for(i = 0; i < ekth->gather_count; i++) {
            displs[i] = gpeer_addrs[pstarts[i] - gstart] - gpeer_addrs[0];
            lens[i] = gpeer_addrs[(pends[i] + 1) - gstart] -
                      gpeer_addrs[pstarts[i] - gstart];
            DEBUG_OUT("for rank %i, sending displacement %i and length %i\n",
                      ekth->rank + i, displs[i], lens[i]);
        }
    }

    MPI_Scatter(lens, 1, MPI_INT, &len, 1, MPI_INT, 0, ekth->gather_comm);
    DEBUG_OUT("my length is %i\n", len);
    *peers_buf = malloc(len);
    MPI_Scatterv(ekth->gather_count ? gpeer_addrs[0] : NULL, lens, displs,
                 MPI_BYTE, *peers_buf, len, MPI_BYTE, 0, ekth->gather_comm);

    if(ekth->gather_count > 0) {
        free(gpeer_addrs[0]);
        free(displs);
        free(lens);
    }

    return (0);
}

int ekt_connect(struct ekt_id *ekth, const char *peer)
{
    int peer_size;
    int pstart, pend;
    char **pcoll_addrs;
    int pcoll_count;
    int *pstarts, *pends;
    char *peers_buf;
    char **peer_addrs;
    int degree;
    int peer_id;

    // pcoll_addrs and pcoll_count is valid on collectors after this call
    peer_size =
        get_peer_rank_info(ekth, peer, &pcoll_addrs, &pcoll_count, &peer_id);
    if(peer_size == -1) {
        return (-1);
    }

    DEBUG_OUT("%s has %i ranks, I am peer_id %i\n", peer, peer_size, peer_id);

    get_peer_ranks(ekth, peer_size, &pstart, &pend, &degree);
    // pstarts and pends is valid on colelctors after this call
    if(gather_peer_ranges(ekth, pstart, pend, &pstarts, &pends) < 0) {
        return (-1);
    }
    if(distribute_peers(ekth, pcoll_addrs, pcoll_count, peer_size, pstarts,
                        pends, &peers_buf) < 0) {
        return (-1);
    }
    deserialize_str_list(peers_buf, (pend - pstart) + 1, &peer_addrs);

    if(ekth->gather_count > 0) {
        free(pcoll_addrs[0]);
        free(pstarts);
        free(pends);
    }

    // install peer
    ABT_mutex_lock(ekth->peer_mutex);
    DEBUG_OUT("installing peer\n");
    add_peer(ekth, peer, peer_size, pstart, pend, peer_addrs, peer_id, degree);
    DEBUG_OUT("waiting for everyone else\n");
    MPI_Barrier(ekth->comm);
    ABT_cond_broadcast(ekth->peer_cond);
    ABT_mutex_unlock(ekth->peer_mutex);

    DEBUG_OUT("finished\n");

    return (0);
}

int ekt_watch(struct ekt_id *ekth, struct ekt_type *type, watch_fn cb)
{
    int type_id = type->type_id;
    int type_hash = type_id % EKT_WATCH_HASH_SIZE;
    struct watch_list_node **cb_node = &ekth->watch_cbs[type_hash];

    while(*cb_node) {
        if((*cb_node)->type_id) {
            WARN_OUT("already watching type_id %i.", type_id);
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

int ekt_handle_announce(struct ekt_id *ekth, int type_id, void *data)
{
    int type_hash = type_id % EKT_WATCH_HASH_SIZE;
    struct watch_list_node *cb_node = ekth->watch_cbs[type_hash];

    while(cb_node) {
        if(cb_node->type_id == type_id) {
            break;
        }
        cb_node = cb_node->next;
    }

    if(!cb_node) {
        WARN_OUT("received announcement with unknown type %d.\n", type_id);
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

    DEBUG_OUT("telling peer '%s'\n", peer->name);
    if(!type) {
        ERR_OUT(" bad type passed to ekt_tell.\n");
        return (-1);
    }
    in.type_id = type->type_id;
    in.pid = peer->id;
    in.degree = peer->degree;
    in.data.len = data_size;
    in.data.buf = data;

    DEBUG_OUT("telling %zu ranks of '%s'\n", peer->rank_count, peer->name);
    for(i = 0; i < peer->rank_count; i++) {
        hg_handle_t handle;
        margo_request req;

        margo_addr_lookup(ekth->mid, peer->peer_addrs[i], &peer_addr);
        hret = margo_create(ekth->mid, peer_addr, ekth->tell_id, &handle);
        if(hret != HG_SUCCESS) {
            ERR_OUT("failed to create hello rpc (%d)\n", hret);
            return (-1);
        }
        in.dst = peer->rank_start + i;
        DEBUG_OUT(
            "sending %s (%i) tell rpc of type %i with degree %i to %zi.\n",
            peer->peer_addrs[i], in.dst, in.type_id, in.degree,
            peer->rank_start + i);
        hret = margo_iforward(handle, &in, &req);
        if(hret != HG_SUCCESS) {
            ERR_OUT("failed to forward notification (%d)\n", hret);
            margo_destroy(handle);
            return (-1);
        }
        margo_destroy(handle);
        DEBUG_OUT("told rank %zi of '%s'\n", peer->rank_start + i, peer->name);
    }

    return (0);
}

int ekt_tell(struct ekt_id *ekth, const char *target, struct ekt_type *type,
             void *data)
{
    struct ekt_peer *peer;
    int data_size;
    void *ser_data = NULL;

    DEBUG_OUT("telling type %i\n", type->type_id);

    // transmit to all other ekt clients
    peer = ekth->peers;
    if(peer) {
        data_size = type->ser(data, type->arg, &ser_data);
        if(data_size < 0) {
            ERR_OUT("serialization failed.\n");
            return(-1);        
        }
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

    return (0);
}

int ekt_is_bidi(struct ekt_id *ekth, const char *name, int wait)
{
    struct ekt_peer *peer;
    hg_addr_t peer_addr;
    hg_handle_t handle;
    query_status_in_t in;
    uint32_t out;
    hg_return_t hret;

    DEBUG_OUT("Asking '%s' if they are connected to '%s'\n", name,
              ekth->app_name);

    ABT_mutex_lock(ekth->peer_mutex);
    for(peer = ekth->peers; peer; peer = peer->next) {
        if(strcmp(peer->name, name) == 0) {
            margo_addr_lookup(ekth->mid, peer->peer_addrs[0], &peer_addr);
            ABT_mutex_unlock(ekth->peer_mutex);
            margo_create(ekth->mid, peer_addr, ekth->query_status_id, &handle);
            in.name = ekth->app_name;
            in.flag = wait;
            margo_forward(handle, &in);
            margo_get_output(handle, &out);
            margo_destroy(handle);
            return (out);
        }
    }
    ABT_mutex_unlock(ekth->peer_mutex);
    ERR_OUT("'%s' is not a connected peer.\n", name);
    return (0);
}

int ekt_peer_size(struct ekt_id *ekth, const char *name)
{
    struct ekt_peer *peer;
    int size = -1;

    ABT_mutex_lock(ekth->peer_mutex);
    for(peer = ekth->peers; peer; peer = peer->next) {
        if(strcmp(peer->name, name) == 0) {
            size = peer->size;
            break;
        }
    }
    ABT_mutex_unlock(ekth->peer_mutex);

    return (size);
}
