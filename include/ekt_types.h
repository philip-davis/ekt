#include <mercury.h>
#include <mercury_macros.h>
#include <mercury_proc_string.h>

typedef int (*serdes_fn)(void *, void *, void **);
typedef int (*watch_fn)(void *, void *);

struct ekt_type {
    serdes_fn ser;
    serdes_fn des;
    void *arg;
    int type_id;
};

struct watch_list_node {
    int type_id;
    watch_fn cb;
    struct ekt_type *type;
    struct watch_list_node *next;
};

struct ekt_peer {
    struct ekt_peer *next;
    char *name;
    size_t size;
    size_t rank_count;
    char **peer_addrs;
};

#define EKT_WATCH_HASH_SIZE 16
struct ekt_id {
    struct watch_list_node *watch_cbs[EKT_WATCH_HASH_SIZE];
    char *app_name;
    MPI_Comm comm;
    MPI_Comm gather_comm;
    MPI_Comm collector_comm;
    int rank;
    int app_size;
    margo_instance_id mid;
    char *my_addr_str;
    char **gather_addrs;
    int gather_addrs_len;
    int gather_count;
    int collector_count;
    char *collector_addrs;
    int collector_addrs_len;

    struct ekt_peer *peers;
    ABT_mutex peer_mutex;

    // margo rpc handles
    hg_id_t hello_id;
    hg_id_t query_addrs_id;
    hg_id_t tell_id;
};

typedef struct ekt_buf {
    hg_size_t len;
    void *buf;
} ekt_buf_t;

static inline hg_return_t hg_proc_ekt_buf_t(hg_proc_t proc, void *data)
{
    hg_return_t ret;
    ekt_buf_t *buf = (ekt_buf_t *)data;

    switch(hg_proc_get_op(proc)) {
    case HG_ENCODE:
        ret = hg_proc_hg_size_t(proc, &buf->len);
        if(ret != HG_SUCCESS) {
            break;
        }
        ret = hg_proc_raw(proc, buf->buf, buf->len);
        if(ret != HG_SUCCESS) {
            break;
        }
        break;
    case HG_DECODE:
        ret = hg_proc_hg_size_t(proc, &buf->len);
        if(ret != HG_SUCCESS) {
            break;
        }
        buf->buf = malloc(buf->len);
        ret = hg_proc_raw(proc, buf->buf, buf->len);
        if(ret != HG_SUCCESS) {
            break;
        }
        break;
    case HG_FREE:
        free(buf->buf);
        ret = HG_SUCCESS;
    }

    return ret;
}

MERCURY_GEN_PROC(hello_in_t, ((hg_string_t)(name))((uint32_t)(size)));
MERCURY_GEN_PROC(hello_out_t, ((uint32_t)(size))((ekt_buf_t)(addrs)));
MERCURY_GEN_PROC(query_addrs_in_t, ((uint32_t)(start))((uint32_t)(end)));
MERCURY_GEN_PROC(tell_in_t, ((uint32_t)(type_id))((ekt_buf_t)(data)));
