typedef int (*serdes_fn)(void *, void *, void **);
typedef int (*watch_fn)(void *, void *);

struct watch_list_node {
    int type_id;
    watch_fn cb;
    struct watch_list_node *next;
};

#define EKT_WATCH_HASH_SIZE 16
struct ekt_id {
    struct watch_list_node *watch_cbs[EKT_WATCH_HASH_SIZE];
    char *app_name;
    MPI_Comm comm;
    int rank;
    int app_size;
    margo_instance_id mid;
    char *my_addr_str;
    char **gather_addrs;
    int gather_count;
    int collector_count;
    char *collector_addrs;
    int collector_addrs_len;
    
    //margo rpc handles
    hg_id_t hello_id;
};

struct ekt_type {
    serdes_fn ser;
    serdes_fn des;
    void *arg;
    int type_id;
};

struct ekt_peer {
    char *app_name;
    size_t peer_size;
    size_t peer_rank_count;
    int *peer_ranks;
};
