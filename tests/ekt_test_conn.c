#include <ekt.h>

#include <margo.h>

#include <mpi.h>

struct my_data {
    uint32_t id;
    int64_t value;
};

struct wait_file {
    struct my_data check_data;
    ABT_mutex wait_mtx;
    ABT_cond wait_cond;
    int done;
};

static int my_type_check(void *my_datav, void *wfilev)
{
    struct wait_file *wait = (struct wait_file *)wfilev;
    struct my_data *check_data  = &wait->check_data;
    struct my_data *my_data = my_datav;
    int ret = 0;

    if(my_data->id != check_data->id || my_data->value != check_data->value) {
        ret = -1;
        printf("WARNING: check_data did not match!\n");
    }

    ABT_mutex_lock(wait->wait_mtx);
    wait->done = 1;
    ABT_cond_signal(wait->wait_cond);
    ABT_mutex_unlock(wait->wait_mtx);

    return (ret);
}

static int serialize_my_data(void *my_datav, void *checkv, void **buf)
{
    struct my_data *my_data = my_datav;

    *buf = malloc(sizeof(my_data->id) + sizeof(my_data->value));
    memcpy(*buf, &my_data->id, sizeof(my_data->id));
    memcpy(&((char *)(*buf))[sizeof(my_data->id)], &my_data->value,
           sizeof(my_data->value));

    return (0);
}

static int deserialize_my_data(void *buf, void *checkv, void **my_datav)
{
    struct my_data *my_data = malloc(sizeof(*my_data));

    memcpy(&my_data->id, buf, sizeof(my_data->id));
    memcpy(&my_data->value, &((char *)buf)[sizeof(my_data->id)],
           sizeof(my_data->value));

    *my_datav = my_data;
}

#define EKT_MY_TYPE 0

int main(int argc, char **argv)
{
    ekt_id ekt_h;
    ekt_type my_type;
    margo_instance_id mid;
    struct my_data check_data = {.id = 4, .value = 2};
    char *app_name;
    char *peer_name = NULL;
    struct wait_file wait = {0};

    if(argc < 2 || argc > 3) {
        printf("Usage: ekt_test_conn <app_name> <peer_name>\n");
        return(-1);
    } 

    app_name = argv[1];
    if(argc > 2) {
        peer_name = argv[2];
    } 

    ABT_mutex_create(&wait.wait_mtx);
    ABT_cond_create(&wait.wait_cond);
    wait.done = 0;
    wait.check_data = check_data;

    MPI_Init(NULL, NULL);
    mid = margo_init("sm", MARGO_SERVER_MODE, 1, 1);

    ekt_init(&ekt_h, "me", MPI_COMM_WORLD, mid);
    ekt_register(EKT_MY_TYPE, serialize_my_data, deserialize_my_data,
                 &wait, &my_type);
    ekt_watch(ekt_h, my_type, my_type_check);
    if(peer_name) {
        ekt_connect(ekt_h, peer_name);
        ekt_tell(ekt_h, my_type, &check_data);
    }

    while(wait.done == 0) {
        ABT_mutex_lock(wait.wait_mtx);
        ABT_cond_wait(wait.wait_cond, wait.wait_mtx);
        ABT_mutex_unlock(wait.wait_mtx);
    }

    ekt_deregister(&my_type);
    ekt_fini(&ekt_h);

    margo_finalize(mid);
    MPI_Finalize();

    return (0);
}
