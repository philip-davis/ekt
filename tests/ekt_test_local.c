#include <ekt.h>

#include <margo.h>

#include <mpi.h>

struct my_data {
    uint32_t id;
    int64_t value;
};

static int my_type_check(void *my_datav, void *checkv)
{
    struct my_data *check_data = checkv;
    struct my_data *my_data = my_datav;

    if(my_data->id != check_data->id || my_data->value != check_data->value) {
        return (-1);
    }

    return (0);
}

static int serialize_my_data(void *my_datav, void *checkv, void **buf)
{
    struct my_data *my_data = my_datav;
    int data_size = sizeof(my_data->id) + sizeof(my_data->value);

    *buf = malloc(data_size);
    memcpy(*buf, &my_data->id, sizeof(my_data->id));
    memcpy(&((char *)(*buf))[sizeof(my_data->id)], &my_data->value,
           sizeof(my_data->value));

    return (data_size);
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

int main()
{
    ekt_id ekt_h;
    ekt_type my_type;
    margo_instance_id mid;
    struct my_data check_data = {.id = 4, .value = 2};

    MPI_Init(NULL, NULL);
    mid = margo_init("sm", MARGO_SERVER_MODE, 1, 1);

    ekt_init(&ekt_h, "me", MPI_COMM_WORLD, mid);
    ekt_register(ekt_h, EKT_MY_TYPE, serialize_my_data, deserialize_my_data,
                 &check_data, &my_type);
    ekt_watch(ekt_h, my_type, my_type_check);
    ekt_tell(ekt_h, NULL, my_type, &check_data);
    ekt_deregister(&my_type);
    ekt_fini(&ekt_h);

    margo_finalize(mid);
    MPI_Finalize();

    return (0);
}
