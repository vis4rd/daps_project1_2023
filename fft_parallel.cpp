#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include <mpi.h>

#define ND [[nodiscard]]

namespace global
{
MPI_Status mpi_status;
int process_count{};
int rank{};

std::vector<float> input;
int input_size{};
}  // namespace global

namespace logger
{
void all(const char*, auto...);
void master(const char*, auto...);
void slave(const char*, auto...);
}  // namespace logger

void initGlobals();
ND int reverseBits(int, int);
ND int getProcessCount();
ND int getProcessRank();
void initInputValues(const char*);
void showResults(const float*, const float*, double, double);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    initGlobals();
    initInputValues("../res/input.txt");

    float seq_real[global::input_size]{};
    float seq_img[global::input_size]{};

    const int max_bit_width = std::log2f(global::input_size);
    for(int i = 1; i < global::input_size; i++)
    {
        seq_real[i] = global::input[reverseBits(i - 1, max_bit_width) + 1];
        seq_img[i] = 0.0;
    }

    logger::master("broadcast initial sequence\n");
    MPI_Bcast(seq_real, global::input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    const double starttime = MPI_Wtime();
    for(int div = 1, key = std::log2f(global::input_size - 1); key > 0; key--, div *= 2)
    {
        if(global::rank != 0)
        {
            using global::rank;

            logger::slave("beginning compute...\n");
            const auto is_even = ((rank + div - 1) / div) % 2;
            const auto is_odd = 1 - is_even;
            const auto butterfly_index = M_PI * ((rank - 1) % (div * 2)) / div;

            seq_real[rank] = seq_real[rank - (div * is_odd)]
                             + (std::cos(butterfly_index) * (seq_real[rank + (div * is_even)]))
                             + (std::sin(butterfly_index) * (seq_img[rank + (div * is_even)]));

            seq_img[rank] = seq_img[rank - (div * is_odd)]
                            + (std::cos(butterfly_index) * (seq_img[rank + (div * is_even)]))
                            - (std::sin(butterfly_index) * (seq_real[rank + (div * is_even)]));

            MPI_Send(&seq_real[rank], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&seq_img[rank], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            logger::slave("ending compute...\n");
        }
        else
        {
            logger::master("beginning receiving temps... (count = %d)\n", global::input_size);
            for(int i = 1; i < global::input_size; i++)
            {
                MPI_Recv(&seq_real[i], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &global::mpi_status);
                MPI_Recv(&seq_img[i], 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &global::mpi_status);
                logger::master(" -- received iteration (i = %d, status = %d, temp[i] = {%f, %f})\n",
                    i,
                    global::mpi_status,
                    seq_real[i],
                    seq_img[i]);
            }
            logger::master("finished receiving temps\n");
        }

        MPI_Barrier(MPI_COMM_WORLD);

        logger::master("broadcast final sequence\n");
        MPI_Bcast(seq_real, global::input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(seq_img, global::input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    const double endtime = MPI_Wtime();
    showResults(seq_real, seq_img, starttime, endtime);

    MPI_Finalize();
    return 0;
}

////////////////

void initGlobals()
{
    global::process_count = getProcessCount();
    global::rank = getProcessRank();
}

int reverseBits(int number, int bit_range)
{
    int reverse_number = 0;
    for(int i = 0; i < bit_range; i++)
    {
        reverse_number |= ((number >> i) & 1) << (bit_range - 1 - i);
    }
    return reverse_number;
}

int getProcessCount()
{
    int process_count{};
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    return process_count;
}

int getProcessRank()
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

MPI_Datatype registerMpiDatatype(int num_of_values,
    const std::vector<int>& sizes_of_values,
    const std::vector<MPI_Aint>& strides_of_values,
    const std::vector<MPI_Datatype>& types_of_values)
{
    MPI_Datatype new_type;
    MPI_Type_create_struct(num_of_values,
        sizes_of_values.data(),
        strides_of_values.data(),
        types_of_values.data(),
        &new_type);
    MPI_Type_commit(&new_type);
    return new_type;
}

void initInputValues(const char* path)
{
    if(global::rank == 0)
    {
        std::ifstream file(path);

        if(not file.is_open())
        {
            return;
        }

        global::input.push_back(0);

        float real{};
        while(file >> real)
        {
            global::input.push_back(real);
        }
    }
    global::input_size = global::input.size();
    MPI_Bcast(&global::input_size, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float temp[global::input_size] = {0};

    if(global::rank == 0)
    {
        for(int i = 0; i < global::input_size; i++)
        {
            temp[i] = global::input[i];
        }
    }

    MPI_Bcast(temp, global::input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(global::rank != 0)
    {
        for(int i = 0; i < global::input_size; i++)
        {
            global::input.push_back(temp[i]);
        }
    }
}

void showResults(const float* seq_real, const float* seq_img, double starttime, double endtime)
{
    if(0 == global::rank)
    {
        std::printf("\n");

        for(int i = 1; i < global::input_size; i++)
        {
            std::printf("X[%d] : %+6.2f", i - 1, seq_real[i]);

            if(seq_img[i] >= 0)
            {
                std::printf(" + i%-6.2f\n", seq_img[i]);
            }
            else
            {
                std::printf(" - i%-6.2f\n", seq_img[i] - 2 * seq_img[i]);
            }
        }

        std::printf("\n");
        std::printf("Total Time : %lf ms\n", (endtime - starttime) * 1000);
        std::printf("\n");
    }
}

namespace logger
{
void master(const char* format, auto... args)
{
    if(global::rank == 0)
    {
        logger::all(format, args...);
    }
}

void slave(const char* format, auto... args)
{
    if(global::rank != 0)
    {
        ::logger::all(format, args...);
    }
}

void all(const char* format, auto... args)
{
#ifdef ENABLE_LOGGING
    const auto rank_idx = not not global::rank;

    std::array<std::string, 2> thread_name = {"master",
        "slave(" + std::to_string(global::rank) + ")"};

    std::stringstream strstr;
    strstr << "LOG | " << thread_name[rank_idx] << " | " << format;
    std::printf(strstr.str().c_str(), args...);
#endif
}
}  // namespace logger
