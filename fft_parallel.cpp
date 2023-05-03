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
int slave_count{};
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

    const int values_per_process = global::input_size / global::slave_count;

    const double starttime = MPI_Wtime();
    for(int div = 1, key = std::log2f(global::input_size - 1); key > 0; key--, div *= 2)
    {
        logger::master("ITERATION %d\n", static_cast<int>(std::log2f(div)));
        if(global::rank != 0)
        {
            using global::rank;

            logger::slave("beginning compute...\n");
            for(int b = 0; b < values_per_process; b++)
            {
                const auto b_rank = (rank - 1) * values_per_process + b + 1;
                const auto is_even = ((b_rank + div - 1) / div) % 2;
                const auto is_odd = 1 - is_even;
                const auto butterfly_index = M_PI * ((b_rank - 1) % (div * 2)) / div;

                seq_real[b_rank] =
                    seq_real[b_rank - (div * is_odd)]
                    + (std::cos(butterfly_index) * (seq_real[b_rank + (div * is_even)]))
                    + (std::sin(butterfly_index) * (seq_img[b_rank + (div * is_even)]));

                seq_img[b_rank] =
                    seq_img[b_rank - (div * is_odd)]
                    + (std::cos(butterfly_index) * (seq_img[b_rank + (div * is_even)]))
                    - (std::sin(butterfly_index) * (seq_real[b_rank + (div * is_even)]));

                logger::slave(
                    " ++ sending to master: b_rank = %d, tag_real = %d, tag_img = %d, seq[b_rank] = {%f, %f}\n",
                    b_rank,
                    0 + values_per_process * b,
                    1 + values_per_process * b,
                    seq_real[b_rank],
                    seq_img[b_rank]);

                MPI_Send(&seq_real[b_rank],
                    1,
                    MPI_FLOAT,
                    0,
                    0 + values_per_process * b,
                    MPI_COMM_WORLD);
                MPI_Send(&seq_img[b_rank],
                    1,
                    MPI_FLOAT,
                    0,
                    1 + values_per_process * b,
                    MPI_COMM_WORLD);
            }

            logger::slave("ending compute...\n");
        }
        else
        {
            logger::master("beginning receiving sequence... (count = %d)\n",
                global::input_size - 1);
            for(int i = 0; i < global::slave_count; i++)
            {
                const auto slave_id = i + 1;
                for(int b = 0; b < values_per_process; b++)
                {
                    const auto b_i = (i * values_per_process) + b + 1;
                    MPI_Recv(&seq_real[b_i],
                        1,
                        MPI_FLOAT,
                        slave_id,
                        0 + values_per_process * b,
                        MPI_COMM_WORLD,
                        &global::mpi_status);
                    MPI_Recv(&seq_img[b_i],
                        1,
                        MPI_FLOAT,
                        slave_id,
                        1 + values_per_process * b,
                        MPI_COMM_WORLD,
                        &global::mpi_status);
                    logger::master(
                        " -- received from slave %d: b_i = %d, tag_real = %d, tag_img = %d, status = %d, seq[b_i] = {%f, %f}\n",
                        slave_id,
                        b_i,
                        0 + values_per_process * b,
                        1 + values_per_process * b,
                        global::mpi_status,
                        seq_real[b_i],
                        seq_img[b_i]);
                }
            }
            logger::master("finished receiving sequence\n");
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
    global::slave_count = global::process_count - 1;
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
