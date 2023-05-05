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
    // Initialize MPI, global variables and input sequence.
    MPI_Init(&argc, &argv);
    initGlobals();
    initInputValues("../res/input.txt");

    float seq_real[global::input_size]{};
    float seq_img[global::input_size]{};

    // Split input into real and imaginary sequences.
    // Apply bit reversing for butterfly indexing algorithm.
    const int max_bit_width = std::log2f(global::input_size);
    for(int i = 1; i < global::input_size; i++)
    {
        seq_real[i] = global::input[reverseBits(i - 1, max_bit_width) + 1];
        seq_img[i] = 0.0;
    }

    // Broadcast sequence of real values from master to all workers. Input is accepted only in real
    // domain hence there's no need to broadcast imaginary sequence.
    logger::master("broadcast initial sequence\n");
    MPI_Bcast(seq_real, global::input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    const int values_per_process = global::input_size / global::slave_count;

    // Begin FFT computation with butterfly indexing applied.
    const double starttime = MPI_Wtime();
    for(int div = 1, key = std::log2f(global::input_size - 1); key > 0; key--, div *= 2)
    {
        logger::master("ITERATION %d\n", static_cast<int>(std::log2f(div)));
        if(global::rank != 0)
        {
            using global::rank;

            // Compute more than one FFT node on each worker if (number of nodes > number of
            // workers). If there are less nodes than workers, the behaviour is undefined.
            logger::slave("beginning compute...\n");
            for(int b = 0; b < values_per_process; b++)
            {
                // Prepare butterfly indexing.
                const auto b_rank = (rank - 1) * values_per_process + b + 1;
                const auto is_even = ((b_rank + div - 1) / div) % 2;
                const auto is_odd = 1 - is_even;
                const auto butterfly_index = M_PI * ((b_rank - 1) % (div * 2)) / div;

                // Calculate FFT sequence on each worker in parallel.
                float temp_real =
                    seq_real[b_rank - (div * is_odd)]
                    + (std::cos(butterfly_index) * (seq_real[b_rank + (div * is_even)]))
                    + (std::sin(butterfly_index) * (seq_img[b_rank + (div * is_even)]));

                float temp_img =
                    seq_img[b_rank - (div * is_odd)]
                    + (std::cos(butterfly_index) * (seq_img[b_rank + (div * is_even)]))
                    - (std::sin(butterfly_index) * (seq_real[b_rank + (div * is_even)]));

                logger::slave(
                    " ++ sending to master: b_rank = %d, tag_real = %d, tag_img = %d, seq[b_rank] = {%f, %f}\n",
                    b_rank,
                    0 + values_per_process * b,
                    1 + values_per_process * b,
                    temp_real,
                    temp_img);

                // Send current just calculated value from each worker to master process.
                MPI_Send(&temp_real, 1, MPI_FLOAT, 0, 0 + values_per_process * b, MPI_COMM_WORLD);
                MPI_Send(&temp_img, 1, MPI_FLOAT, 0, 1 + values_per_process * b, MPI_COMM_WORLD);
            }

            logger::slave("ending compute...\n");
        }
        else
        {
            // Receive on master sequence elements sent from all workers. The number of receive
            // operations must match with send operations and comply with butterfly indexing.
            logger::master("beginning receiving sequence... (count = %d)\n",
                global::input_size - 1);

            // For each worker process.
            for(int i = 0; i < global::slave_count; i++)
            {
                const auto slave_id = i + 1;

                // For each value computed by each worker.
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

        // Wait for all workers to send their computations to master and for master to receive them.
        MPI_Barrier(MPI_COMM_WORLD);

        // Broadcast newly calculated real and imaginary sequences from master to all other workers.
        logger::master("broadcast final sequence\n");
        MPI_Bcast(seq_real, global::input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(seq_img, global::input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // Print computed FFT results.
    const double endtime = MPI_Wtime();
    showResults(seq_real, seq_img, starttime, endtime);

    // Clean up MPI internal state and exit the program.
    MPI_Finalize();
    return 0;
}

////////////////

/**
 * @brief Initialize global variables set by MPI.
 *
 */
void initGlobals()
{
    global::process_count = getProcessCount();
    global::slave_count = global::process_count - 1;
    global::rank = getProcessRank();
}

/**
 * @brief Reverse bits to reverse recursive butterfly indexing algorithm.
 *
 * @param number Index of a given butterfly node of type int32.
 * @param bit_range Range of least significant bits to reverse in the number parameter.
 * @return int Index of a node with reversed bit_range least significant bits.
 */
int reverseBits(int number, int bit_range)
{
    int reverse_number = 0;
    for(int i = 0; i < bit_range; i++)
    {
        reverse_number |= ((number >> i) & 1) << (bit_range - 1 - i);
    }
    return reverse_number;
}

/**
 * @brief Read the process count from MPI.
 *
 * @return int Number of parallely running processes.
 */
int getProcessCount()
{
    int process_count{};
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    return process_count;
}

/**
 * @brief Obtain current process rank. Master receives rank 0, meanwhile workers acquire rank > 0.
 *
 * Ranks are unique, meaning that no two processes can be labeled with the same rank.
 *
 * @return int Rank of the current process.
 */
int getProcessRank()
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/**
 * @brief Initialize all processes with input values read by master.
 *
 * Flow:
 * 1) Read from file on master process.
 * 2) Broadcast the size of input to all workers.
 * 3) Declare array of obtained size on each process.
 * 4) Broadcast input sequence from master to all workers.
 *
 * @param path Directory path to a text file with an array of input values.
 */
void initInputValues(const char* path)
{
    // Read from input file on master.
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

    // Broadcast input size to workers so that they know how large the array has to be.
    global::input_size = global::input.size();
    MPI_Bcast(&global::input_size, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Declare temporary C array to store the input values in.
    float temp[global::input_size] = {0};

    // Fill the temporary array on master process.
    if(global::rank == 0)
    {
        for(int i = 0; i < global::input_size; i++)
        {
            temp[i] = global::input[i];
        }
    }

    // Broadcast temporary array from master to all workers.
    MPI_Bcast(temp, global::input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // On each process, fill global input std::vector with temporary array.
    if(global::rank != 0)
    {
        for(int i = 0; i < global::input_size; i++)
        {
            global::input.push_back(temp[i]);
        }
    }
}

/**
 * @brief Print results of computation.
 *
 * @param seq_real Sequence of results in real domain.
 * @param seq_img Sequence of results in imaginary domain.
 * @param starttime Timestamp of algorithm before computation.
 * @param endtime Timestamp of algorithm after computation has finished.
 */
void showResults(const float* seq_real, const float* seq_img, double starttime, double endtime)
{
    // Printing only on master process.
    if(0 == global::rank)
    {
        std::printf("\n");
        for(int i = 1; i < global::input_size; i++)
        {
            const char img_sign = seq_img[i] >= 0 ? '+' : '-';
            std::printf("X[%3d] = %6.2f %c i%-6.2f\n",
                i,
                seq_real[i],
                img_sign,
                std::fabs(seq_img[i]));
        }

        std::printf("\nParallel FFT computation time: %.4lf ms\n\n", (endtime - starttime) * 1000);
    }
}

namespace logger
{

/**
 * @brief Print logging message on every process.
 *
 * @param format Format string provided to std::printf().
 * @param args Formatting parameters provided to std::printf().
 */
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

/**
 * @brief Print logging message only on master process.
 *
 * @param format Format string provided to std::printf().
 * @param args Formatting parameters provided to std::printf().
 */
void master(const char* format, auto... args)
{
    if(global::rank == 0)
    {
        logger::all(format, args...);
    }
}

/**
 * @brief Print logging message only on worker processes (effectively excluding master).
 *
 * @param format Format string provided to std::printf().
 * @param args Formatting parameters provided to std::printf().
 */
void slave(const char* format, auto... args)
{
    if(global::rank != 0)
    {
        ::logger::all(format, args...);
    }
}

}  // namespace logger
