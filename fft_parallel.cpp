#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>

#include <mpi.h>

#define ND [[nodiscard]]

struct Complex
{
    float real{};
    float img{};
};

ND int reverseBits(int, int);
ND int getProcessCount();
ND int getProcessRank();
void log(int, const char*, auto...);

int main(int argc, char** argv)
{
    MPI_Status status;

    MPI_Init(&argc, &argv);

    const int process_count = getProcessCount();
    const int rank = getProcessRank();

    const int input_values_count = [&rank]() -> int {
        int input_values_count{};
        if(rank == 0)
        {
            std::printf("Enter No. of input values(it should be in form of 2^x) : ");
            std::scanf("%d", &input_values_count);
        }
        MPI_Bcast(&input_values_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return input_values_count;
    }();

    if(rank == 0)
    {
        log(rank,
            "process_count = %d, rank = %d, input_count = %d\n",
            process_count,
            rank,
            input_values_count);
    }

    std::vector<Complex> input_values(input_values_count + 1);
    Complex seq[input_values_count + 1]{};
    Complex temp[input_values_count + 1]{};

    MPI_Datatype MPI_COMPLEX_T;
    MPI_Type_create_struct(sizeof(Complex) / sizeof(float),
        (const int[2]){sizeof(float), sizeof(float)},
        (const MPI_Aint[2]){0, 4},
        (const MPI_Datatype[2]){MPI_FLOAT, MPI_FLOAT},
        &MPI_COMPLEX_T);
    MPI_Type_commit(&MPI_COMPLEX_T);

    if(rank == 0)
    {
        std::printf("Enter total %d values in floating point format(separated with space) : ",
            input_values_count);

        for(int i = 1; i < input_values_count + 1; i++)
        {
            std::scanf("%f", &input_values[i].real);
            input_values[i].img = 0.0;
        }

        const int max_bit_width = std::log2f(input_values_count);
        for(int i = 1; i < input_values_count + 1; i++)
        {
            seq[i].real = input_values[reverseBits(i - 1, max_bit_width) + 1].real;
            seq[i].img = 0.0;
        }
    }

    const double starttime = MPI_Wtime();
    log(rank, "broadcast initial sequence\n");
    MPI_Bcast(seq, input_values_count + 1, MPI_COMPLEX_T, 0, MPI_COMM_WORLD);

    int iter = 1;
    for(int div = 1, key = std::log2f(input_values_count); key > 0; key--)
    {
        log(rank, "iteration(%d): div = %d, key = %d\n", iter++, div, key);
        if(rank != 0)
        {
            log(rank, "beginning compute...\n");
            if(((rank + div - 1) / div) % 2 == 0)
            {
                temp[rank].real =
                    seq[rank - div].real
                    + (std::cos(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank].real))
                    + (std::sin(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank].img));

                temp[rank].img =
                    seq[rank - div].img
                    + (std::cos(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank].img))
                    - (std::sin(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank].real));

                MPI_Send(&temp[rank], 1, MPI_COMPLEX_T, 0, 0, MPI_COMM_WORLD);
            }
            else
            {
                temp[rank].real =
                    seq[rank].real
                    + (std::cos(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank + div].real))
                    + (std::sin(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank + div].img));

                temp[rank].img =
                    seq[rank].img
                    + (std::cos(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank + div].img))
                    - (std::sin(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank + div].real));

                MPI_Send(&temp[rank], 1, MPI_COMPLEX_T, 0, 0, MPI_COMM_WORLD);
            }
            log(rank, "ending compute...\n");
        }
        else
        {
            log(rank, "beginning receiving temps...\n");
            for(int i = 1; i < input_values_count + 1; i++)
            {
                MPI_Recv(&temp[i], 1, MPI_COMPLEX_T, i, 0, MPI_COMM_WORLD, &status);
            }
            log(rank, "finished receiving temps\n");
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            for(int i = 1; i < input_values_count + 1; i++)
            {
                seq[i].real = temp[i].real;
                seq[i].img = temp[i].img;
            }
            div *= 2;
        }

        log(rank, "broadcast final sequence\n");
        MPI_Bcast(seq, input_values_count + 1, MPI_COMPLEX_T, 0, MPI_COMM_WORLD);
        log(rank, "broadcast final div\n");
        MPI_Bcast(&div, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    const double endtime = MPI_Wtime();

    if(0 == rank)
    {
        std::printf("\n");

        for(int i = 1; i < input_values_count + 1; i++)
        {
            std::printf("X[%d] : %f", i - 1, seq[i].real);

            if(seq[i].img >= 0)
            {
                std::printf("+j%f\n", seq[i].img);
            }
            else
            {
                std::printf("-j%f\n", seq[i].img - 2 * seq[i].img);
            }
        }

        std::printf("\n");
        std::printf("Total Time : %lf ms\n", (endtime - starttime) * 1000);
        std::printf("\n");
    }

    MPI_Finalize();
    return 0;
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
    int rank{};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

void log(int rank, const char* format, auto... args)
{
#ifdef ENABLE_LOGGING
    const auto rank_idx = not not rank;

    std::array<std::string, 2> thread_name = {"master", "slave(" + std::to_string(rank) + ")"};

    std::stringstream strstr;
    strstr << "LOG | " << thread_name[rank_idx] << " | " << format;
    std::printf(strstr.str().c_str(), args...);
#endif
}
