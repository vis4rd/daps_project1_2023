#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>

struct Complex
{
    float real{};
    float img{};
};

int reverseBits(int number, int NO_OF_BITS)
{
    int reverse_num = 0;

    for (int i = 0; i < NO_OF_BITS; i++)
    {
        int temp = (number & (1 << i));
        if (temp)
        {
            reverse_num |= (1 << ((NO_OF_BITS - 1) - i));
        }
    }

    return reverse_num;
}

int main(int argc, char **argv)
{
    MPI_Status status;

    MPI_Init(&argc, &argv);

    const int process_count = []() -> int
    {
        int process_count{};
        MPI_Comm_size(MPI_COMM_WORLD, &process_count);
        return process_count;
    }();

    const int rank = []() -> int
    {
        int rank{};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }();

    double bb1time{};
    double ab1time{};
    const int input_values_count = [&bb1time, &ab1time, &rank]() -> int
    {
        int input_values_count{};
        if (rank == 0)
        {
            std::printf("Enter No. of input values(it should be in form of 2^x) : ");
            std::scanf("%d", &input_values_count);
        }
        bb1time = MPI_Wtime();
        MPI_Bcast(&input_values_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        ab1time = MPI_Wtime();
        return input_values_count;
    }();

    std::vector<Complex> input_values(input_values_count + 1);
    Complex seq[input_values_count + 1]{};
    Complex temp[input_values_count + 1]{};

    input_values[0].real = 0.0;
    input_values[0].img = 0.0;
    seq[0].real = 0.0;
    seq[0].img = 0.0;
    temp[0].real = 0.0;
    temp[0].img = 0.0;

    MPI_Datatype MPI_COMPLEX_T;

    MPI_Type_create_struct(
        sizeof(Complex) / sizeof(float),
        (const int[2]){sizeof(float), sizeof(float)},
        (const MPI_Aint[2]){0, 4},
        (const MPI_Datatype[2]){MPI_FLOAT, MPI_FLOAT},
        &MPI_COMPLEX_T);
    MPI_Type_commit(&MPI_COMPLEX_T);

    if (rank == 0)
    {
        std::printf("Enter total %d values in floating point formet(separated with space) : ", input_values_count);

        for (int i = 1; i < input_values_count + 1; i++)
        {
            std::scanf("%f", &input_values[i].real);
            input_values[i].img = 0.0;
        }

        for (int i = 1; i < input_values_count + 1; i++)
        {
            seq[i].real = input_values[reverseBits(i - 1, std::log2f(input_values_count) / std::log2f(2)) + 1].real; // log2 (x) = logy (x) / logy (2)
            seq[i].img = 0.0;
        }
    }

    const double bb2time = MPI_Wtime();
    MPI_Bcast(seq, input_values_count + 1, MPI_COMPLEX_T, 0, MPI_COMM_WORLD);
    const double ab2time = MPI_Wtime();

    const double starttime = MPI_Wtime();

    for (int div = 1, key = std::log2f(input_values_count); key > 0; key--)
    {
        if (rank != 0)
        {
            if (((rank + div - 1) / div) % 2 == 0)
            {
                temp[rank].real = seq[rank - div].real +
                                  (std::cos(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank].real)) +
                                  (std::sin(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank].img));

                temp[rank].img = seq[rank - div].img +
                                 (std::cos(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank].img)) -
                                 (std::sin(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank].real));

                MPI_Send(&temp[rank], 1, MPI_COMPLEX_T, 0, 0, MPI_COMM_WORLD);
            }
            else
            {
                temp[rank].real = seq[rank].real +
                                  (std::cos(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank + div].real)) +
                                  (std::sin(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank + div].img));

                temp[rank].img = seq[rank].img +
                                 (std::cos(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank + div].img)) -
                                 (std::sin(M_PI * ((rank - 1) % (div * 2)) / div) * (seq[rank + div].real));

                MPI_Send(&temp[rank], 1, MPI_COMPLEX_T, 0, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            for (int i = 1; i < input_values_count + 1; i++)
            {
                MPI_Recv(&temp[i], 1, MPI_COMPLEX_T, i, 0, MPI_COMM_WORLD, &status);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (int i = 1; i < input_values_count + 1; i++)
            {
                seq[i].real = temp[i].real;
                seq[i].img = temp[i].img;
            }
            div *= 2;
        }

        MPI_Bcast(seq, input_values_count + 1, MPI_COMPLEX_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&div, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    const double endtime = MPI_Wtime();

    if (0 == rank)
    {
        std::printf("\n");

        for (int i = 1; i < input_values_count + 1; i++)
        {
            std::printf("X[%d] : %f", i - 1, seq[i].real);

            if (seq[i].img >= 0)
            {
                std::printf("+j%f\n", seq[i].img);
            }
            else
            {
                std::printf("-j%f\n", seq[i].img - 2 * seq[i].img);
            }
        }

        std::printf("\n");
        std::printf("Time to broadcast num variable : %lf ms\n", (ab1time - bb1time) * 1000);
        std::printf("Time to broadcast input_values and seq array : %lf ms\n", (ab2time - bb2time) * 1000);
        std::printf("Time to compute FFT parallely : %lf ms\n", (endtime - starttime) * 1000);
        std::printf("Total Time : %lf ms\n", ((endtime - starttime) + (ab2time - bb2time) + (ab1time - bb1time)) * 1000);
        std::printf("\n");
    }

    const int finalize_success = MPI_Finalize();
    std::printf("Rank = %d, success = %d (after)\n", rank, finalize_success);

    return 0;
}
