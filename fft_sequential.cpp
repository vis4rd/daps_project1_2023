#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv)
{
    int num;
    std::printf("Enter No. of input values(it should be in form of 2^x) : ");
    std::scanf("%d", &num);

    float temp, input[num], result_real[num], result_img[num], sum = 0.0;

    std::printf("Enter total %d values in floating point format(separated with space) : ", num);

    for(int i = 0; i < num; i++)
    {
        std::scanf("%f", &input[i]);
    }

    std::clock_t start = std::clock();

    for(int k = 0; k < num; k++)
    {
        result_real[k] = 0.0;
        result_img[k] = 0.0;

        for(int n = 0; n < num; n++)
        {
            temp = -2 * M_PI * n * k / num;
            result_real[k] = result_real[k] + (input[n] * std::cos(temp));
            result_img[k] = result_img[k] + (input[n] * std::sin(temp));
        }
    }

    std::clock_t end = std::clock();

    std::printf("\n");

    for(int k = 0; k < num; k++)
    {
        std::printf("X[%d] : %+6.2f", k, result_real[k]);
        if(result_img[k] >= 0)
        {
            std::printf(" + i%-6.2f\n", result_img[k]);
        }
        else
        {
            std::printf(" - i%-6.2f\n", result_img[k] - 2 * result_img[k]);
        }
    }

    std::printf("\n");

    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    std::printf("Time to compute FFT sequentially : %lf ms\n", cpu_time_used * 1000);

    return 0;
}
