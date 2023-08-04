#include "matmul.h"

#include <sys/time.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define RUNS 1


namespace matmul
{
    void MatmulOperator::CHECK_MATRICES(const struct matrix *A, const struct matrix *B, const struct matrix *C)
    {
        assert(A->column == B->row);
        assert(C->column == B->column);
        assert(C->row == A->row);
    }

    void MatmulOperator::naive_mat_mul(const struct matmul_params *params)
    {
        int i, j, k;
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        CHECK_MATRICES(A, B, C);

        for (i = 0; i < C->row; i++)
            for (j = 0; j < C->column; j++)
            {
                float acc = 0;
                for (k = 0; k < A->column; k++)
                    acc += data_A[i * A->column + k] * data_B[k * B->column + j];
                data_C[i * C->column + j] = acc;
            }
    }


    float interval_to_ms(struct timeval *start, struct timeval *end)
    {
        float us_seconds = (end->tv_sec - start->tv_sec) * 1000000 + (end->tv_usec - start->tv_usec);
        return us_seconds / 1000;
    }

    void MatmulOperator::evaluate(IMP_TYPE type, const struct matmul_params *params)
    {
        struct timeval start, end;
        int ms;
        std::string function_name;

        gettimeofday(&start, NULL);
        // choose implementation
        switch (type)
        {
        case NAIVE:
            function_name = "naive_mat_mul";
            for (int i = 0; i < RUNS; i++)
                this->naive_mat_mul(params);
            break;
        case UNROLL:
            function_name = "mat_mul_unrolling";
            for (int i = 0; i < RUNS; i++)
                this->mat_mul_unrolling(params);
            break;
        case REORDER:
            function_name = "mat_mul_reordering";
            for (int i = 0; i < RUNS; i++)
                this->mat_mul_reordering(params);
            break;
        case TILING:
            function_name = "mat_mul_tiling";
            for (int i = 0; i < RUNS; i++)
                this->mat_mul_tiling(params);
            break;
        case MULTITHREAD:
            function_name = "mat_mul_multithreading";
            for (int i = 0; i < RUNS; i++)
                this->mat_mul_multithreading(params);
            break;
        case TRANSPOSE_SIMD:
            function_name = "mat_mul_transpose_simd";
            for (int i = 0; i < RUNS; i++)
                this->mat_mul_transpose_simd(params);
            break;
#ifdef CUDA_ENABLE
        case CUDA:
            function_name = "mat_mul_cuda";
            for (int i = 0; i < RUNS; i++)
                this->mat_mul_cuda(params);
            break;
#endif
        case FAST:
            function_name = "mat_mul_fast";
            for (int i = 0; i < RUNS; i++)
                this->mat_mul_fast(params);
            break;
        default:
            break;
        }
        gettimeofday(&end, NULL);
        ms = interval_to_ms(&start, &end);
        std::cout << function_name << ": " << ms << " ms" << std::endl;
    }

}
