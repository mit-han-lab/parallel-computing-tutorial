#include "matmul.h"
#include <stdio.h>
#include <assert.h>
#include <pthread.h>

namespace matmul
{
    /* This function assume legal matrices */
    void *thread_func(void *args)
    {
        struct thread_args *mat_args = (struct thread_args *)args;
        const struct matrix *A = mat_args->A;
        const struct matrix *B = mat_args->B;
        const struct matrix *C = mat_args->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        int start_i = mat_args->start_i, end_i = mat_args->end_i;

        for (int i = start_i; i < end_i; i++)
            for (int j = 0; j < C->column; j++)
            {
                float acc = 0;
                for (int k = 0; k < A->column; k++)
                    acc += data_A[i * A->column + k] * data_B[k * B->column + j];
                data_C[i * C->column + j] = acc;
            }

        return NULL;
    }

    void MatmulOperator::mat_mul_multithreading(const struct matmul_params *params)
    {
        int j, num_thread = params->opt_params.num_thread;

        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        CHECK_MATRICES(A, B, C);
        assert(num_thread != 0);
        assert(C->row % num_thread == 0);

        pthread_t thread_pool[num_thread];
        struct thread_args threads_args[num_thread];

        // Thread creation
        for (j = 0; j < num_thread; j++)
        {
            threads_args[j].start_i = j * (C->row / num_thread);
            threads_args[j].end_i = (j + 1) * (C->row / num_thread);
            threads_args[j].A = A;
            threads_args[j].B = B;
            threads_args[j].C = C;
            pthread_create(&thread_pool[j], NULL, thread_func, &threads_args[j]);
        }
        // Join threads
        for (j = 0; j < num_thread; j++)
        {
            pthread_join(thread_pool[j], NULL);
        }
    }
}
