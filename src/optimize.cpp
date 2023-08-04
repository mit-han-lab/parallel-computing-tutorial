#include "matmul.h"
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#ifdef __SSE__
#include <xmmintrin.h> // intel SSE intrinsic
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif


namespace matmul
{

    void *fast_thread_func(void *args)
    {
        struct thread_args *mat_args = (struct thread_args *)args;
        const struct matrix *A = mat_args->A;
        const struct matrix *B = mat_args->B;
        const struct matrix *C = mat_args->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        int start_i = mat_args->start_i, end_i = mat_args->end_i;

        int BLK_SIZE = mat_args->blk_size;
        assert((end_i - start_i) % BLK_SIZE == 0);
        assert(A->column % BLK_SIZE == 0);
        assert(C->column % BLK_SIZE == 0);
        assert(BLK_SIZE % 4 == 0);

        for (int ti = start_i; ti < end_i; ti += BLK_SIZE)
        {
            for (int tj = 0; tj < C->column; tj += BLK_SIZE)
            {
                for (int i = ti; i < ti + BLK_SIZE; i++)
                    for (int j = tj; j < tj + BLK_SIZE; j+=4)
                    {
                        float acc0[4] = {}, acc1[4] = {}, acc2[4] = {}, acc3[4] = {};
                        #ifdef __SSE__
                        __m128 *acc0_fp_128 = (__m128*)acc0;
                        __m128 *acc1_fp_128 = (__m128*)acc1;
                        __m128 *acc2_fp_128 = (__m128*)acc2;
                        __m128 *acc3_fp_128 = (__m128*)acc3;

                        for (int k = 0; k < A->column; k+=4){
                            __m128 Aik_Aik3 = _mm_load_ps(&data_A[i * A->column + k]);
                            __m128 val;
                            val = _mm_mul_ps(Aik_Aik3, _mm_load_ps(&data_B[j * B->column + k]));
                            *acc0_fp_128 = _mm_add_ps(*acc0_fp_128, val);

                            val = _mm_mul_ps(Aik_Aik3, _mm_load_ps(&data_B[(j+1) * B->column + k]));
                            *acc1_fp_128 = _mm_add_ps(*acc1_fp_128, val);

                            val = _mm_mul_ps(Aik_Aik3, _mm_load_ps(&data_B[(j+2) * B->column + k]));
                            *acc2_fp_128 = _mm_add_ps(*acc2_fp_128, val);

                            val = _mm_mul_ps(Aik_Aik3, _mm_load_ps(&data_B[(j+3) * B->column + k]));
                            *acc3_fp_128 = _mm_add_ps(*acc3_fp_128, val);
                        }
                        #endif
                        #ifdef __ARM_NEON
                        float32x4_t *acc0_fp_128 = (float32x4_t*)acc0;
                        float32x4_t *acc1_fp_128 = (float32x4_t*)acc1;
                        float32x4_t *acc2_fp_128 = (float32x4_t*)acc2;
                        float32x4_t *acc3_fp_128 = (float32x4_t*)acc3;

                        for (int k = 0; k < A->column; k+=4){
                            float32x4_t Aik_Aik3 = vld1q_f32(&data_A[i * A->column + k]);
                            float32x4_t val;
                            val = vmulq_f32(Aik_Aik3, vld1q_f32(&data_B[j * B->column + k]));
                            *acc0_fp_128 = vaddq_f32(*acc0_fp_128, val);

                            val = vmulq_f32(Aik_Aik3, vld1q_f32(&data_B[(j+1) * B->column + k]));
                            *acc1_fp_128 = vaddq_f32(*acc1_fp_128, val);

                            val = vmulq_f32(Aik_Aik3, vld1q_f32(&data_B[(j+2) * B->column + k]));
                            *acc2_fp_128 = vaddq_f32(*acc2_fp_128, val);

                            val = vmulq_f32(Aik_Aik3, vld1q_f32(&data_B[(j+3) * B->column + k]));
                            *acc3_fp_128 = vaddq_f32(*acc3_fp_128, val);
                        }
                        #endif
                        data_C[i * C->column + j] = acc0[0] + acc0[1] + acc0[2] + acc0[3];
                        data_C[i * C->column + j + 1] = acc1[0] + acc1[1] + acc1[2] + acc1[3];
                        data_C[i * C->column + j + 2] = acc2[0] + acc2[1] + acc2[2] + acc2[3];
                        data_C[i * C->column + j + 3] = acc3[0] + acc3[1] + acc3[2] + acc3[3];
                    }
            }
        }

        return NULL;
    }

    void MatmulOperator::mat_mul_fast(const struct matmul_params *params)
    {
        int j, num_thread = params->opt_params.num_thread;

        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

        assert(A->column == B->column);
        assert(C->column == B->row);
        assert(C->row == A->row);
        assert(num_thread != 0);
        assert(C->row % num_thread == 0);

        pthread_t thread_pool[num_thread];
        struct thread_args threads_args[num_thread];

        // Thread creation
        for (j = 0; j < num_thread; j++)
        {
            threads_args[j].start_i = j * (C->row / num_thread);
            threads_args[j].end_i = (j + 1) * (C->row / num_thread);
            threads_args[j].blk_size = params->opt_params.blk_size;
            threads_args[j].A = A;
            threads_args[j].B = B;
            threads_args[j].C = C;
            pthread_create(&thread_pool[j], NULL, fast_thread_func, &threads_args[j]);
        }
        // Join threads
        for (j = 0; j < num_thread; j++)
        {
            pthread_join(thread_pool[j], NULL);
        }
    }

}

