#include "matmul.h"
#include <stdio.h>
#include <assert.h>
#ifdef __SSE__
#include <xmmintrin.h> // intel SSE intrinsic
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define MAX_TRANSPOSE_BUFFER 10 * 1024 * 1024
float transpose_tmp[MAX_TRANSPOSE_BUFFER];

namespace matmul
{
    inline void simd_mul_fp_128(const float *a, const float *b, float *c)
    {
        #ifdef __SSE__
        __m128 val = _mm_mul_ps(_mm_load_ps(a), _mm_load_ps(b));
        __m128 acc = _mm_add_ps(_mm_load_ps(c), val);
        _mm_store_ps(c, acc);
        #endif
        #ifdef __ARM_NEON
        float32x4_t val = vmulq_f32(vld1q_f32(a), vld1q_f32(b));
        float32x4_t *c_vec = (float32x4_t *)c;
        *c_vec = vaddq_f32(*c_vec, val);
        #endif
    }

    void MatmulOperator::mat_mul_transpose_simd(const struct matmul_params *params)
    {
        int i, j, k;

        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        CHECK_MATRICES(A, B, C);

        // transpose the B
        for (i = 0; i < B->column; i++)
            for (j = 0; j < B->row; j++)
                transpose_tmp[i * B->row + j] = data_B[j * B->column + i];

        for (i = 0; i < C->row; i++)
            for (j = 0; j < C->column; j++)
            {
                float accumulators[4] = {};
                for (k = 0; k < A->column; k += 4)
                    simd_mul_fp_128(&data_A[i * A->column + k], &transpose_tmp[j * B->row + k], accumulators);
                data_C[i * C->column + j] = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
            }
    }
}
