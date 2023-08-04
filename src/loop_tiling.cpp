#include "matmul.h"
#include <stdio.h>
#include <assert.h>

namespace matmul
{
    void MatmulOperator::mat_mul_tiling(const struct matmul_params *params)
    {
        int i, j, k, ti, tj, tk;
        float Aik;

        int BLK_SIZE = params->opt_params.blk_size;

        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        CHECK_MATRICES(A, B, C);
        assert(C->row % BLK_SIZE == 0);
        assert(A->column % BLK_SIZE == 0);
        assert(C->column % BLK_SIZE == 0);

        for (i = 0; i < C->row; i++)
            for (j = 0; j < C->column; j++)
                data_C[i * C->column + j] = 0;

        for (ti = 0; ti < C->row; ti += BLK_SIZE)
        {
            for (tk = 0; tk < A->column; tk += BLK_SIZE)
            {
                for (tj = 0; tj < C->column; tj += BLK_SIZE)
                {
                    for (i = ti; i < ti + BLK_SIZE; i++)
                        for (k = tk; k < tk + BLK_SIZE; k++)
                        {
                            Aik = data_A[i * A->column + k];
                            for (j = tj; j < tj + BLK_SIZE; j++)
                            {
                                data_C[i * C->column + j] += Aik * data_B[k * B->column + j];
                            }
                        }
                }
            }
        }
    }
}
