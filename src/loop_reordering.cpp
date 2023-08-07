#include "matmul.h"
#include <stdio.h>

namespace matmul
{
    void MatmulOperator::mat_mul_reordering(const struct matmul_params *params)
    {
        int i, j, k;
        float Aik;

        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        CHECK_MATRICES(A, B, C);

        for (i = 0; i < C->row; i++)
            for (j = 0; j < C->column; j++)
                data_C[i * C->column + j] = 0;

        for (i = 0; i < C->row; i++)
            for (k = 0; k < A->column; k++)
            {
                Aik = data_A[i * A->column + k];
                for (j = 0; j < C->column; j++)
                {
                    data_C[i * C->column + j] += Aik * data_B[k * B->column + j];
                }
            }
    }

}

