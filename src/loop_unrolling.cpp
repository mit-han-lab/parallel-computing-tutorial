#ifndef MATMUL_H_
#define MATMUL_H_

#include "matmul.h"
#include <stdio.h>

namespace matmul
{
    void MatmulOperator::mat_mul_unrolling(const struct matmul_params *params)
    {
        int i, j, k;

        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        CHECK_MATRICES(A, B, C);

        for (i = 0; i < C->row; i++)
            for (j = 0; j < C->column; j += 8)
            {
                float acc0 = 0;
                float acc1 = 0;
                float acc2 = 0;
                float acc3 = 0;
                float acc4 = 0;
                float acc5 = 0;
                float acc6 = 0;
                float acc7 = 0;
                for (k = 0; k < A->column; k += 4)
                {
                    float Aik0 = data_A[i * A->column + k];
                    float Aik1 = data_A[i * A->column + k + 1];
                    float Aik2 = data_A[i * A->column + k + 2];
                    float Aik3 = data_A[i * A->column + k + 3];

                    acc0 += Aik0 * data_B[k * B->column + j];
                    acc0 += Aik1 * data_B[(k + 1) * B->column + j];
                    acc0 += Aik2 * data_B[(k + 2) * B->column + j];
                    acc0 += Aik3 * data_B[(k + 3) * B->column + j];

                    acc1 += Aik0 * data_B[k * B->column + j + 1];
                    acc1 += Aik1 * data_B[(k + 1) * B->column + j + 1];
                    acc1 += Aik2 * data_B[(k + 2) * B->column + j + 1];
                    acc1 += Aik3 * data_B[(k + 3) * B->column + j + 1];

                    acc2 += Aik0 * data_B[k * B->column + j + 2];
                    acc2 += Aik1 * data_B[(k + 1) * B->column + j + 2];
                    acc2 += Aik2 * data_B[(k + 2) * B->column + j + 2];
                    acc2 += Aik3 * data_B[(k + 3) * B->column + j + 2];

                    acc3 += Aik0 * data_B[k * B->column + j + 3];
                    acc3 += Aik1 * data_B[(k + 1) * B->column + j + 3];
                    acc3 += Aik2 * data_B[(k + 2) * B->column + j + 3];
                    acc3 += Aik3 * data_B[(k + 3) * B->column + j + 3];

                    acc4 += Aik0 * data_B[k * B->column + j + 4];
                    acc4 += Aik1 * data_B[(k + 1) * B->column + j + 4];
                    acc4 += Aik2 * data_B[(k + 2) * B->column + j + 4];
                    acc4 += Aik3 * data_B[(k + 3) * B->column + j + 4];

                    acc5 += Aik0 * data_B[k * B->column + j + 5];
                    acc5 += Aik1 * data_B[(k + 1) * B->column + j + 5];
                    acc5 += Aik2 * data_B[(k + 2) * B->column + j + 5];
                    acc5 += Aik3 * data_B[(k + 3) * B->column + j + 5];

                    acc6 += Aik0 * data_B[k * B->column + j + 6];
                    acc6 += Aik1 * data_B[(k + 1) * B->column + j + 6];
                    acc6 += Aik2 * data_B[(k + 2) * B->column + j + 6];
                    acc6 += Aik3 * data_B[(k + 3) * B->column + j + 6];

                    acc7 += Aik0 * data_B[k * B->column + j + 7];
                    acc7 += Aik1 * data_B[(k + 1) * B->column + j + 7];
                    acc7 += Aik2 * data_B[(k + 2) * B->column + j + 7];
                    acc7 += Aik3 * data_B[(k + 3) * B->column + j + 7];
                }
                data_C[i * C->column + j] = acc0;
                data_C[i * C->column + j + 1] = acc1;
                data_C[i * C->column + j + 2] = acc2;
                data_C[i * C->column + j + 3] = acc3;
                data_C[i * C->column + j + 4] = acc4;
                data_C[i * C->column + j + 5] = acc5;
                data_C[i * C->column + j + 6] = acc6;
                data_C[i * C->column + j + 7] = acc7;
            }
    }

}

#endif // MATMUL_H_
