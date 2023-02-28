#include "conv.h"

#include <stdio.h>
#include <math.h>
#include <iostream>

#define MAX_PRECISION_ERROR 0.01

#define INPUT_H 64
#define INPUT_W 64
#define INPUT_C 128
#define KERNEL_H 3
#define KERNEL_W 3
#define STRIDE 1
#define OUTPUT_H 64
#define OUTPUT_W 64
#define OUTPUT_C 128
#define IM2COL_COLUMNS 64 // This should be multiple of num_thread and BLK_SIZE
#define NUM_THREAD 4
#define BLK_SIZE 16

float input[INPUT_H * INPUT_W * INPUT_C]; // HWC
float weight[KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_C]; // HWIO
float transpose_weight[KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_C]; // OHWI
float naive[OUTPUT_H * OUTPUT_W * OUTPUT_C]; // HWC
float output[OUTPUT_H * OUTPUT_W * OUTPUT_C]; // HWC

using namespace conv;

bool check_identical(float matA[], float matB[], int size)
{
    for (int i = 0; i < size; i++)
    {
        if (abs((matA[i] - matB[i]) / (matA[i])) > MAX_PRECISION_ERROR)
        {
            printf("%d: %f, %.8f\n", i, matA[i], matB[i]);
            return false;
        }
    }
    return true;
}

void initialize_matrix(float A[], int size)
{
    for (int i = 0; i < size; i++)
    {
        A[i] = (float)(rand()) / (float)(RAND_MAX);
    }
}

void print_c(float C[]){
    for (int i = 0; i < OUTPUT_H; i++){
        for (int j = 0; j < OUTPUT_W; j++){
            for (int k = 0; k < OUTPUT_C; k++){
                std::cout << C[(i * OUTPUT_W + j) * OUTPUT_C + k] << ",";
            }
        }
        std::cout << std::endl;
    }
}


int main()
{
    // initialize
    initialize_matrix(input, INPUT_H * INPUT_W * INPUT_C);
    initialize_matrix(weight, INPUT_C * KERNEL_H * KERNEL_H * OUTPUT_C);

    struct conv_params params;
    params.input = input; params.input_h = INPUT_H; params.input_w = INPUT_W; params.input_c = INPUT_C;
    params.weight = weight; params.kernel_h = KERNEL_H; params.kernel_w = KERNEL_W; params.stride = STRIDE;
    params.output = naive; params.output_h = OUTPUT_H; params.output_w = OUTPUT_W; params.output_c = OUTPUT_C;
    params.im2col_columns = IM2COL_COLUMNS;
    params.opt.blk_size = BLK_SIZE; params.opt.num_thread = NUM_THREAD;

    conv::convOperator conv_op = conv::convOperator();

    conv_op.evaluate(conv_op.NAIVE, &params);

    params.output = output;
    // unrolling
    conv_op.evaluate(conv_op.UNROLL, &params);
    if (!check_identical(naive, output, OUTPUT_H * OUTPUT_W * OUTPUT_C))
        printf("incorrect output from UNROLL\n");

    // reordering
    conv_op.evaluate(conv_op.REORDER, &params);
    if (!check_identical(naive, output, OUTPUT_H * OUTPUT_W * OUTPUT_C))
        printf("incorrect output from REORDER\n");

    // multithreading
    conv_op.evaluate(conv_op.MULTITHREAD, &params);
    if (!check_identical(naive, output, OUTPUT_H * OUTPUT_W * OUTPUT_C))
        printf("incorrect output from MULTITHREAD\n");

    // transpose
    conv_op.evaluate(conv_op.TRANSPOSE, &params);
    if (!check_identical(naive, output, OUTPUT_H * OUTPUT_W * OUTPUT_C))
        printf("incorrect output from TRANSPOSE\n");

    // transpose + simd
    conv_op.evaluate(conv_op.TRANSPOSE_SIMD, &params);
    if (!check_identical(naive, output, OUTPUT_H * OUTPUT_W * OUTPUT_C))
        printf("incorrect output from TRANSPOSE_SIMD\n");

    // fast
    // This op assumes B is transposed
    for (int i = 0; i < OUTPUT_C; i++)
        for (int j = 0; j < KERNEL_H * KERNEL_W * INPUT_C; j++)
            transpose_weight[i * KERNEL_H * KERNEL_W * INPUT_C + j] = weight[j * OUTPUT_C + i];
    params.weight = transpose_weight;
    conv_op.evaluate(conv_op.FAST, &params);
    if (!check_identical(naive, output, OUTPUT_H * OUTPUT_W * OUTPUT_C))
        printf("incorrect output from FAST\n");



    return 0;
}