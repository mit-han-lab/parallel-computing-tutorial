#ifndef CONV_H_
#define CONV_H_
#include "conv.h"

#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#define RUNS 1
#define IM2COL_BUFFER_SIZE 1024 * 1024 * 12
float im2col_buffer[IM2COL_BUFFER_SIZE];
float Tweight[IM2COL_BUFFER_SIZE]; // OHWI

using namespace matmul;

namespace conv
{
    float get_pixel(int h, int w, int c, int input_h, int input_w, int input_c, float *input)
    {
        if (h < 0 || h >= input_h)
            return 0.0f;
        if (w < 0 || w >= input_w)
            return 0.0f;

        return input[(h * input_w + w) * input_c + c];
    }

    void get_channels_of_a_pixel(int h, int w, int input_h, int input_w, int input_c, float *input, float *buffer)
    {
        if (h < 0 || h >= input_h)
            memset(buffer, 0, input_c * sizeof(float));
        else if (w < 0 || w >= input_w)
            memset(buffer, 0, input_c * sizeof(float));
        else
            memcpy(buffer, &input[(h * input_w + w) * input_c], input_c * sizeof(float));
    }

    void convOperator::naive_conv2d(const struct conv_params *params)
    {
        int input_h = params->input_h, input_w = params->input_w, input_c = params->input_c;
        int output_h = params->output_h, output_w = params->output_w, output_c = params->output_c;
        int kernel_h = params->kernel_h, kernel_w = params->kernel_w, stride = params->stride;
        float *input = params->input, *weight = params->weight, *output = params->output;


        for (int h = 0; h < output_h; h++)
        {
            for (int w = 0; w < output_w; w++)
            {
                for (int o = 0; o < output_c; o++)
                {
                    float acc = 0;

                    for (int k_h = 0; k_h < kernel_h; k_h++)
                    {
                        for (int k_w = 0; k_w < kernel_w; k_w++)
                        {
                            for (int i_ch = 0; i_ch < input_c; i_ch++)
                            {
                                int start_y = h * stride;
                                int start_x = w * stride;

                                float pixel = get_pixel(start_x + k_h - kernel_h / 2, start_y + k_w - kernel_w / 2, i_ch, input_h, input_w, input_c, input);
                                // assume weights are in the HWIO format
                                int weight_idx = ((k_h * kernel_w + k_w) * input_c + i_ch) * output_c + o;
                                float kernel = weight[weight_idx];
                                acc += pixel * kernel;
                            }
                        }
                    }
                    output[(h * output_w + w) * output_c + o] = acc;
                }
            }
        }
    }

    void convOperator::im2col_conv2d(const struct conv_params *params, IMP_TYPE type)
    {
        int input_h = params->input_h, input_w = params->input_w, input_c = params->input_c;
        int output_h = params->output_h, output_w = params->output_w, output_c = params->output_c;
        int kernel_h = params->kernel_h, kernel_w = params->kernel_w, stride = params->stride;
        int im2col_columns = params->im2col_columns;
        // assume weights are in the HWIO format
        float *input = params->input, *weight = params->weight, *output = params->output;

        matmul::MatmulOperator matmul_op = matmul::MatmulOperator();
        struct matmul_params matmul_params;

        matmul_params.A.column = kernel_h * kernel_w * input_c;
        matmul_params.A.row = im2col_columns;
        matmul_params.B.column = output_c;
        matmul_params.B.row = kernel_h * kernel_w * input_c;
        matmul_params.B.data_ptr = weight;
        matmul_params.C.column = output_c;
        matmul_params.C.row = im2col_columns;
        matmul_params.opt_params.blk_size = params->opt.blk_size;
        matmul_params.opt_params.num_thread = params->opt.num_thread;

        // make sure the hyperparameter is supported
        assert(im2col_columns != 0);

        int buffer_cnt = 0;
        int col_cnt = 0;
        for (int h = 0; h < output_h; h++)
        {
            for (int w = 0; w < output_w; w ++)
            {
                // im2col
                int start_y = h * stride;
                int start_x = (w) * stride;
                for (int k_h = -kernel_h / 2; k_h <= kernel_h / 2; k_h++)
                {
                    for (int k_w = -kernel_w / 2; k_w <= kernel_w / 2; k_w++)
                    {
                        get_channels_of_a_pixel(start_x + k_h, start_y + k_w, input_h, input_w, input_c, input, &im2col_buffer[buffer_cnt * input_c]);
                        buffer_cnt += 1;
                    }
                }
                col_cnt++;

                if (im2col_columns != col_cnt)
                    continue;
                col_cnt = 0;
                buffer_cnt = 0;

                // matrix multiplication
                matmul_params.A.data_ptr = im2col_buffer;
                matmul_params.C.data_ptr = output;

                switch (type) {
                    case UNROLL:
                        matmul_op.mat_mul_unrolling(&matmul_params);
                        break;
                    case REORDER:
                        matmul_op.mat_mul_reordering(&matmul_params);
                        break;
                    case TILING:
                        matmul_op.mat_mul_tiling(&matmul_params);
                        break;
                    case MULTITHREAD:
                        matmul_op.mat_mul_multithreading(&matmul_params);
                        break;
                    case TRANSPOSE:
                        matmul_op.mat_mul_transpose(&matmul_params);
                        break;
                    case TRANSPOSE_SIMD:
                        matmul_op.mat_mul_transpose_simd(&matmul_params);
                        break;
                    case FAST:
                        // This op assumes B is transposed
                        matmul_params.B.column = kernel_h * kernel_w * input_c;
                        matmul_params.B.row = output_c;
                        matmul_op.mat_mul_fast(&matmul_params);
                        // return;
                        break;
                    default:
                        matmul_op.naive_mat_mul(&matmul_params);
                        break;
                }
                output += im2col_columns * output_c;
            }
        }

    }

    float interval_to_ms(struct timeval *start, struct timeval *end)
    {
        float us_seconds = (end->tv_sec - start->tv_sec) * 1000000 + (end->tv_usec - start->tv_usec);
        return us_seconds / 1000;
    }

    void convOperator::evaluate(IMP_TYPE type, const struct conv_params *params)
    {
        struct timeval start, end;
        int ms;
        std::string function_name;

        gettimeofday(&start, NULL);
        // choose implementation
        switch (type)
        {
        case NAIVE:
            function_name = "naive_conv2d";
            for (int i = 0; i < RUNS; i++)
                this->naive_conv2d(params);
            break;
        default:
            function_name = "im2col_conv2d";
            for (int i = 0; i < RUNS; i++)
                this->im2col_conv2d(params, type);
            
            switch (type)
            {
                case UNROLL:
                    function_name = "unrolling";
                    break;
                case REORDER:
                    function_name = "reordering";
                    break;
                case TILING:
                    function_name = "tiling";
                    break;
                case MULTITHREAD:
                    function_name = "multithreading";
                    break;
                case TRANSPOSE:
                    function_name = "transpose";
                    break;
                case TRANSPOSE_SIMD:
                    function_name = "transpose_simd";
                    break;
                case FAST:
                    function_name = "fast";
                    break;
                default:
                    break;
            }
            break;
        }
        gettimeofday(&end, NULL);
        ms = interval_to_ms(&start, &end);
        std::cout << function_name << ": " << ms << " ms" << std::endl;
    }
}
#endif