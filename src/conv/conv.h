#include "lib/matmul.h"
struct conv_params
{
    int input_h, input_w, input_c;
    int output_h, output_w, output_c;
    int kernel_h, kernel_w, stride;
    int im2col_columns;
    float *input, *weight, *output;
    optimization_params opt;
};
namespace conv
{

    class convOperator
    {
    public:
        enum IMP_TYPE
        {
            NAIVE = 0,
            UNROLL = 1,
            REORDER = 2,
            TILING = 3,
            MULTITHREAD = 4,
            TRANSPOSE = 5,
            TRANSPOSE_SIMD = 6,
            FAST = 7
        };
        void naive_conv2d(const struct conv_params *params);
        void im2col_conv2d(const struct conv_params *params, IMP_TYPE type);
        void evaluate(IMP_TYPE type, const struct conv_params *params);
    };
}