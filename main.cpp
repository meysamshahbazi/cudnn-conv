#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

/**
 * Minimal example to apply sigmoid activation on a tensor 
 * using cuDNN.
 **/

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


int main(int argc, char** argv)
{    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    checkCUDNN(cudnnCreate(&cudnn));
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));


    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/256,
                                      /*image_height=*/22,
                                      /*image_width=*/22));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/10,
                                        /*image_height=*/19,
                                        /*image_width=*/19));


    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/10,
                                        /*in_channels=*/256,
                                        /*kernel_height=*/4,
                                        /*kernel_width=*/4));
    
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                            /*pad_height=*/0,
                                            /*pad_width=*/0,
                                            /*vertical_stride=*/1,
                                            /*horizontal_stride=*/1,
                                            /*dilation_height=*/1,
                                            /*dilation_width=*/1,
                                            /*mode=*/CUDNN_CONVOLUTION, // TODO: it may need change ... 
                                            /*computeType=*/CUDNN_DATA_FLOAT));


    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;

    int returnedAlgoCount = 1;
    cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &returnedAlgoCount);
    checkCUDNN(
    cudnnFindConvolutionForwardAlgorithm(/* cudnnHandle_t handle */ cudnn,
                                        /* const cudnnTensorDescriptor_t xDesc */input_descriptor,
                                        /* const cudnnFilterDescriptor_t wDesc */ kernel_descriptor,
                                        /* const cudnnConvolutionDescriptor_t convDesc */convolution_descriptor,
                                        /* const cudnnTensorDescriptor_t yDesc */output_descriptor,
                                        /* const int requestedAlgoCoun */1,
                                        /* int *returnedAlgoCount */&returnedAlgoCount,
                                        /* cudnnConvolutionFwdAlgoPerf_t *perfResults */ &convolution_algorithm) );

    size_t workspace_bytes = 0;

    int n;
    int c;
    int h;
    int w;

    checkCUDNN(
    cudnnGetConvolution2dForwardOutputDim(
    /* const cudnnConvolutionDescriptor_t  convDesc */ convolution_descriptor,
    /* const cudnnTensorDescriptor_t       inputTensorDesc */input_descriptor,
    /* const cudnnFilterDescriptor_t       filterDesc */kernel_descriptor,
    &n,
    &c,
    &h,
    &w));

    std::cout<<n<<"\t"<<c<<"\t"<<h<<"\t"<<w<<"\n";



    cudnnGetConvolutionForwardWorkspaceSize(/* cudnnHandle_t handle */cudnn,
                                        /* const cudnnTensorDescriptor_t xDesc */input_descriptor,
                                        /* const cudnnFilterDescriptor_t wDesc */kernel_descriptor,
                                        /* const cudnnConvolutionDescriptor_t convDesc */convolution_descriptor,
                                        /* const cudnnTensorDescriptor_t yDesc */output_descriptor,
                                        /* cudnnConvolutionFwdAlgo_t algo */
                                        // convolution_algorithm.algo,
                                        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                        /* size_t *sizeInBytes */&workspace_bytes);

/*     typedef enum {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8
} cudnnConvolutionFwdAlgo_t;
 */
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
            << std::endl;

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    int input_bytes = 1 * 256 * 22 * 22 * sizeof(float);

    float* d_input{nullptr};
    cudaMalloc(&d_input, input_bytes);

    // cudaMemset(d_input, 1.0f, input_bytes);
    float* h_input = new float[256 * 22 * 22];
    for(int i = 0;i<256 * 22 * 22;i++) h_input[i] = 1.0f;
    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);

    int output_bytes = 1 * 10 * 19 * 19 * sizeof(float);
    float* d_output{nullptr};
    cudaMalloc(&d_output, output_bytes);
    // cudaMemset(d_output, 0.0f, output_bytes);

    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, 10*256*4*4*sizeof(float));
    float* h_kernel = new float[10*256*4*4];
    for(int i = 0;i<10*256*4*4;i++) h_kernel[i] = 1.0f;
    // cudaMemset(d_kernel, 1.0f, 10*256*4*4*sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, 10*256*4*4*sizeof(float), cudaMemcpyHostToDevice);


    const float alpha = 1, beta = 0;
    /* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

    /* Function to perform the forward pass for batch convolution */
    checkCUDNN(
    cudnnConvolutionForward(/* cudnnHandle_t handle */cudnn,
                            /* const void *alpha */&alpha,
                            /* const cudnnTensorDescriptor_t xDesc */input_descriptor,
                            /* const void *x */d_input,
                            /* const cudnnFilterDescriptor_t wDesc */kernel_descriptor,
                            /* const void *w */d_kernel,
                            /* const cudnnConvolutionDescriptor_t convDesc */convolution_descriptor,
                            /* cudnnConvolutionFwdAlgo_t algo */
                            // convolution_algorithm.algo,
                            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                            /* void *workSpace */d_workspace,
                            /* size_t workSpaceSizeInBytes */workspace_bytes,
                            /* const void *beta */&beta,
                            /* const cudnnTensorDescriptor_t yDesc */output_descriptor,
                            /* void *y */d_output) );

    cudaDeviceSynchronize();
    float* h_output = new float[10 * 19 * 19];
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    // cudaStreamSynchronize(0);

    // Do something with h_output ...
    // for(int i=0;i<1 * 10 * 19 * 19;i++)
    int index = 0;
    for(int i =0;i<10;i++)
    {
        for(int j =0;j<19;j++)
        {
            // std::cout<<"[ "
            for(int k=0;k<19;k++)
            {

                std::cout<<h_output[index]<<"\t";
                index ++;
            }
            std::cout<<"\n";
        }
    }
        // std::cout<<h_output[i]<<"\t";

    std::cout<<std::endl;

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);




    return 0;
}

