#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <omp.h>
#include <iostream>

// CUDA kernel. Each thread takes care of one element of c

#define BLOCK_SIZE 1024;

__global__ void gemm(float* a, float* b, float* c, int m, int n, int k){
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint column = blockIdx.x * blockDim.x + threadIdx.x;

    if (column<<k and row<<m){
        float sum = 0.0f;
        for(int i=0; i<n; i++){

            sum += *(a + row*n + i) * *(b + i*k + column);
        }

        *(c + row* k + column) = sum;
    }
}


__device__  void ReLU(float* input, uint size){
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id<size){
        float  element = input[id];
        float sign = (int32_t(element)>>31) + 1.0f;
        input[id] = sign * element;
    }
}

__global__ void Leaky_ReLU(float* input, uint size){
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id<size){
        float  element = input[id];
        float scale = 1.0f/2.0f;
        float sign = (int32_t(element)>>31) + 2.0f;
        input[id] = scale * sign * element;
    }
}


__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    uint id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

__global__ void matrixAdd(double * a, double * b, double * c,int width, int height ){
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i< width and j < height){
        *(c + i*width + j) = *(a + i*width + j) + *(b + i*width + j);
    }
}
template<typename type>
__global__
void Add(size_t size, type* x, type* y){
    for(int i=0; i<size; i++){
        *(y + i) = *(x + i) + *(y + i);
    }
}

template<typename type>
type myMax(type a, type b){
    return a > b ? a : b;
}

template<typename type>

class exampleClass{
public:
    exampleClass(size_t input_size = 1<<30){
        this->size = input_size;
        cudaMallocManaged(&this->x, this->size * sizeof(type));
        cudaMallocManaged(&this->y, this->size * sizeof(type));
#pragma omp parallel for simd
        for(int i=0; i<size; i++){
            *(x+i) = (type)1.0;
            *(y+i) = (type)2.0;
        }

    }

    void add(int threads, int blocks, int check=1){
        type max_error = (type)0.0;
        Add<type><<<threads, blocks>>>(this->size, this->x, this->y);

        if (check){
            for(int i=0; i< this->size; i++)
                max_error = myMax<type>(max_error, abs(*(y+i) - (type)3));
        }
    }

    ~exampleClass(void){
        std::cout<<"Destructing Class Now"<<std::endl;
        cudaFree(this->x);
        cudaFree(this->y);
        std::cout<<"Freed Memory"<<std::endl;
    }


private:
    type* x;
    type* y;
    size_t size;
};

int main( int argc, char* argv[] )
{
    std::cout<<"Here";

    float* a = (float*)malloc(1000 * 50000 * sizeof(float ));
    float* b = (float*)malloc(50000 * 7000 * sizeof(float ));
    float* c = (float*)malloc(1000 * 7000 * sizeof(float ));
    int m = 1000;
    int n = 50000;
    int k = 7000;
    int* d_m, *d_n, *d_k;

    cudaMalloc(&d_m, sizeof(int));
    cudaMalloc(&d_n, sizeof(int));
    cudaMalloc(&d_k, sizeof(int ));

    cudaMemcpy(d_m, &m, sizeof(int ), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_n, &n, sizeof(int ), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_k, &k, sizeof(int ), cudaMemcpyDeviceToHost);

#pragma omp parallel for
    for(int i =0; i<1000; i++){
#pragma omp simd
        for(int j=0; j<50000; j++){
            *(a + i* 1000 + j) = float(i + j) + 1.0f;
        }
    }

#pragma omp parallel for
    for(int i =0; i<50000; i++){
#pragma omp simd
        for(int j=0; j<7000; j++){
            *(b + i*50000 + j) = float(i +j) + 1.0f;
        }
    }

    float* d_a, *d_b, *d_c;

    cudaMalloc(&d_a, 1000 * 50000 * sizeof(float ));
    cudaMalloc(&d_b, 50000 * 7000 * sizeof(float ));
    cudaMalloc(&d_c, 1000 * 7000 * sizeof(float ));

    cudaMemcpy(d_a, a, 1000 * 50000 * sizeof(float ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 50000 * 7000 * sizeof(float ), cudaMemcpyHostToDevice);

    std::cout<<"Here";
    dim3 dimGrid((7000 + 64 -1)/64, (1000 + 64 -1)/64);
    dim3 dimBlock(64, 64);

    std::cout<<"HERE HERE HERE HERE";
    gemm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, *d_m, *d_n, *d_k);

    cudaMemcpy(c, d_c, 1000*7000* sizeof(float ), cudaMemcpyDeviceToHost);
    
    exampleClass<float> myClass(1<< 10);
    myClass.add(1024, 512, 1);
}