#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

void readInput(int& N, double** A, double** B) {

}

__global__ void matrix_addition(int N, double* A, double* B, double* sol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int index = row * N + col;

    sol[index] = A[index] + B[index];
}

int main() {
    int N = 2;
    double* A, *dA;
    double* B, *dB;
    double* sol, *dsol;

    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    sol = (double*)malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    cudaMalloc(&dA, N * N * sizeof(double));
    cudaMalloc(&dB, N * N * sizeof(double));
    cudaMalloc(&dsol, N * N * sizeof(double));

    cudaMemcpy(dA, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 1;
    int num_blocks = 4;

    matrix_addition<<<num_blocks, block_size>>>(N, dA, dB, dsol);

    cudaMemcpy(sol, dsol, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", dsol[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}