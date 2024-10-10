#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

void readInput(const char *filename, int &N, double **A, double **B) {
    std::ifstream infile(filename);
    infile >> N;

    *A = (double *)malloc(N * N * sizeof(double));
    *B = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            infile >> (*A)[i * N + j];

    for (int i = 0; i < N; ++i)
        infile >> (*B)[i];
    
    infile.close();
}

__global__ void luDecomposition(double *A, double *L, double *U, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        for (int j = 0; j < N; j++) {
            if (j < row) {
                L[row * N + j] = A[row * N + j]; // copy
                U[j * N + row] = 0; // U has zeros below diagonal
            } else {
                U[j * N + row] = A[row * N + j]; // copy what ?
                L[row * N + j] = (row == j) ? 1.0 : 0.0;
            }
        }
    }

    // Perform elimination
    for (int k = 0; k < N; k++) {
        if (row > k) {
            double factor = U[k * N + row] / U[k * N + k];
            for (int j = k; j < N; j++) {
                U[k * N + j] -= factor * U[row * N + j];
            }
            L[row * N + k] = factor;
        }
    }
}

__global__ void forwardSubstitution(double *L, double *B, double *Y, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        Y[row] = B[row];
        for (int j = 0; j < row; j++) {
            Y[row] -= L[row * N + j] * Y[j];
        }
    }
}

__global__ void backwardSubstitution(double *U, double *Y, double *X, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        X[row] = Y[row];
        for (int j = row + 1; j < N; j++) {
            X[row] -= U[row * N + j] * X[j];
        }
        X[row] /= U[row * N + row];
    }
}

int main() {
    int N;
    double *A, *B;
    readInput("input.txt", N, &A, &B);

    double *L, *U, *Y, *X;
    cudaMallocManaged(&L, N * N * sizeof(double));
    cudaMallocManaged(&U, N * N * sizeof(double));
    cudaMallocManaged(&Y, N * sizeof(double));
    cudaMallocManaged(&X, N * sizeof(double));

    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    luDecomposition<<<num_blocks, block_size>>>(A, L, U, N);
    forwardSubstitution<<<num_blocks, block_size>>>(L, B, Y, N);
    backwardSubstitution<<<num_blocks, block_size>>>(U, Y, X, N);

    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        printf("%f\n", X[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Call your LU decomposition kernel here
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("LU Decomposition time: %f ms\n", milliseconds);
    return 0;
}