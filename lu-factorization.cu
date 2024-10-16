#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

void readInput(const char *filename, int &N, double **A, double **B) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
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
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N) {
        for (int j = 0; j < N; j++) {
            if (j < row) {
                L[row * N + j] = A[row * N + j]; // L below diagonal
                U[row * N + j] = A[row * N + j]; // U has zeros below diagonal
            } else {
                U[row * N + j] = A[row * N + j]; // U above diagonal
                L[row * N + j] = (row == j) ? 1.0 : 0.0; // L diagonal elements
            }
        }
    }

    __syncthreads(); // Ensure all threads have updated L and U

    // Perform elimination
    for (int k = 0; k < N; k++) {
        if (row > k) {
            double factor = U[k * N + k] != 0 ? (U[row * N + k] / U[k * N + k]) : 0.0;
            for (int j = k; j < N; j++) {
                U[row * N + j] -= factor * U[k * N + j];
            }
            L[row * N + k] = factor;
        }
    }
}


__global__ void forwardSubstitution(double *L, double *B, double *Y, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    extern __shared__ double shared[];
    double *L_shared = (double *)shared;
    double *B_shared = (double *)&L_shared[N * N];
    double *Y_shared = (double *)&B_shared[N];
    if (row < N) {
        for (int j = 0; j < N; j++) {
            L_shared[row * N + j] = L[row * N + j];
        }
        B_shared[row] = B[row];
        Y_shared[row] = B_shared[row];
        for (int j = 0; j < row; j++) {
            Y_shared[row] -= L_shared[row * N + j] * Y_shared[j];
        }
        for (int j = 0; j < N; j++) {
            Y[row] = Y_shared[row];
        }
    }
}

__global__ void backwardSubstitution(double *U, double *Y, double *X, int N) {
    // Calculate the row index in the reversed order
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    extern __shared__ double shared[];
    double *U_shared = (double *)shared;
    double *Y_shared = (double *)&U_shared[N * N];
    double *X_shared = (double *)&Y_shared[N];
    
    // Iterate through the rows in reverse order
    // for (int r = N - 1 - row; r >= 0; r -= blockDim.x) {
    //     if (r >= 0) {
    //         X[r] = Y[r]; // Start with Y
    //         for (int j = r + 1; j < N; j++) {
    //             X[r] -= U[r * N + j] * X[j]; // Update X[r]
    //         }
    //         X[r] /= U[r * N + r]; // Normalize
    //     }
    // }
    // my code
    if (row < N) {
    //   printf("row = %d\n", row);
        row = N - row -1;
        for (int j = 0; j < N; j++) {
            U_shared[row * N + j] = U[row * N + j];
        }
        Y_shared[row] = Y[row];
        X_shared[row] = Y_shared[row];
        for (int j = N - 1; j >= row + 1; j--) {
            // printf("X[%d] used = %f\n", j, X[j]);
            X_shared[row] -= U_shared[row * N + j] * X[j];
        }
        X_shared[row] /= U_shared[row * N + row];
        for (int j = 0; j < N; j++) {
            X[row] = X_shared[row];
        }
        // printf("X[%d] = %f\n", row, X[row]);
    }
}

int main() {
    int N;
    double *A, *B, *X;
    readInput("input10.txt", N, &A, &B);
    X = (double *)malloc(N * sizeof(double));

    // print N, A, B
    printf("N: %d\n", N);
    printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("B:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", B[i]);
    }

    double *d_A, *d_B, *d_L, *d_U, *d_Y, *d_X;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));
    cudaMalloc(&d_L, N * N * sizeof(double));
    cudaMalloc(&d_U, N * N * sizeof(double));
    cudaMalloc(&d_Y, N * sizeof(double));
    cudaMalloc(&d_X, N * sizeof(double));

    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 gridConfig(1, 1, 1);
    dim3 blockConfig(1, N, 1);

    int sharedMemSize = N * N * sizeof(double) + N * sizeof(double) + N * sizeof(double);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    luDecomposition<<<gridConfig, blockConfig>>>(d_A, d_L, d_U, N);


    double* L = (double*)malloc(N * N * sizeof(double));
    double* U = (double*)malloc(N * N * sizeof(double));

    cudaMemcpy(L, d_L, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // print L and U
    printf("L:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", L[i * N + j]);
        }
        printf("\n");
    }
    printf("U:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", U[i * N + j]);
        }
        printf("\n");
    }
    
    forwardSubstitution<<<gridConfig, blockConfig, sharedMemSize>>>(d_L, d_B, d_Y, N);

    double* Y = (double*)malloc(N * sizeof(double));

    cudaMemcpy(Y, d_Y, N * sizeof(double), cudaMemcpyDeviceToHost);

    // print Y
    printf("Y:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", Y[i]);
    }
    
    backwardSubstitution<<<gridConfig, blockConfig, sharedMemSize>>>(d_U, d_Y, d_X, N);

    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    cudaMemcpy(X, d_X, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("X:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", X[i]);
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("LU Decomposition time: %f ms\n", milliseconds);


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_Y);
    cudaFree(d_X);

    // Free host memory
    free(A);
    free(B);
    free(X);
    return 0;
}

