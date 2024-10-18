#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <chrono>

std::chrono::duration<double> read_time(0);
std::chrono::duration<double> l_time(0);
std::chrono::duration<double> u_time(0);
std::chrono::duration<double> total_time(0);

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

    auto start = std::chrono::high_resolution_clock::now();
    if (row < N) {
        for (int j = 0; j < N; j++) {
            if (j < row) {
                L[row * N + j] = A[row * N + j]; // L below diagonal
                U[row * N + j] = A[row * N + j]; // U below diagonal
            } else {
                U[row * N + j] = A[row * N + j]; // U above diagonal
                L[row * N + j] = (row == j) ? 1.0 : 0.0; // L diagonal elements
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    l_time += end - start;
    u_time += end - start;

    __syncthreads(); // Ensure all threads have updated L and U

    // Perform elimination
    for (int k = 0; k < N; k++) {
        if (row > k) {
            start = std::chrono::high_resolution_clock::now();
            double factor = U[k * N + k] != 0 ? (U[row * N + k] / U[k * N + k]) : 0.0;
            U[row * N + k] = 0.0;
            L[row * N + k] = factor;
            end = std::chrono::high_resolution_clock::now();
            l_time += end - start;
            for (int j = k + 1; j < N; j++) {
                U[row * N + j] -= factor * U[k * N + j];
            }
            end = std::chrono::high_resolution_clock::now();
            u_time += end - start;
        }
    }
}


void forwardSubstitution(double* A, double* B, double* Y, int N) {
    for (int row = 0; row < N; row++) {
        Y[row] = B[row];
        for (int j = 0; j < row; j++) {
            Y[row] -= A[row * N + j] * Y[j];
        }
    }
}

void backwardSubstitution(double* A, double* B, double* X, int N) {
    for (int row = N - 1; row >= 0; row--) {
        X[row] = B[row];
        for (int j = row + 1; j < N; j++) {
            X[row] -= A[row * N + j] * X[j];
        }
        X[row] /= A[row * N + row];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./main.cu " << "<input_file>" << "<output_file>" << std::endl;
        exit(EXIT_FAILURE);
    }
    int N;
    double *A, *B, *X;
    auto read_start = std::chrono::high_resolution_clock::now();
    readInput(argv[1], N, &A, &B);
    auto read_end = std::chrono::high_resolution_clock::now();
    read_time = read_end - read_start;

    X = (double *)malloc(N * sizeof(double));
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

    // int sharedMemSize = N * N * sizeof(double) + N * sizeof(double) + N * sizeof(double);

    auto start = std::chrono::high_resolution_clock::now();
    luDecomposition<<<gridConfig, blockConfig>>>(d_A, d_L, d_U, N);
    auto end = std::chrono::high_resolution_clock::now();

    auto lu_decomposition_time = end - start;

    double* L = (double*)malloc(N * N * sizeof(double));
    double* U = (double*)malloc(N * N * sizeof(double));

    cudaMemcpy(L, d_L, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

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
    
    auto start_sub = std::chrono::high_resolution_clock::now();
    forwardSubstitution(L, B, X, N);
    backwardSubstitution(U, X, X, N);
    auto end_sub = std::chrono::high_resolution_clock::now();

    total_time =  lu_decomposition_time + end_sub - start_sub;

    printf("Read time: %f ms\n", read_time.count());
    printf("L time: %f ms\n", l_time.count());
    printf("U time: %f ms\n", u_time.count());
    printf("Total time: %f ms\n", total_time.count());

    printf("X:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", X[i]);
    }

    std::ofstream outfile(argv[2]);
    if (!outfile) {
        std::cerr << "Error opening file for writing: "<< argv[2] << std::endl;
        exit(EXIT_FAILURE);
    }

    outfile << N << std::endl;

    // Write L matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            outfile << L[i * N + j] << " ";
        }
        outfile << std::endl;
    }

    // Write U matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            outfile << U[i * N + j] << " ";
        }
        outfile << std::endl;
    }

    // Write solution vector X
    for (int i = 0; i < N; i++) {
        outfile << X[i] << std::endl;
    }

    outfile.close();

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