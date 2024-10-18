#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <chrono>

std::chrono::duration<double> read_time(0);
std::chrono::duration<double> l_time(0);
std::chrono::duration<double> u_time(0);
std::chrono::duration<double> lu_decomposition_time(0);
std::chrono::duration<double> total_time(0);

#define TILE 16

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

// Row elimination Kernel
__global__ void elimination(double *A, double* L, double* U, int n, int index, int bsize) {
    int idThread = threadIdx.x;
    int idBlock = blockIdx.x;

    int pivotRow = (index * n);
    int currentRow = (((bsize * idBlock) + idThread) * n);
    int start = currentRow + index;
    int end = currentRow + n;

    if (currentRow > pivotRow && currentRow < n * n) {
        for (int i = currentRow; i < start + 1; i++) {
            U[i] = 0;
        }
        for (int i = start + 1; i < end; ++i) {
            U[i] = U[i] - (L[start] * U[pivotRow + (i - currentRow)]);
        }
    }
}

__global__ void scaleIndex(double* U, double *L, int n, int index) {
    int start = (index * n + index);
    int end = (index * n + n);
    L[start] = 1;
    for (int i = index + 1; i < n; ++i) {
        L[i * n + index] = (U[i * n + index] / U[start]);
    }

}

void forwardSubstitution(double* L, double* B, double* Y, int N) {
    for (int row = 0; row < N; row++) {
        Y[row] = B[row];
        for (int j = 0; j < row; j++) {
            Y[row] -= L[row * N + j] * Y[j];
        }
    }
}

void backwardSubstitution(double* U, double* Y, double* X, int N) {
    for (int row = N - 1; row >= 0; row--) {
        X[row] = Y[row];
        for (int j = row + 1; j < N; j++) {
            X[row] -= U[row * N + j] * X[j];
        }
        X[row] /= U[row * N + row];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./cuda " << "<input_file> " << "<output_file>" << std::endl;
        exit(EXIT_FAILURE);
    }
    int N;
    double *A, *B, *X, *Y;
    auto read_start = std::chrono::high_resolution_clock::now();
    readInput(argv[1], N, &A, &B);
    auto read_end = std::chrono::high_resolution_clock::now();
    read_time = read_end - read_start;

    X = (double *)malloc(N * sizeof(double));
    Y = (double *)malloc(N * sizeof(double));
    double *d_A, *d_L, *d_U;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_L, N * N * sizeof(double));
    cudaMalloc(&d_U, N * N * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, A, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // dim3 gridConfig((N/TILE) + ((N%TILE) ? 1 : 0), 1, 1);
    // dim3 blockConfig(TILE, 1, 1);
    int gridConfig = (N/TILE) + ((N%TILE) ? 1 : 0);
    int blockConfig = TILE;

    double* L = (double*)malloc(N * N * sizeof(double));
    double* U = (double*)malloc(N * N * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        scaleIndex<<<1,1>>>(d_U, d_L, N, i);
        elimination<<<gridConfig, TILE>>>(d_A, d_L, d_U, N, i, TILE);
    }

    cudaMemcpy(A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();

    lu_decomposition_time = end - start;

    auto start_sub = std::chrono::high_resolution_clock::now();
    forwardSubstitution(L, B, Y, N);
    backwardSubstitution(U, Y, X, N);
    auto end_sub = std::chrono::high_resolution_clock::now();

    total_time =  lu_decomposition_time + end_sub - start_sub;

    std::cout << "Read time: " << read_time.count() << "s" << std::endl;
    std::cout << "LU decomposition time: " << lu_decomposition_time.count() << "s" << std::endl;
    std::cout << "Total time: " << total_time.count() << "s" << std::endl;

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

    // Free host memory
    free(A);
    free(B);
    free(X);
    return 0;
}
