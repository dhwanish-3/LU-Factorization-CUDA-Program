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

#define TILE 8

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

void writeToFile(const char* filename, int N, double* L, double* U, double* X) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file for writing: "<< filename << std::endl;
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

    // Write X
    for (int i = 0; i < N; i++) {
        outfile << X[i] << std::endl;
    }

    outfile.close();
}

// Row elimination Kernel
__global__ void rowElimination(double* L, double* U, int N, int index) {
    int pivotRow = index * N;
    int currentRow = (blockDim.x * blockIdx.x + threadIdx.x) * N;
    int start = currentRow + index;
    int end = currentRow + N;

    extern __shared__ double Us[];
    __shared__ double pivot;
    pivot = L[start];
    for (int i = 0; i < N; i++) {
        Us[i] = U[pivotRow + i];
    }

    if (currentRow > pivotRow && currentRow < N * N) {
        for (int i = start + 1; i < end; ++i) {
            U[i] = U[i] - (pivot * Us[i - currentRow]);
        }
    }
}

// kernel without shared memory
// __global__ void rowElimination(double* L, double* U, int N, int index, int bsize) {
//     int pivotRow = index * N;
//     int currentRow = ((TILE * blockIdx.x) + threadIdx.x) * N;
//     int start = currentRow + index;
//     int end = currentRow + N;

//     if (currentRow > pivotRow && currentRow < N * N) {
//         for (int i = currentRow; i < start + 1; i++) {
//             U[i] = 0;
//         }
//         for (int i = start + 1; i < end; ++i) {
//             U[i] = U[i] - (L[start] * U[pivotRow + (i - currentRow)]);
//         }
//     }
// }

// kernel without shared memory
// __global__ void computeL(double* U, double *L, int N, int index) {
//     int id = index + threadIdx.x + 1;
//     int start = (index * N + index);
//     L[start] = 1; // diagonal elements of L
//     if (id < N) {
//         L[id * N + index] = (U[id * N + index] / U[start]);
//     }
// }

__global__ void computeL(double* U, double *L, int N, int index) {
    int id = index + threadIdx.x + 1;
    int start = (index * N + index);

    __shared__ double pivot;
    pivot = U[start];
    if (id < N) {
        L[id * N + index] = U[id * N + index] / pivot;
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
    if (argc < 4) {
        std::cerr << "Usage: ./cuda " << "<input_file> <output_file> <timing_file>" << std::endl;
        exit(EXIT_FAILURE);
    }
    int N;
    double *A, *B;
    auto read_start = std::chrono::high_resolution_clock::now();
    readInput(argv[1], N, &A, &B);
    auto read_end = std::chrono::high_resolution_clock::now();
    read_time = read_end - read_start;

    double *d_L, *d_U;
    cudaMalloc(&d_L, N * N * sizeof(double));
    cudaMalloc(&d_U, N * N * sizeof(double));
    cudaMemcpy(d_U, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    
    double* X = (double *)malloc(N * sizeof(double));
    double* Y = (double *)malloc(N * sizeof(double));
    double* L = (double*)malloc(N * N * sizeof(double));
    double* U = (double*)malloc(N * N * sizeof(double));
    
    cudaEvent_t startLU, stopLU;
    cudaEventCreate(&startLU);
    cudaEventCreate(&stopLU);
    for (int i = 0; i < N; ++i) {
        cudaEventRecord(startLU);
        
        computeL<<<1, N - i - 1>>>(d_U, d_L, N, i);
        
        cudaEventRecord(stopLU);
        cudaEventSynchronize(stopLU);
        float l1_time = 0;
        cudaEventElapsedTime(&l1_time,startLU, stopLU);
        l_time += std::chrono::duration<double>(l1_time/1000);
        
        int sharedMem = N * sizeof(double);
        cudaDeviceSynchronize();
        cudaEventRecord(startLU);
        
        dim3 gridConfig(((N)/TILE) + ((N%TILE) ? 1 : 0), 1, 1);
        dim3 blockConfig(TILE, 1, 1);
        rowElimination<<<gridConfig, blockConfig, sharedMem>>>(d_L, d_U, N, i);

        cudaEventRecord(stopLU);
        cudaEventSynchronize(stopLU);
        float u1_time = 0;
        cudaEventElapsedTime(&u1_time,startLU, stopLU);
        u_time += std::chrono::duration<double>(u1_time/1000);
    }

    cudaMemcpy(L, d_L, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // removing unnecessary elements from L and U
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                L[i * N + j] = 1;
            }
            if (i < j) {
                L[i * N + j] = 0;
            }
            if (i > j) {
                U[i * N + j] = 0;
            }
        }
    }
    
    auto start_sub = std::chrono::high_resolution_clock::now();
    forwardSubstitution(L, B, Y, N);
    backwardSubstitution(U, Y, X, N);
    auto end_sub = std::chrono::high_resolution_clock::now();

    lu_decomposition_time = l_time + u_time;
    total_time = read_time + l_time + u_time + end_sub - start_sub;

    // write timing to file
    std::ofstream timingFile(argv[3]);
    if (!timingFile) {
        std::cerr << "Error opening file for writing: " << argv[3] << std::endl;
        exit(EXIT_FAILURE);
    }
    timingFile << "Read time: " << read_time.count() << "s" << std::endl;
    timingFile << "L time: " << l_time.count() << "s" << std::endl;
    timingFile << "U time: " << u_time.count() << "s" << std::endl;
    timingFile << "LU decomposition time: " << lu_decomposition_time.count() << "s" << std::endl;
    timingFile << "Total time: " << total_time.count() << "s" << std::endl;
    timingFile.close();

    // write output to file
    writeToFile(argv[2], N, L, U, X);

    // Free device memory
    cudaFree(d_L);
    cudaFree(d_U);
    // Free host memory
    free(A);
    free(B);
    free(X);
    return 0;
}
