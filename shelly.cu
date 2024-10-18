#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>  // For setting precision
#include <chrono>   // For timing

#define IDX(i, j, N) ((i) * (N) + (j)) // Macro to compute row-major index

_global_ void kernel_LU_decomposition(double* A, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= k + 1 && i < N) {
        A[IDX(i, k, N)] = A[IDX(i, k, N)] / A[IDX(k, k, N)];  // Modify L

        if (j >= k + 1 && j < N) {
            A[IDX(i, j, N)] = A[IDX(i, j, N)] - A[IDX(i, k, N)] * A[IDX(k, j, N)]; // Modify U
        }
    }
}

void LU_decomposition(double* A, int N) {
    // Setting block and grid size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int k = 0; k < N; ++k) {
        kernel_LU_decomposition<<<numBlocks, threadsPerBlock>>>(A, N, k);
        cudaDeviceSynchronize();  // Synchronize after each step
    }
}

// Forward substitution to solve L * Y = B
void forward_substitution(const double* L, const double* B, double* Y, int N) {
    for (int i = 0; i < N; ++i) {
        Y[i] = B[i];  // Start with B[i]
        for (int j = 0; j < i; ++j) {
            Y[i] -= L[IDX(i, j, N)] * Y[j];  // Subtract L(i,j) * Y(j)
        }
    }
}

// Backward substitution to solve U * X = Y
void backward_substitution(const double* U, const double* Y, double* X, int N) {
    for (int i = N - 1; i >= 0; --i) {
        X[i] = Y[i];  // Start with Y[i]
        for (int j = i + 1; j < N; ++j) {
            X[i] -= U[IDX(i, j, N)] * X[j];  // Subtract U(i,j) * X(j)
        }
        X[i] /= U[IDX(i, i, N)];  // Divide by U(i,i)
    }
}

int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <timing_file>" << std::endl;
        return -1;
    }

    std::ifstream inputFile(argv[1]);
    std::ofstream outputFile(argv[2]);
    std::ofstream timingFile(argv[3]);

    if (!inputFile || !outputFile || !timingFile) {
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }

    // Start timing for file reading
    auto start_read = std::chrono::high_resolution_clock::now();

    int N;
    // Reading the size of the system N
    inputFile >> N;

    // Allocating memory for matrix A (N x N)
    double* A = (double*)malloc(N * N * sizeof(double));
    std::vector<double> B(N); // Column matrix B

    // Reading matrix A in row-major order from the input file
    for (int i = 0; i < N * N; ++i) {
        inputFile >> A[i];
    }

    // Reading vector B
    for (int i = 0; i < N; ++i) {
        inputFile >> B[i];
    }

    // End timing for file reading
    auto end_read = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> read_duration = end_read - start_read;

    // Allocating device memory
    double* d_A;
    cudaMalloc(&d_A, N * N * sizeof(double));

    // Copying matrix A to device
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Start timing for LU decomposition
    auto start_lu = std::chrono::high_resolution_clock::now();

    // Performing LU decomposition
    LU_decomposition(d_A, N);

    // End timing for LU decomposition
    auto end_lu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> lu_duration = end_lu - start_lu;

    // Copying the result back to host
    cudaMemcpy(A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Allocate memory for solution vector X and intermediate Y
    std::vector<double> X(N), Y(N);

    // Start timing for solving system of equations
    auto start_solve = std::chrono::high_resolution_clock::now();

    // Forward and backward substitution
    forward_substitution(A, B.data(), Y.data(), N);  // Solve LY = B
    backward_substitution(A, Y.data(), X.data(), N); // Solve UX = Y

    // End timing for solving system of equations
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_duration = end_solve - start_solve;

    // Writing output to file with double precision
    outputFile << N << std::endl;

    outputFile << std::fixed << std::setprecision(17); // Set precision for floating point numbers to 17 digits

    // Writing L matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                outputFile << 1.0 << " ";  // Diagonal elements of L are 1
            } else if (i > j) {
                outputFile << A[IDX(i, j, N)] << " ";  // Below diagonal (L)
            } else {
                outputFile << 0.0 << " ";  // Above diagonal is 0 in L
            }
        }
        outputFile << std::endl;
    }

    // Writing U matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i <= j) {
                outputFile << A[IDX(i, j, N)] << " ";  // On or above diagonal (U)
            } else {
                outputFile << 0.0 << " ";  // Below diagonal is 0 in U
            }
        }
        outputFile << std::endl;
    }

    // Writing solution vector X
    for (int i = 0; i < N; ++i) {
        outputFile << X[i] << std::endl;
    }

    // Writing timing information to the timing file
    timingFile << std::fixed << std::setprecision(17);
    timingFile << "Time taken to read A and B matrices: " << read_duration.count() << " seconds" << std::endl;
    timingFile << "Time taken in computing lower and upper triangular matrices: " << lu_duration.count() << " seconds" << std::endl;
    timingFile << "Time taken to solve the system of equations: " << solve_duration.count() << " seconds" << std::endl;
    timingFile << "Total time taken: " << (read_duration + lu_duration + solve_duration).count() << " seconds" << std::endl;

    // Freeing device and host memory
    cudaFree(d_A);
    free(A);

    // Close files
    inputFile.close();
    outputFile.close();
    timingFile.close();

    return 0;
}