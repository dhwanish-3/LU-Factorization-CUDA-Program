// CPU implementation of LU decomposition to verify the correctness of the GPU implementation

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
void luDecomposition(double* A, double* L, double* U, int N) {
    for (int i = 0; i < N; i++) {
        // Upper triangular matrix U
        for (int k = i; k < N; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += (L[i * N + j] * U[j * N + k]);
            }
            U[i * N + k] = A[i * N + k] - sum;
        }

        // Lower triangular matrix L
        for (int k = i; k < N; k++) {
            if (i == k) {
                L[i * N + i] = 1.0;  // Diagonal elements of L are 1
            } else {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += (L[k * N + j] * U[j * N + i]);
                }
                L[k * N + i] = (A[k * N + i] - sum) / U[i * N + i];
            }
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
        std::cerr << "Usage: ./main " << "<input_file>" << "<output_file>" << std::endl;
        exit(EXIT_FAILURE);
    }
    int N;
    double *A, *B, *X;
    readInput(argv[1], N, &A, &B);

    X = (double *)malloc(N * sizeof(double));
    double* L = (double*)malloc(N * N * sizeof(double));
    double* U = (double*)malloc(N * N * sizeof(double));

    luDecomposition(A, L, U, N);
    forwardSubstitution(L, B, X, N);
    backwardSubstitution(U, X, X, N);


    std::ofstream outfile(argv[2]);
    if (!outfile) {
        std::cerr << "Error opening file for writing: "<< argv[2] << std::endl;
        exit(EXIT_FAILURE);
    }

    outfile << N << std::endl;

    // Write L matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i >= j) outfile << L[i * N + j] << " ";
            else outfile << "0 ";
        }
        outfile << std::endl;
    }

    // Write U matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= j) outfile << U[i * N + j] << " ";
            else outfile << "0 ";
        }
        outfile << std::endl;
    }

    // Write solution vector X
    for (int i = 0; i < N; i++) {
        outfile << X[i] << std::endl;
    }

    outfile.close();


    // Free host memory
    free(A);
    free(B);
    free(X);
    return 0;
}