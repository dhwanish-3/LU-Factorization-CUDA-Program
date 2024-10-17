__global__ void forwardSubstitution(double *L, double *B, double *Y, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    extern __shared__ double shared[];
    double *L_shared = (double *)shared;
    double *B_shared = (double *)&L_shared[N * N];
    double *Y_shared = (double *)&B_shared[N];
    if (row < N) {
        printf("row = %d\n", row);
        for (int j = 0; j < N; j++) {
            L_shared[row * N + j] = L[row * N + j];
        }
        B_shared[row] = B[row];
        Y_shared[row] = B_shared[row];
        for (int j = 0; j < row; j++) {
            printf("Y-shared[%d] used = %f\n", j, Y_shared[j]);
            Y_shared[row] -= L_shared[row * N + j] * Y_shared[j];
        }
        Y[row] = Y_shared[row];
        printf("Y[%d] = %f\n", row, Y[row]);
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
        printf("row = %d\n", row);
        row = N - row -1;
        for (int j = 0; j < N; j++) {
            U_shared[row * N + j] = U[row * N + j];
        }
        Y_shared[row] = Y[row];
        X_shared[row] = Y_shared[row];
        for (int j = N - 1; j >= row + 1; j--) {
            printf("X-shared[%d] used = %f\n", j, X_shared[j]);
            X_shared[row] -= U_shared[row * N + j] * X_shared[j];
        }
        X_shared[row] /= U_shared[row * N + row];
        X[row] = X_shared[row];
        printf("X[%d] = %f\n", row, X[row]);
    }
}