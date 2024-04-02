#include <stdio.h>
#include <cuda_runtime.h>

#define N 4

// kernel that computes matrix multiplication
// inputs:
//	a -> pointer to first matrix (operand)
//	b -> pointer to second matrix (operand)
//	c -> pointer to result matrix
//	n -> size of the square matrix
__global__ void matrix_mul(int *a, int *b, int *c, int n) {
    //every thread will compute one index of the resultant 2D matrix
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    for (int k = 0; k < n; k++)
        sum += a[i * n + k] * b[k * n + j]; // get the sum of the multiplication of row i and column j
    c[i * n + j] = sum;	//place the result at position [i,j]
}

void print_matrix(int* m, int n, char* title){
    printf("%s\n", title);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%d\t", m[i * n + j]);
        printf("\n");
    }	
    printf("\n");
}

int main() {
    int n = N;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = n * n * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // define matrixes (serialized for convenience)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i * j;
            b[i * n + j] = i + 2*j;
        }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(N, N);   // define thread distribution inside block as a 2D array (NxN)
    dim3 gridSize(1, 1);	//define how many blocks, for these small matrix only 1 block will be used
    matrix_mul<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    char a_title[] = "A:";
    print_matrix(a, n, a_title);
    
    char b_title[] = "B:";
    print_matrix(b, n, b_title);
    
    char c_title[] = "C:";
    print_matrix(c, n, c_title);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
