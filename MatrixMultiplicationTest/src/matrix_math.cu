#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
//#include "utilityCore.hpp"
#include "matrix_math.h"
#include "cublas_v2.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAError(const char *msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


/*****************
 * Configuration *
 *****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 1024
#define TILE_WIDTH 16
#define ThreadColumn 2


/***********************************************
 * Kernel state  *
 ***********************************************/

dim3 threadsPerBlock(blockSize);
dim3 fullBlocksPerGrid((25 + blockSize - 1) / blockSize);

float *dev_A;
float *dev_B;
float *dev_C;


/******************
 * init *
 ******************/


/**
 * Initialize memory, update some globals
 */
void Matrix_Math::init(int dim) {
    //dim3 fullBlocksPerGrid(1);

    cudaMalloc((void**)&dev_A, dim*dim * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

    cudaMalloc((void**)&dev_B, dim*dim * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

    cudaMalloc((void**)&dev_C, dim*dim * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_C failed!");

//    cudaMemcpy(dev_A, hst_A, 25 * sizeof(float), cudaMemcpyHostToDevice);
//    checkCUDAErrorWithLine("cudaMemcpy hst_A to dev_A failed!");
//
//    cudaMemcpy(dev_B, hst_B, 25 * sizeof(float), cudaMemcpyHostToDevice);
//    checkCUDAErrorWithLine("cudaMemcpy hst_B to dev_B failed!");

    //cudaThreadSynchronize();
}

/******************
 * Matrix_Math *
 ******************/

__global__ void mat_mul(float* Md, float* Nd, float* Pd, int Width)
{
    int x = threadIdx.x+blockIdx.x*blockDim.x;
    int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(x>Width || y>Width)
    	return;

	float Pvalue = 0;
	for (int k = 0; k < Width; ++k)
		Pvalue+=Md[y * Width + k]*Nd[k * Width + x];
	Pd[y*Width + x] = Pvalue;
}

// block shared memory
__global__ void mat_mul_new(float * A, float * B, float * C,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

// replace for loop
__global__ void mat_mul_new1(float * A, float * B, float * C,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    if(Row > numARows || Col > numBColumns)
    	return;
    float Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][0] * ds_N[0][tx] + ds_M[ty][1] * ds_N[1][tx] +
          ds_M[ty][2] * ds_N[2][tx] + ds_M[ty][3] * ds_N[3][tx] +
          ds_M[ty][4] * ds_N[4][tx] + ds_M[ty][5] * ds_N[5][tx] +
          ds_M[ty][6] * ds_N[6][tx] + ds_M[ty][7] * ds_N[7][tx] +
          ds_M[ty][8] * ds_N[8][tx] + ds_M[ty][9] * ds_N[9][tx] +
          ds_M[ty][10] * ds_N[10][tx] + ds_M[ty][11] * ds_N[11][tx] +
          ds_M[ty][12] * ds_N[12][tx] + ds_M[ty][13] * ds_N[13][tx] +
          ds_M[ty][14] * ds_N[14][tx] + ds_M[ty][15] * ds_N[15][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

//block + thread
__global__ void mat_mul_new2(float * A, float * B, float * C,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    __shared__ float ds_M[TILE_WIDTH][ThreadColumn*TILE_WIDTH];
    __shared__ float ds_N[ThreadColumn*TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       //Row = by * TILE_WIDTH + ty,
       //Col = bx * TILE_WIDTH + tx;
       Row = by*TILE_WIDTH + ty,
       Col = bx*ThreadColumn*TILE_WIDTH + tx;
    float Pvalue1 = 0;
    float Pvalue2 = 0;

    for (int m = 0; m < ((numAColumns-1)/TILE_WIDTH+1)/2; ++m) {
    	for(int k=0;k<ThreadColumn; ++k)
          ds_M[ty][tx+k*TILE_WIDTH] = A[Row*numAColumns + ThreadColumn*m*TILE_WIDTH+tx + k*TILE_WIDTH];
    		//ds_M[ty][tx+k*TILE_WIDTH] = A[Row + (ThreadColumn*m*TILE_WIDTH+tx + k*TILE_WIDTH)*numAColumns];
    	for(int h=0;h<ThreadColumn; ++h)
    		for(int k=0; k<ThreadColumn; ++k)
    			ds_N[ty+h*TILE_WIDTH][tx+k*TILE_WIDTH] = B[(m*TILE_WIDTH*ThreadColumn+ty + h*TILE_WIDTH)*numBColumns + (Col + k*TILE_WIDTH)];
    			//ds_N[ty+h*TILE_WIDTH][tx+k*TILE_WIDTH] = B[(m*TILE_WIDTH*ThreadColumn+ty + h*TILE_WIDTH) + (Col + k*TILE_WIDTH)*numBColumns];
       __syncthreads();
       for(int h=0;h<ThreadColumn*TILE_WIDTH;++h)
    	   Pvalue1 +=ds_M[ty][h] * ds_N[h][tx];
       for(int k=0;k<ThreadColumn*TILE_WIDTH;++k)
    	   Pvalue2 +=ds_M[ty][k] * ds_N[k][tx+TILE_WIDTH];
       __syncthreads();
    }

     C[Row*numCColumns+Col] = Pvalue1;
     C[Row*numCColumns+Col+TILE_WIDTH] = Pvalue2;
    //C[Row+Col*numCColumns] = Pvalue1;
    //C[Row+(Col+TILE_WIDTH)*numCColumns] = Pvalue2;
}

//block + thread
__global__ void mat_mul_new3(float* Md, float* Nd, float* Pd, int Width)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float Pvalue1 = 0, Pvalue2=0;

	for (int m = 0; m < gridDim.x; ++m) {
		__shared__ float Mds[TILE_WIDTH][ThreadColumn*TILE_WIDTH];
		__shared__ float Nds[ThreadColumn*TILE_WIDTH][ThreadColumn*TILE_WIDTH];
		for(int k=0;k<ThreadColumn; ++k)
			Mds[ty][tx+k*TILE_WIDTH] = *(Md + (by*blockDim.y + ty) * Width +  ThreadColumn*m*blockDim.x + tx + k*TILE_WIDTH );
		for(int h=0;h<ThreadColumn; ++h)
			for(int k=0; k<ThreadColumn; ++k)
				Nds[ty+h*TILE_WIDTH][tx+k*TILE_WIDTH] = *(Nd + (m*blockDim.y*ThreadColumn + ty + h*TILE_WIDTH) * Width  + bx*blockDim.x*ThreadColumn + tx + k*TILE_WIDTH);
		__syncthreads();
		for(int h=0;h<ThreadColumn*TILE_WIDTH;++h)
			Pvalue1 +=Mds[ty][h] * Nds[h][tx];
		for(int k=0;k<ThreadColumn*TILE_WIDTH;++k)
			Pvalue2 +=Mds[ty][k] * Nds[k][tx+TILE_WIDTH];
		__syncthreads();
	}

	Pd[(by*blockDim.y + ty) * Width +  bx*ThreadColumn*blockDim.x + tx] = Pvalue1;
	Pd[(by*blockDim.y + ty) * Width +  bx*ThreadColumn*blockDim.x + tx + TILE_WIDTH] = Pvalue2;
}

//block + thread + expand for loop
__global__ void mat_mul_new4(float* Md, float* Nd, float* Pd, int Width)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float Pvalue1 = 0, Pvalue2=0;

	for (int m = 0; m < gridDim.x; ++m) {
		__shared__ float Mds[TILE_WIDTH][ThreadColumn*TILE_WIDTH];
		__shared__ float Nds[ThreadColumn*TILE_WIDTH][ThreadColumn*TILE_WIDTH];
		for(int k=0;k<ThreadColumn; ++k)
			Mds[ty][tx+k*TILE_WIDTH] = *(Md + (by*blockDim.y + ty) * Width +  ThreadColumn*m*blockDim.x + tx + k*TILE_WIDTH );
		for(int h=0;h<ThreadColumn; ++h)
			for(int k=0; k<ThreadColumn; ++k)
				Nds[ty+h*TILE_WIDTH][tx+k*TILE_WIDTH] = *(Nd + (m*blockDim.y*ThreadColumn + ty + h*TILE_WIDTH) * Width  +
				  bx*blockDim.x*ThreadColumn + tx + k*TILE_WIDTH);
		__syncthreads();
			Pvalue1 += Mds[ty][0] * Nds[0][tx] + Mds[ty][1] * Nds[1][tx] +
				Mds[ty][2] * Nds[2][tx] + Mds[ty][3] * Nds[3][tx] +
				Mds[ty][4] * Nds[4][tx] + Mds[ty][5] * Nds[5][tx] +
				Mds[ty][6] * Nds[6][tx] + Mds[ty][7] * Nds[7][tx] +
				Mds[ty][8] * Nds[8][tx] + Mds[ty][9] * Nds[9][tx] +
				Mds[ty][10] * Nds[10][tx] + Mds[ty][11] * Nds[11][tx] +
				Mds[ty][12] * Nds[12][tx] + Mds[ty][13] * Nds[13][tx] +
				Mds[ty][14] * Nds[14][tx] + Mds[ty][15] * Nds[15][tx] +
				Mds[ty][16] * Nds[16][tx] + Mds[ty][17] * Nds[17][tx] +
				Mds[ty][18] * Nds[18][tx] + Mds[ty][19] * Nds[19][tx] +
				Mds[ty][20] * Nds[20][tx] + Mds[ty][21] * Nds[21][tx] +
				Mds[ty][22] * Nds[22][tx] + Mds[ty][23] * Nds[23][tx] +
				Mds[ty][24] * Nds[24][tx] + Mds[ty][25] * Nds[25][tx] +
				Mds[ty][26] * Nds[26][tx] + Mds[ty][27] * Nds[27][tx] +
				Mds[ty][28] * Nds[28][tx] + Mds[ty][29] * Nds[29][tx] +
				Mds[ty][30] * Nds[30][tx] + Mds[ty][31] * Nds[31][tx];

			Pvalue2 += Mds[ty][0] * Nds[0][tx+TILE_WIDTH] + Mds[ty][1] * Nds[1][tx+TILE_WIDTH] +
				Mds[ty][2] * Nds[2][tx+TILE_WIDTH] + Mds[ty][3] * Nds[3][tx+TILE_WIDTH] +
				Mds[ty][4] * Nds[4][tx+TILE_WIDTH] + Mds[ty][5] * Nds[5][tx+TILE_WIDTH] +
				Mds[ty][6] * Nds[6][tx+TILE_WIDTH] + Mds[ty][7] * Nds[7][tx+TILE_WIDTH] +
				Mds[ty][8] * Nds[8][tx+TILE_WIDTH] + Mds[ty][9] * Nds[9][tx+TILE_WIDTH] +
				Mds[ty][10] * Nds[10][tx+TILE_WIDTH] + Mds[ty][11] * Nds[11][tx+TILE_WIDTH] +
				Mds[ty][12] * Nds[12][tx+TILE_WIDTH] + Mds[ty][13] * Nds[13][tx+TILE_WIDTH] +
				Mds[ty][14] * Nds[14][tx+TILE_WIDTH] + Mds[ty][15] * Nds[15][tx+TILE_WIDTH] +
				Mds[ty][16] * Nds[16][tx+TILE_WIDTH] + Mds[ty][17] * Nds[17][tx+TILE_WIDTH] +
				Mds[ty][18] * Nds[18][tx+TILE_WIDTH] + Mds[ty][19] * Nds[19][tx+TILE_WIDTH] +
				Mds[ty][20] * Nds[20][tx+TILE_WIDTH] + Mds[ty][21] * Nds[21][tx+TILE_WIDTH] +
				Mds[ty][22] * Nds[22][tx+TILE_WIDTH] + Mds[ty][23] * Nds[23][tx+TILE_WIDTH] +
				Mds[ty][24] * Nds[24][tx+TILE_WIDTH] + Mds[ty][25] * Nds[25][tx+TILE_WIDTH] +
				Mds[ty][26] * Nds[26][tx+TILE_WIDTH] + Mds[ty][27] * Nds[27][tx+TILE_WIDTH] +
				Mds[ty][28] * Nds[28][tx+TILE_WIDTH] + Mds[ty][29] * Nds[29][tx+TILE_WIDTH] +
				Mds[ty][30] * Nds[30][tx+TILE_WIDTH] + Mds[ty][31] * Nds[31][tx+TILE_WIDTH];
		__syncthreads();
	}

	Pd[(by*blockDim.y + ty) * Width +  bx*ThreadColumn*blockDim.x + tx] = Pvalue1;
	Pd[(by*blockDim.y + ty) * Width +  bx*ThreadColumn*blockDim.x + tx + TILE_WIDTH] = Pvalue2;
}


// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
//void mat_mul_cublas(const float *A, const float *B, float *C, const int m, const int k, const int n) {
//	int lda=m,ldb=k,ldc=m;
//	const float alf = 1;
//	const float bet = 0;
//	const float *alpha = &alf;
//	const float *beta = &bet;
//
//	// Create a handle for CUBLAS
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//
//	// Do the actual multiplication
//	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//
//	// Destroy the handle
//	cublasDestroy(handle);
//}


/******************
 * Matrix_Math *
 ******************/

float Matrix_Math::mul(float* A, float* B, float* C, int dim){
    init(dim);
	cudaMemcpy(dev_A, A, dim*dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, dim*dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid((dim+TILE_WIDTH-1)/TILE_WIDTH, (dim+TILE_WIDTH-1)/TILE_WIDTH, 1);
	//dim3 dimGrid(dim/TILE_WIDTH+1, dim/TILE_WIDTH+1);

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_mul<<< dimGrid,dimBlock >>>(dev_A, dev_B, dev_C, dim);
	cudaEventRecord(stop);

	cudaMemcpy( C, dev_C, dim*dim * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}

float Matrix_Math::mul_new(float* A, float* B, float* C, int dim){
    init(dim);
	cudaMemcpy(dev_A, A, dim*dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, dim*dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid((dim-1)/TILE_WIDTH+1, (dim-1)/TILE_WIDTH+1, 1);
	//dim3 dimGrid(dim/TILE_WIDTH+1, dim/TILE_WIDTH+1);

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaThreadSynchronize();
	mat_mul_new<<< dimGrid,dimBlock >>>(dev_A, dev_B, dev_C, dim, dim, dim, dim, dim, dim);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy( C, dev_C, dim*dim * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}

float Matrix_Math::mul_new1(float* A, float* B, float* C, int dim){
    init(dim);
	cudaMemcpy(dev_A, A, dim*dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, dim*dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid((dim-1)/TILE_WIDTH+1, (dim-1)/TILE_WIDTH+1, 1);
	//dim3 dimGrid(dim/TILE_WIDTH+1, dim/TILE_WIDTH+1);

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaThreadSynchronize();
	mat_mul_new1<<< dimGrid,dimBlock >>>(dev_A, dev_B, dev_C, dim, dim, dim, dim, dim, dim);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy( C, dev_C, dim*dim * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}

float Matrix_Math::mul_new2(float* A, float* B, float* C, int dim){
    init(dim);
	cudaMemcpy(dev_A, A, dim*dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, dim*dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid(((dim-1)/TILE_WIDTH+1)/2, (dim-1)/TILE_WIDTH+1, 1);
	//dim3 dimGrid(dim/TILE_WIDTH/2, dim/TILE_WIDTH);

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaThreadSynchronize();
	mat_mul_new2<<< dimGrid,dimBlock >>>(dev_A, dev_B, dev_C, dim, dim, dim, dim, dim, dim);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy( C, dev_C, dim*dim * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}

float Matrix_Math::mul_new3(float* A, float* B, float* C, int dim){
    init(dim);
	cudaMemcpy(dev_A, A, dim*dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, dim*dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid((dim-1)/TILE_WIDTH+1, (dim-1)/TILE_WIDTH+1, 1);
	//dim3 dimGrid(dim/TILE_WIDTH, dim/TILE_WIDTH);

	//dimGrid.x/=ThreadColumn;

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaThreadSynchronize();
	mat_mul_new3<<< dimGrid,dimBlock >>>(dev_A, dev_B, dev_C, dim);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy( C, dev_C, dim*dim * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}

float Matrix_Math::mul_new4(float* A, float* B, float* C, int dim){
    init(dim);
	cudaMemcpy(dev_A, A, dim*dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, dim*dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	//dim3 dimGrid((dim-1)/TILE_WIDTH+1, (dim-1)/TILE_WIDTH+1, 1);
	dim3 dimGrid(dim/TILE_WIDTH, dim/TILE_WIDTH);

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaThreadSynchronize();
	mat_mul_new4<<< dimGrid,dimBlock >>>(dev_A, dev_B, dev_C, dim);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy( C, dev_C, dim*dim * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}

float Matrix_Math::mul_cublas(float* A, float* B, float* C, int dim){
    init(dim);
	cudaMemcpy(dev_A, A, dim*dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, dim*dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid((dim-1)/TILE_WIDTH+1, (dim-1)/TILE_WIDTH+1, 1);
	//dim3 dimGrid(dim/TILE_WIDTH+1, dim/TILE_WIDTH+1);

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	//mat_mul_cublas(dev_A, dev_B, dev_C, dim, dim, dim);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy( C, dev_C, dim*dim * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}
