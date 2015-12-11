#include "matrix_math.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
//#include "mkl.h"
#include <ctime>

using namespace std;

void printMatrix(float* mat, int row_dim, int col_dim) {
	int i = 0;
	for(int row = 0; row < row_dim; row++){
		for(int col = 0; col < col_dim; col++){
			i = (row * col_dim) + col;
			printf("%6.4f\t", mat[i]);
		}
		printf("\n");
	}
}

void MatrixMul(const float *A, const float *B, float *C, int Width) {
    int i, j, k;
    for(i=0; i<Width; i++)
		for(j=0; j<Width; j++){
			float s=0;
			for(k=0; k<Width; k++)
				s+=A[i*Width+k]*B[k*Width+j];
			C[i*Width+j]=s;
		}
}

int main(){
	float timeMul;

	int m = 5;
	int n = 5;
	int dim =256;

	float *hst_A = new float[dim * dim];
	float *hst_B = new float[dim * dim];
	float *hst_C = new float[dim * dim];
	float *hst_C_ref = new float[dim * dim];
	float ref_norm;
	float error_norm;

	for(int i = 0; i < dim*dim; i++){
		hst_A[i]=rand()/(float)RAND_MAX;
		hst_B[i]=rand()/(float)RAND_MAX;
	}

	cout << "Matrix A:" << endl;
	//printMatrix(hst_A, dim, dim);

	cout << "Matrix B:" << endl;
	//printMatrix(hst_B, dim, dim);

	std::clock_t start;
	double duration;
	start = std::clock();
	MatrixMul(hst_A, hst_B, hst_C_ref, dim);
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC * (double)1000;
	cout << "A * B should be:" << endl;
	//printMatrix(hst_C_ref, dim, dim);
	printf("Time: %.4f ms \n", duration);

	//cblas_dgemm(dim, 0, 0, dim, dim, dim, 1.0, hst_A, dim, hst_B, dim, 0.0, hst_C, dim);

	cout << "A * B:" << endl;
	timeMul = Matrix_Math::mul(hst_A, hst_B, hst_C, dim);
	//printMatrix(hst_C, dim, dim);
	printf("Time: %.4f ms \n", timeMul);

	error_norm=0;
	ref_norm=0;
	for(int i = 0; i < dim*dim; i++){
		float diff=hst_C_ref[i]-hst_C[i];
		error_norm+=diff*diff;
		ref_norm+=hst_C_ref[i]*hst_C_ref[i];
	}
	printf("Test %s \n", (sqrtf(error_norm/ref_norm)<1E-6) ? "PASSED" : "FAILED");
	printf("%f \n", sqrtf(error_norm/ref_norm));

	cout << "A * B new :" << endl;
	timeMul = Matrix_Math::mul_new(hst_A, hst_B, hst_C, dim);
	//printMatrix(hst_C, dim, dim);
	printf("Time: %.4f ms \n", timeMul);

	error_norm=0;
	ref_norm=0;
	for(int i = 0; i < dim*dim; i++){
		float diff=hst_C_ref[i]-hst_C[i];
		error_norm+=diff*diff;
		ref_norm+=hst_C_ref[i]*hst_C_ref[i];
	}
	printf("Test %s \n", (sqrtf(error_norm/ref_norm)<1E-6) ? "PASSED" : "FAILED");
	printf("%f \n", sqrtf(error_norm/ref_norm));


	cout << "A * B new 2:" << endl;
	timeMul = Matrix_Math::mul_new2(hst_A, hst_B, hst_C, dim);
	//printMatrix(hst_C, dim, dim);
	printf("Time: %.4f ms \n", timeMul);

	error_norm=0;
	ref_norm=0;
	for(int i = 0; i < dim*dim; i++){
		float diff=hst_C_ref[i]-hst_C[i];
		error_norm+=diff*diff;
		ref_norm+=hst_C_ref[i]*hst_C_ref[i];
	}
	printf("Test %s \n", (sqrtf(error_norm/ref_norm)<1E-6) ? "PASSED" : "FAILED");
	printf("%f \n", sqrtf(error_norm/ref_norm));
//
//
//	cout << "A * B new 4:" << endl;
//	timeMul = Matrix_Math::mul_new4(hst_A, hst_B, hst_C, dim);
//	//printMatrix(hst_C, dim, dim);
//	printf("Time: %.4f ms \n", timeMul);
//
//	error_norm=0;
//	ref_norm=0;
//	for(int i = 0; i < dim*dim; i++){
//		float diff=hst_C_ref[i]-hst_C[i];
//		error_norm+=diff*diff;
//		ref_norm+=hst_C_ref[i]*hst_C_ref[i];
//	}
//	printf("Test %s \n", (sqrtf(error_norm/ref_norm)<1E-6) ? "PASSED" : "FAILED");
//	printf("%f \n", sqrtf(error_norm/ref_norm));


//	cout << "A * B cublas:" << endl;
//	//timeMul = Matrix_Math::mul_cublas(hst_A, hst_B, hst_C, dim);
//	//printMatrix(hst_C, dim, dim);
//	printf("Time: %.4f ms \n", timeMul);
//
//	error_norm=0;
//	ref_norm=0;
//	for(int i = 0; i < dim*dim; i++){
//		float diff=hst_C_ref[i]-hst_C[i];
//		error_norm+=diff*diff;
//		ref_norm+=hst_C_ref[i]*hst_C_ref[i];
//	}
//	printf("Test %s \n", (sqrtf(error_norm/ref_norm)<1E-6) ? "PASSED" : "FAILED");
//	printf("%f \n", sqrtf(error_norm/ref_norm));

}
