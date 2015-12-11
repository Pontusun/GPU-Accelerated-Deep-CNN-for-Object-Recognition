#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <fstream>
#include <iostream>

extern float hst_A[], hst_B[];
extern float *hst_C;

namespace Matrix_Math {
	void init(int);
	float add(float*, float*, float*);
	float sub(float*, float*, float*);
	float mul(float*, float*, float*, int);
	float mul_new(float*, float*, float*, int);
	float mul_new1(float*, float*, float*, int);
	float mul_new2(float*, float*, float*, int);
	float mul_new3(float*, float*, float*, int);
	float mul_new4(float*, float*, float*, int);
	float mul_cublas(float*, float*, float*, int);
}
