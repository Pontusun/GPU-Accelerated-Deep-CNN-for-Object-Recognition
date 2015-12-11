#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels, const int spatial_dim, const Dtype* data, Dtype* out) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < num * spatial_dim; index += blockDim.x * gridDim.x){
		int number = index / spatial_dim;
		int spatial = index % spatial_dim;
		Dtype result = -FLT_MAX;
		for (int chanel = 0; chanel < channels; ++chanel) {
		  result = max(data[(number * channels + chanel) * spatial_dim + spatial], result);
		}
		out[index] = result;
	}
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count, const int num, const int channels, const int spatial_dim, const Dtype* channel_max, Dtype* data) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x){
		int number = index / channels / spatial_dim;
		int spatial = index % spatial_dim;
		data[index] = data[index] - channel_max[number * spatial_dim + spatial];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x){
		out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < num * spatial_dim; index += blockDim.x * gridDim.x){
		int number = index / spatial_dim;
		int spatial = index % spatial_dim;
		Dtype sum = 0;
		for (int chanel = 0; chanel < channels; ++chanel) {
		  sum += data[(number * channels + chanel) * spatial_dim + spatial];
		}
		channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count, const int num, const int channels, const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x){
		int number = index / channels / spatial_dim;
		int spatial = index % spatial_dim;
		data[index] = data[index] / channel_sum[number * spatial_dim + spatial];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels, const int spatial_dim, const Dtype* data_1, const Dtype* data_2,Dtype* channel_dot) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < num * spatial_dim; index += blockDim.x * gridDim.x){
		int number = index / spatial_dim;
		int spatial = index % spatial_dim;
		int idx;
		Dtype dot = 0;
		for (int chanel = 0; chanel < channels; ++chanel) {
			idx = (number * channels + chanel) * spatial_dim + spatial;
			dot += data_1[idx] * data_2[idx];
		}
		channel_dot[index] = dot;
	}
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int count = bottom[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_data, top_data);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int count = top[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, top_diff, bottom_diff);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_,
      top_diff, top_data, scale_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, bottom_diff);
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);


}  // namespace caffe
