#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){

		//compute number, chanel, pooled result coordinate according to the index
	    const int col = index % pooled_width;
	    const int row = (index / pooled_width) % pooled_height;
	    const int chanel = (index / pooled_width / pooled_height) % channels;
	    const int number = index / pooled_width / pooled_height / channels;

	    //compute the start and end coordinate of the input patch
	    int row_start = row * stride_h - pad_h;
	    int col_start = col * stride_w - pad_w;
	    int row_end = min(row_start + kernel_h, height + pad_h);
	    int col_end = min(col_start + kernel_w, width + pad_w);
	    //pool size must be computed here before the the coord are limited
	    const int pool_size = (row_end - row_start) * (col_end - col_start);

	    //limit the  coordinate
	    row_start = max(row_start, 0);
	    col_start = max(col_start, 0);
	    row_end = min(row_end, height);
	    col_end = min(col_end, width);

	    //compute average, loop through the pooling patch
	    Dtype result = 0;
	    const Dtype* const bottom_data_to_compute = bottom_data + (number * channels + chanel) * height * width;
	    for (int i = row_start; i < row_end; ++i) {
	      for (int j = col_start; j < col_end; ++j) {
	    	  result += bottom_data_to_compute[(i*width) + j];
	      }
	    }
	    top_data[index] = result / pool_size;
  }
}

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
		//compute number, chanel, pooled result coordinate according to the index
	    const int col = index % pooled_width;
	    const int row = (index / pooled_width) % pooled_height;
	    const int chanel = (index / pooled_width / pooled_height) % channels;
	    const int number = index / pooled_width / pooled_height / channels;

	    //compute the start and end coordinate of the input patch
	    int row_start = row * stride_h - pad_h;
	    int col_start = col * stride_w - pad_w;
	    int row_end = min(row_start + kernel_h, height + pad_h);
	    int col_end = min(col_start + kernel_w, width + pad_w);

	    //limit the  coordinate
	    row_start = max(row_start, 0);
	    col_start = max(col_start, 0);
	    row_end = min(row_end, height);
	    col_end = min(col_end, width);

	    //find the max
	     Dtype result = -999999.9;
	     int	result_idx = -1;
	     const Dtype* const bottom_data_to_compute = bottom_data + (number * channels + chanel) * height * width;
	     for (int i = row_start; i < row_end; ++i) {
	       for (int j = col_start; j < col_end; ++j) {
	         if (bottom_data_to_compute[(i*width) + j] > result) {
	           result = bottom_data_to_compute[(i*width) + j];
	           result_idx = (i*width) + j;
	         }
	       }
	     }
	     top_data[index] = result;
	     if (mask) {
	       mask[index] = result_idx;
	     } else {
	       top_mask[index] = result_idx;
	     }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){

		//get number, chanel, coordinate according to the index
	    const int col = index % width;
	    const int row = (index / width) % height;
	    const int chanel = (index / width / height) % channels;
	    const int number = index / width / height / channels;

	    //get corresponding coordinate in the input (here the input is pooled result)
	    const int row_start = (row + pad_h < kernel_h) ? 0 : (row + pad_h - kernel_h) / stride_h + 1;
	    const int row_end = min((row + pad_h) / stride_h + 1, pooled_height);
	    const int col_start = (col + pad_w < kernel_w) ? 0 : (col + pad_w - kernel_w) / stride_w + 1;
	    const int col_end = min((col + pad_w) / stride_w + 1, pooled_width);

	    //compute gradient
	      Dtype gradient = 0;
	      const Dtype* const top_diff_to_compute = top_diff + (number * channels + chanel) * pooled_height * pooled_width;
	      if (mask) {
	        const int* const mask_slice = mask + (number * channels + chanel) * pooled_height * pooled_width;
	        for (int i = row_start; i < row_end; ++i) {
	          for (int j = col_start; j < col_end; ++j) {
	            if (mask_slice[(i*pooled_width) + j] == (row* width) + col) {
	              gradient += top_diff_to_compute[(i*pooled_width) + j];
	            }
	          }
	        }
	      } else {
	        const Dtype* const top_mask_slice = top_mask + (number * channels + chanel) * pooled_height * pooled_width;
	        for (int i = row_start; i < row_end; ++i) {
	                for (int j = col_start; j < col_end; ++j) {
	            if (top_mask_slice[(i*pooled_width) + j] == (row* width) + col) {
	              gradient += top_diff_to_compute[(i*pooled_width) + j];
	            }
	          }
	        }
	      }
	      bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
	for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){

		//get number, chanel, coordinate according to the index
	    const int col = index % width;
	    const int row = (index / width) % height;
	    const int chanel = (index / width / height) % channels;
	    const int number = index / width / height / channels;

	    //get corresponding coordinate in the input (here the input is pooled result)
	    const int row_start = (row + pad_h < kernel_h) ? 0 : (row + pad_h - kernel_h) / stride_h + 1;
	    const int row_end = min((row + pad_h) / stride_h + 1, pooled_height);
	    const int col_start = (col + pad_w < kernel_w) ? 0 : (col + pad_w - kernel_w) / stride_w + 1;
	    const int col_end = min((col + pad_w) / stride_w + 1, pooled_width);

	    //compute gradient, loop through the pooled result
	    Dtype gradient = 0;
	    const Dtype* const top_diff_to_compute = top_diff + (number * channels + chanel) * pooled_height * pooled_width;
	    for (int i = row_start; i < row_end; ++i) {
	        for (int j = col_start; j < col_end; ++j) {
	        // compute the pooling size
	        int row_start_output = i * stride_h - pad_h;
	        int col_start_output = j * stride_w - pad_w;
	        int row_end_output = min(row_start_output + kernel_h, height + pad_h);
	        int col_end_output = min(col_start_output + kernel_w, width + pad_w);
	        int pool_size = (row_end_output - row_start_output) * (col_end_output - col_start_output);

	        //compute the gradient
	        gradient += top_diff_to_compute[(i*pooled_width) + j] / pool_size;
	      }
	    }
	    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
