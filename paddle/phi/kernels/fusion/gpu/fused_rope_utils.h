// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace phi {
namespace fusion {

template <typename T, typename MPType, int NInputs, int VecSize>
using VectorizedFusedRopeCudaKernelFunc =
    void (*)(phi::Array<const T*, NInputs> ins_data,
             phi::Array<const T*, 2> sin_cos_data,
             const int64_t* position_ids_data,
             bool flag_sin_cos,
             int sign,
             int64_t batch_size,
             int64_t seq_len,
             int64_t num_heads,
             int64_t head_dim,
             phi::Array<T*, NInputs> outs_data,
             int num_inputs,
             MPType div_c);


template <int NDim>
struct FusedRopeParam{
  __device__ __host__ __forceinline__ FusedRopeParam() {}
  __device__ __host__ __forceinline__ FusedRopeParam(const phi::Array<int64_t, NDim>& dims_, const phi::Array<int64_t, NDim>& strides_) {
    for (int i = 0; i < NDim; ++i) {
      dims[i] = dims_[i];
      strides[i] = strides_[i];
    }
  }
  phi::Array<int64_t, NDim> dims;
  phi::Array<int64_t, NDim> strides;
};

template <typename IndexT, int Rank>
struct DataPermuteParam{
  public:
    DataPermuteParam(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides, const std::vector<int>& perm) {
      static_assert(dims.size() == Rank, "dims.size() should equal to Rank");
      IndexT dst_strides[Rank];
      for (auto i = 0; i < Rank; ++i) {
        permuted_dims[i] = static_cast<IndexT>(dims[perm_[i]]);
        dst_strides[i] = static_cast<IndexT>(strides[perm[i]]);
      }
      permuted_contiguous_helper = IdxAndOffsetHelper<IndexT, Rank>(permuted_dims);  // contiguous
      permuted_non_contiguous_helper = IdxAndOffsetStrideHelper<IndexT, Rank>(dst_strides);  
    }

  public:
    IdxAndOffsetStrideHelper<IndexT, Rank> permuted_non_contiguous_helper;  // non-contiguous
    IdxAndOffsetHelper<IndexT, Rank> permuted_contiguous_helper;  // contiguous
    IndexT permuted_dims[Rank]{};
};

template <typename T, typename MPType, int VecSize = 2>
__device__ void VectorizedGetSinCos(phi::Array<const T*, 2> sin_cos_data,
                                    const int64_t* position_ids_data,
                                    bool flag_sin_cos,
                                    int64_t index,
                                    int64_t seq_len,
                                    int64_t num_heads,
                                    int64_t head_dim,
                                    MPType* out_sin,
                                    MPType* out_cos,
                                    MPType div_c) {
  MPType* sin_value = out_sin;
  MPType* cos_value = out_cos;

  if (flag_sin_cos) {
#pragma unroll
    for (int64_t nx = 0; nx < VecSize; ++nx) {
      int64_t index_wc = (index + nx) % (seq_len * num_heads * head_dim);
      int64_t pos_seq_ori = index_wc / (num_heads * head_dim);
      int64_t pos_seq;
      if (position_ids_data) {
        int64_t pos_bs = (index + nx) / (seq_len * num_heads * head_dim);
        int64_t index_ids = pos_bs * seq_len + pos_seq_ori;
        pos_seq = position_ids_data[index_ids];
      } else {
        pos_seq = pos_seq_ori;
      }
      int64_t pos_head = index_wc % head_dim;
      int64_t index_sc = pos_seq * head_dim + pos_head;
      const T* sin_input = sin_cos_data[0] + index_sc;
      const T* cos_input = sin_cos_data[1] + index_sc;

      sin_value[nx] = static_cast<MPType>(sin_input[0]);
      cos_value[nx] = static_cast<MPType>(cos_input[0]);
    }
  } else {
#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      // get sin_index and cos_index
      int64_t index_wc = (index + nx) % (seq_len * num_heads * head_dim);
      int64_t pos_seq = index_wc / (num_heads * head_dim);
      MPType idx = static_cast<MPType>((index_wc % head_dim) / 2 * 2.0);
      MPType indicses =
          static_cast<MPType>(1) /
          pow(static_cast<MPType>(10000), idx * static_cast<MPType>(div_c));
      MPType value = pos_seq * indicses;
      sin_value[nx] = sin(value);
      cos_value[nx] = cos(value);
    }
  }
}


template <typename T, typename MPType, int VecSize = 2>
__device__ void VectorizedGetSinCos_new(phi::Array<const T*, 2> sin_cos_data,
                                    const int64_t* position_ids_data,
                                    bool flag_sin_cos,
                                    int64_t index,
                                    FusedRopeParam<4> param,
                                    MPType* out_sin,
                                    MPType* out_cos,
                                    MPType div_c) {
  // param with dims [h, bs, seq_len, head_dim]
  // param with stides [bs*seq_len*head_dim, seq_len*head_dim, head_dim, 1]

  MPType* sin_value = out_sin;
  MPType* cos_value = out_cos;

  if (flag_sin_cos) {
#pragma unroll
    for (int64_t nx = 0; nx < VecSize; ++nx) {
      int64_t pos_seq_ori = ((index + nx) / param.strides[2]) % param.dims[2];

      int64_t pos_seq;
      if (position_ids_data) {
        int64_t pos_bs = ((index + nx) / param.strides[1]) % param.dims[1];
        int64_t index_ids = pos_bs * param.dims[2] + pos_seq_ori;
        pos_seq = position_ids_data[index_ids];
      } else {
        pos_seq = pos_seq_ori;
      }
      int64_t pos_head = ((index + nx) / param.strides[3]) % param.dims[3];
      int64_t index_sc = pos_seq * param.strides[2] + pos_head;
      const T* sin_input = sin_cos_data[0] + index_sc;
      const T* cos_input = sin_cos_data[1] + index_sc;

      sin_value[nx] = static_cast<MPType>(sin_input[0]);
      cos_value[nx] = static_cast<MPType>(cos_input[0]);
    }
  } else {

    // TODO (MarioLulab):
#if 0
#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      // get sin_index and cos_index
      int64_t index_wc = (index + nx) % (seq_len * num_heads * head_dim);
      int64_t pos_seq = index_wc / (num_heads * head_dim);
      MPType idx = static_cast<MPType>((index_wc % head_dim) / 2 * 2.0);
      MPType indicses =
          static_cast<MPType>(1) /
          pow(static_cast<MPType>(10000), idx * static_cast<MPType>(div_c));
      MPType value = pos_seq * indicses;
      sin_value[nx] = sin(value);
      cos_value[nx] = cos(value);
    }
#endif
  }
}

template <typename T, typename MPType, int NInputs, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateEveryTwoKernel(
    phi::Array<const T*, NInputs> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    phi::Array<T*, NInputs> outs_data,
    int num_inputs,
    MPType div_c) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
    VectorizedGetSinCos(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index,
                        seq_len,
                        num_heads,
                        head_dim,
                        sin_value,
                        cos_value,
                        div_c);

#pragma unroll
    for (int iter = 0; iter < NInputs; iter++) {
      if (iter >= num_inputs) break;
      const T* input = ins_data[iter] + index;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
      for (int nx = 0; nx < kVectorsPerThread; ++nx) {
        int pr_index = nx * 2;
        int ls_index = pr_index + 1;

        MPType p0 = static_cast<MPType>(input[pr_index]);
        MPType p1 = static_cast<MPType>(input[ls_index]);

        if (sign == 1) {
          result[pr_index] = cos_value[pr_index] * p0;
          result[pr_index] -= sin_value[pr_index] * p1;

          result[ls_index] = sin_value[ls_index] * p0;
          result[ls_index] += cos_value[ls_index] * p1;
        } else if (sign == -1) {
          result[pr_index] =
              cos_value[pr_index] * p0 + sin_value[ls_index] * p1;
          result[ls_index] =
              cos_value[ls_index] * p1 - sin_value[pr_index] * p0;
        }

        store[pr_index] = static_cast<T>(result[pr_index]);
        store[ls_index] = static_cast<T>(result[ls_index]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}

template <typename T, typename MPType, int NInputs, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateHalfKernel(
    phi::Array<const T*, NInputs> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    phi::Array<T*, NInputs> outs_data,
    int num_inputs,
    MPType div_c) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
    VectorizedGetSinCos(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index,
                        seq_len,
                        num_heads,
                        head_dim,
                        sin_value,
                        cos_value,
                        div_c);

    // use rotate_half mode
    int stride_r = head_dim / 2;
#pragma unroll
    for (int iter = 0; iter < NInputs; iter++) {
      if (iter >= num_inputs) break;
      // get value_index and rotate_half_index
      int index_v = index;
      int index_r = (index % head_dim) < stride_r ? (index + stride_r)
                                                  : (index - stride_r);
      MPType sign_r = (index % head_dim) < stride_r ? static_cast<MPType>(-1)
                                                    : static_cast<MPType>(1);
      const T* input_v = ins_data[iter] + index_v;
      const T* input_r = ins_data[iter] + index_r;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
      for (int nx = 0; nx < VecSize; ++nx) {
        MPType p0 = static_cast<MPType>(input_v[nx]);
        MPType p1 = static_cast<MPType>(input_r[nx]);

        result[nx] = cos_value[nx] * p0 + sign * sign_r * sin_value[nx] * p1;

        store[nx] = static_cast<T>(result[nx]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}



template <typename T, typename MPType, int NInputs, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateHalfKernel_new(
    phi::Array<const T*, NInputs> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    phi::Array<int64_t, NInputs> inputs_num_heads,
    FusedRopeParam<4> param,
    phi::Array<T*, NInputs> outs_data,
    int num_inputs,
    MPType div_c,
    int64_t multiply_heads) {

  // param is non-contiguous:
  // param with dims [h, bs, seq_len, head_dim]
  // param with stides [h*seq_len*head_dim, head_dim, h*head_dim, 1]
      
  int64_t index_contiguous =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = param.dims[0] * param.dims[1] * param.dims[2] * param.dims[3];
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;

  for (; index_contiguous < size; index_contiguous += stride) {

    // param_contiguous with dims [h, bs, seq_len, head_dim]
    // param_contiguous with stides [bs*seq_len*head_dim, seq_len*head_dim, head_dim, 1]
    FusedRopeParam<4> param_contiguous {param.dims, param.strides};
    param_contiguous.strides[0] = param_contiguous.dims[1] * param_contiguous.dims[2] * param_contiguous.dims[3];
    param_contiguous.strides[1] = param_contiguous.dims[2] * param_contiguous.dims[3];
    param_contiguous.strides[2] = param_contiguous.dims[3];
    param_contiguous.strides[3] = 1;

    VectorizedGetSinCos_new(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index_contiguous,  // contiguous
                        param_contiguous,
                        sin_value,
                        cos_value,
                        div_c);

      // contiguous index -> non-contiguous index
      int64_t index_h = (index_contiguous / param_contiguous.strides[0]) % param_contiguous.dims[0];
      int64_t index_bs = (index_contiguous / param_contiguous.strides[1]) % param_contiguous.dims[1];
      int64_t index_seq = (index_contiguous / param_contiguous.strides[2]) % param_contiguous.dims[2];
      int64_t index_dim = (index_contiguous / param_contiguous.strides[3]) % param_contiguous.dims[3];

    // use rotate_half mode
    int stride_r = param.dims[3] / 2;
#pragma unroll
    for (int iter = 0; iter < NInputs; iter++) {
      if (iter >= num_inputs) break;

      if (index_h >= inputs_num_heads[iter]) break;

      // get value_index and rotate_half_index
      int64_t index_non_contiguous = 0;
      if (iter == 0){
        index_non_contiguous = index_h * param.strides[0] + \
                                      index_bs * param.strides[1] + \
                                      index_seq * param.strides[2] + \
                                      index_dim * param.strides[3];  // non-contiguous

      }
      else{
        index_non_contiguous = index_h * param.strides[0] + \
                                      index_bs * param.strides[1] / multiply_heads + \
                                      index_seq * param.strides[2] / multiply_heads + \
                                      index_dim * param.strides[3];  // non-contiguous
      }

      int index_v = index_non_contiguous;
      int index_r = (index_non_contiguous % param.dims[3]) < stride_r ? (index_non_contiguous + stride_r)
                                                  : (index_non_contiguous - stride_r);
      MPType sign_r = (index_non_contiguous % param.dims[3]) < stride_r ? static_cast<MPType>(-1)
                                                    : static_cast<MPType>(1);

      const T* input_v = ins_data[iter] + index_v;
      const T* input_r = ins_data[iter] + index_r;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index_non_contiguous);

#pragma unroll
      for (int nx = 0; nx < VecSize; ++nx) {
        MPType p0 = static_cast<MPType>(input_v[nx]);
        MPType p1 = static_cast<MPType>(input_r[nx]);

        result[nx] = cos_value[nx] * p0 + sign * sign_r * sin_value[nx] * p1;

        store[nx] = static_cast<T>(result[nx]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}


}  // namespace fusion
}  // namespace phi
