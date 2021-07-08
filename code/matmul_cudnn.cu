/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops.h"
#include "cuda_helper.h"

void Matmul::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  int inputX = inputs[0].dim[0];
  int inputN = inputs[0].dim[1];
  int inputC = inputs[0].dim[2];
  int outputC = outputs[0].dim[2];
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, inputX, inputN, outputC, 1));
  if (actiMode != AC_MODE_NONE) {
    cudnnActivationMode_t mode;
    switch (actiMode) {
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));
  }
  // allocate tensors
  size_t filterSize = sizeof(DATATYPE) * inputC * outputC;
  size_t outputSize = sizeof(DATATYPE) * inputX * inputN * outputC;
  checkCUDA(cudaMalloc(&filterPtr, filterSize));
  checkCUDA(cudaMalloc(&outputs[0].ptr, outputSize));
}

void Matmul::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  if (actiMode != AC_MODE_NONE) {
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
  checkCUDA(cudaFree(filterPtr));
  checkCUDA(cudaFree(outputs[0].ptr));
}

void Matmul::forward(void)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int inputX = inputs[0].dim[0];
  int inputN = inputs[0].dim[1];
  int batch = inputX * inputN;
  int inputC = inputs[0].dim[2];
  int outputC = outputs[0].dim[2];
  checkCUDA(cublasSgemm(model->blas, CUBLAS_OP_T, CUBLAS_OP_N,
      outputC, batch, inputC, &alpha, (float*)filterPtr, inputC,
      (float*)inputs[0].ptr, inputC, &beta, (float*)outputs[0].ptr, outputC));
  if (actiMode != AC_MODE_NONE)
    checkCUDNN(cudnnActivationForward(model->dnn, actiDesc,
        &alpha, outputTensor, outputs[0].ptr,
        &beta, outputTensor, outputs[0].ptr));
}

void Model::measure_matmul_cost(Matmul* mm)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int inputX = mm->inputs[0].dim[0];
  int inputN = mm->inputs[0].dim[1];
  int batch = inputX * inputN;
  int inputC = mm->inputs[0].dim[2];
  int outputC = mm->outputs[0].dim[2];
  if (mm->actiMode != OpBase::AC_MODE_NONE) {
    cudnnActivationMode_t mode;
    switch (mm->actiMode) {
      case OpBase::AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case OpBase::AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case OpBase::AC_MODE_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));
  }
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, inputX, inputN, outputC, 1));

  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    checkCUDA(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
        outputC, batch, inputC, &alpha, filterPtr, inputC,
        inputPtr, inputC, &beta, outputPtr, outputC));
    if (mm->actiMode != OpBase::AC_MODE_NONE)
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, outputTensor, outputPtr,
          &beta, outputTensor, outputPtr));
  } 
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  mm->runtime = milliseconds / REPEAT_TIMES;
//#ifdef VERBOSE
  printf("measure[Matmul]: i(%d %d %d) o(%d) acti(%d) cost(%.4lf)\n",
         mm->inputs[0].dim[0], mm->inputs[0].dim[1], inputC, outputC,
         mm->actiMode, mm->runtime);
//#endif
}

