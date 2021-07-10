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

void Element::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  // set descriptors
  int inputN = inputs[0].dim[0];
  int inputC = max(inputs[0].dim[1], 1);
  int inputH = max(inputs[0].dim[2], 1);
  int inputW = max(inputs[0].dim[3], 1);
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, inputN, inputC, inputH, inputW));

  cudnnOpTensorOp_t opType;
  switch (type) {
    case OP_EW_ADD:
      opType = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      opType = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, opType, CUDNN_DATA_FLOAT,
      CUDNN_NOT_PROPAGATE_NAN));
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE);
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  checkCUDA(cudaMalloc(&outputs[0].ptr, outputSize));
}

void Element::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
  checkCUDA(cudaFree(outputs[0].ptr));
}

void Element::forward(void)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN(cudnnOpTensor(model->dnn, opDesc, &alpha, inputTensor, inputs[0].ptr,
      &alpha, inputTensor, inputs[1].ptr, &beta, inputTensor, outputs[0].ptr));
}

void Model::measure_element_cost(Element* ele)
{
  string key=export_op_key(*ele);
  //printf("<pre_measure>, %s\n",key.c_str());

  if(mp.find(key)!=mp.end())
  {
	  ele->runtime=mp[key].runtime;
	  ele->power=mp[key].power;
          ele->energy=mp[key].power*mp[key].runtime;
	if(!mute)
	{
	  printf("<found from mp>, %s, ",key.c_str());
	  printf("runtime=%f power=%f energe=%f\n", mp[key].runtime, mp[key].power, mp[key].power*mp[key].runtime);
	}
        return ;

  }

  
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int inputN = ele->inputs[0].dim[0];
  int inputC = max(ele->inputs[0].dim[1], 1);
  int inputH = max(ele->inputs[0].dim[2], 1);
  int inputW = max(ele->inputs[0].dim[3], 1);
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, inputN, inputC, inputH, inputW));

  cudnnOpTensorOp_t opType;
  switch (ele->type) {
    case OpBase::OP_EW_ADD:
      opType = CUDNN_OP_TENSOR_ADD;
      break;
    case OpBase::OP_EW_MUL:
      opType = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, opType, CUDNN_DATA_FLOAT,
      CUDNN_NOT_PROPAGATE_NAN));

  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    checkCUDNN(cudnnOpTensor(dnn, opDesc, &alpha, inputTensor, inputPtr,
        &alpha, inputTensor, filterPtr, &beta, inputTensor, outputPtr));
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
//  double runtime=ele->runtime = milliseconds / REPEAT_TIMES;
  {
	  long times=0;
	  double current_time=get_current_time();
	  for (int i = 0; ; i++,times++) {
		  if(i%CHECK_TIME_PERIOD==0&&get_current_time()-current_time>stress_time) break;
		  checkCUDNN(cudnnOpTensor(dnn, opDesc, &alpha, inputTensor, inputPtr,
					  &alpha, inputTensor, filterPtr, &beta, inputTensor, outputPtr));
	  }
	  checkCUDA(cudaDeviceSynchronize());
  }

  sleep(idle_time);

  long times=0; 
  double current_time=get_current_time();
  start_check_power();
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));

  for (int i = 0; ; i++,times++) {
    if(i%CHECK_TIME_PERIOD==0&&get_current_time()-current_time>measure_time) break;
    checkCUDNN(cudnnOpTensor(dnn, opDesc, &alpha, inputTensor, inputPtr,
        &alpha, inputTensor, filterPtr, &beta, inputTensor, outputPtr));
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float gpu_time;
  cudaEventElapsedTime(&gpu_time, startEvent, endEvent);

  double power=finish_check_power();
  double runtime=ele->runtime = gpu_time/times;

  printf("<measure>, %s, ",key.c_str());
  printf("runtime=%f power=%f energy=%f\n",runtime,power,power*runtime);
  ele->power=power;
  ele->energy=power*runtime;
  mp[key].runtime=runtime;
  mp[key].power=power;
  db_output<<key<<"|"<<runtime<<"|"<<power<<endl;
  db_output.flush();
#ifdef VERBOSE
  printf("measure[Element]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
         ele->inputs[0].dim[0], ele->inputs[0].dim[1], ele->inputs[0].dim[2],
         ele->inputs[0].dim[3], ele->type, ele->runtime);
#endif
}

