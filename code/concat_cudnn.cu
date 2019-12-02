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

void Concat::map(void)
{
  size_t outputSize = sizeof(DATATYPE);
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  checkCUDA(cudaMalloc(&outputs[0].ptr, outputSize));
}

void Concat::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].ptr));
}

void Concat::forward(void)
{
  off_t offset = 0;
  for (int i = 0; i < numInputs; i++) {
    size_t size = sizeof(DATATYPE);
    for (int j = 0; j < inputs[i].numDim; j++)
      size *= inputs[i].dim[j];
    if (needCopy[i]&&!is_inception)
      checkCUDA(cudaMemcpyAsync(((char*)outputs[0].ptr) + offset,
                                inputs[i].ptr, size,
                                cudaMemcpyDeviceToDevice));
    offset += size;
  }
}

void Model::measure_concat_cost(Concat* concat)
{
  string key=export_op_key(*concat);
  for (int j = 0; j < concat->numInputs; j++) {
	  if (concat->needCopy[j]) key+=",<1>";
	  else key+=",<0>";
	  }
  //printf("<pre_measure>, %s\n",key.c_str());
  if(mp.find(key)!=mp.end())
  {
	  concat->runtime=mp[key].runtime;
	  concat->power=mp[key].power;
          concat->energy=mp[key].power*mp[key].runtime;
	if(!mute)
	{
	  printf("<found from mp>, %s, ",key.c_str());
	  printf("runtime=%f power=%f energe=%f\n", mp[key].runtime, mp[key].power, mp[key].power*mp[key].runtime);
	}
        return ;

  }

  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    for (int j = 0; j < concat->numInputs; j++) {
      if (concat->needCopy[j]) {
        size_t size = sizeof(DATATYPE);
        for (int k = 0; k < concat->inputs[j].numDim; k++)
          size *= concat->inputs[j].dim[k];
        checkCUDA(cudaMemcpyAsync(outputPtr, inputPtr, size,
                                  cudaMemcpyDeviceToDevice));
      }
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  //double runtime=concat->runtime = milliseconds / REPEAT_TIMES;


  long times=0;
  double current_time=get_current_time();
  double current_time2;
  start_check_power();
  for (int i = 0; ; i++,times++) {
    if(i%CHECK_TIME_PERIOD==0&&(current_time2=get_current_time())-current_time>measure_time) break;
    for (int j = 0; j < concat->numInputs; j++) {
      if (concat->needCopy[j]) {
        size_t size = sizeof(DATATYPE);
        for (int k = 0; k < concat->inputs[j].numDim; k++)
          size *= concat->inputs[j].dim[k];
        checkCUDA(cudaMemcpyAsync(outputPtr, inputPtr, size,
                                  cudaMemcpyDeviceToDevice));
      }
    }
  }
  double power=finish_check_power();
  double runtime=concat->runtime = (current_time2-current_time)/times;
   
  printf("<measure>, %s, ",key.c_str());
  printf("runtime=%f power=%f energy=%f\n",runtime,power,power*runtime);
  concat->power=power;
  concat->energy=power*runtime;
  mp[key].runtime=runtime;
  mp[key].power=power;
  db_output<<key<<"|"<<runtime<<"|"<<power<<endl;
  db_output.flush();


#ifdef VERBOSE
  printf("measure[Concat]: cost(%.4lf)\n", concat->runtime);
#endif
}

