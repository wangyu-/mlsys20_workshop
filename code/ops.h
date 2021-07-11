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

#ifndef _CNN_OPS_H_
#define _CNN_OPS_H_

#ifdef USE_CUDNN
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef TRT
#include "NvInfer.h"
#include "NvUtils.h"

using namespace nvinfer1;
#endif

#ifdef USE_MKL
#include "mkl_dnn.h"
#endif

#include <cassert>
#include <map>
#include <array>
#include <vector>
#include <set>
#include <list>
#include <iostream>
#include <fstream>
#include <string.h>
using namespace std;
#include <unistd.h>

#define MAX_DIM 4
#define MAX_NUM_INPUTS 6
#define MAX_NUM_OUTPUTS 6
#define BATCH_SIZE 1
#define MOD 8
#define OLD_SIZE (128 * 1024 * 1024)// 128MB
#define MAX_TENSOR_SIZE (128 * 1024 * 1024 *MOD)// 128MB
#define REPEAT_TIMES 500
#define CONCAT_KEY_LENGTH (MAX_NUM_INPUTS + 5)
#define SPLIT_KEY_LENGTH (MAX_NUM_OUTPUTS + 4)
const size_t WORK_SPACE_SIZE = (size_t)2 * 1024 * 1024 * 1024; // 2GB
typedef float DATATYPE;

class Model;
class OpBase;
struct Op {
  Op(void);
  //bool operator==(const Op& b) const;
  //bool operator<(const Op& b) const;
  size_t guid;
  OpBase* ptr;
};

struct Edge {
  Edge(int _idx, Op _op);
  int idx;
  Op op;
};

struct EdgeCompare {
  bool operator()(const Edge& a, const Edge& b) const {
    if (a.op.guid != b.op.guid) return a.op.guid < b.op.guid;
    if (a.op.ptr != b.op.ptr) return a.op.ptr < b.op.ptr;
    if (a.idx != b.idx) return a.idx < b.idx;
    return false;
  };
};

struct OpCompare {
  bool operator()(const Op& a, const Op& b) const {
    if (a.guid != b.guid) return a.guid < b.guid;
    return a.ptr < b.ptr;
  };
};

struct Tensor {
  //bool operator==(const Tensor& b);
  int numDim, dim[MAX_DIM];
  int idx; // idx is used for Ops with multiple outputs (e.g., split)
  Op op;
  void* ptr;
};

class OpBase {
public:
  enum OpType {
    OP_NOOP,
    OP_ANY,
    OP_CONV2D,
    OP_LINEAR,
    OP_POOL2D_MAX,
    OP_POOL2D_AVG,
    OP_RELU,
    OP_SIGMOID,
    OP_BATCHNORM,
    OP_CONCAT,
    OP_SPLIT,
    // RNN operators
    OP_EW_ADD,
    OP_EW_MUL,
    OP_MATMUL,
  };
  enum OpParameter {
    PM_OP_TYPE,			// AnyOp
    PM_NUM_INPUTS, 		// AnyOp
    PM_NUM_OUTPUTS,		// AnyOp
    PM_KERNEL_H,		// Conv2D, Pool2D
    PM_KERNEL_W,		// Conv2D, Pool2D
    PM_STRIDE_H,		// Conv2D, Pool2D
    PM_STRIDE_W,		// Conv2D, Pool2D
    PM_PAD_H,			// Conv2D, Pool2D
    PM_PAD_W,			// Conv2D, Pool2D
    PM_RELU,			// Conv2D, Pool2D
    PM_OUTPUT_C,
    PM_ACTI,
  };
  enum ActiMode {
    AC_MODE_NONE,
    AC_MODE_SIGMOID,
    AC_MODE_RELU,
    AC_MODE_TANH,
  };

  OpBase(Tensor input, Model* _model, OpType _type);
  OpBase(Tensor input0, Tensor input1, Model* _model, OpType _type);
  OpBase(int n, Tensor* inputs, Model* _model, OpType _type);
  virtual bool get_parameter(OpParameter, int*) = 0;
  virtual void forward(void) = 0;
  virtual void map(void) = 0;
  virtual void unmap(void) = 0;
  virtual void collect_costs(float& exe_time, float& flops,
                             float& mem_acc, int& num_kernels) = 0;
public:
  Tensor inputs[MAX_NUM_INPUTS], outputs[MAX_NUM_OUTPUTS];
  int numInputs, numOutputs;
  OpType type;
  Model *model;
  double runtime=0;
  double power=0;
  double energy=0;
};

typedef std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator op_it_t;
typedef cudnnConvolutionFwdAlgo_t algo_t;
/*
struct conv_choice_t
{	
	op_it_t op_it;
	algo_t algo;

};*/
struct cost_t 
{
	algo_t algo;
	double runtime;
	double energy;
};
struct algo_data_t
{
	algo_t algo;
	std::map<algo_t,cost_t> algo_cost_mp;
};
typedef std::map<Op, algo_data_t, OpCompare> conv_algo_mp_t;
class Graph {
public:
  double total_runtime=0;
  double total_energy=0;
  conv_algo_mp_t conv_algo_mp;
  Graph(Model *_model);
  void export_to_file(std::string file_name);
  Tensor conv2d(Tensor _input, int _outputC, int _kernelH, int _kernelW,
                int _strideH, int _strideW, int _padH, int _padW,
                bool _relu = false);
  Tensor matmul(Tensor _input, int _outputC,
                OpBase::ActiMode _actiMode = OpBase::AC_MODE_NONE);
  Tensor pool2d_max(Tensor _input, int _kernelH, int _kernelW,
                    int _strideH, int _strideW, int _padH, int _padW,
                    bool _relu = false);
  Tensor pool2d_avg(Tensor _input, int _kernelH, int _kernelW,
                    int _strideH, int _strideW, int _padH, int _padW,
                    bool _relu = false);
  Tensor relu(Tensor _input, bool _inPlace = true);
  Tensor sigmoid(Tensor _input, bool _inPlace = true);
  Tensor batchnorm(Tensor _input);
  Tensor concat(int n, Tensor* _inputs);
  void split(Tensor _input, int c1, int c2, Tensor* outputs);
  void split(Tensor _input, int num, int* channels, Tensor* outputs);
  Tensor noop(Tensor _input);
  Tensor add(Tensor _t1, Tensor _t2);
  Tensor mul(Tensor _t1, Tensor _t2);
  size_t num_in_edges(Op op);
  size_t num_out_edges(Op op);
  bool has_edge(Op src, Op dst, int idx);
  size_t hash(void);
  void print(void);
  bool check_correctness(void);
  float total_cost(void);
  float run(Model *model);
  void print_costs(void);
#ifdef TRT
  void buildTRTNetwork(INetworkDefinition *network);
private:
  void buildTRTNetworkHelper(INetworkDefinition *network, std::map<Edge, ITensor *, EdgeCompare>& outputs, Edge edge);
#endif
  void export_op(ofstream &file_stream, Op &op);
public:
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges;
  Model *model;
  float totalCost;
};


class Conv2D : public OpBase {
public:
  Conv2D(Model* _model, Tensor _input, int _outputC,
         int _kernelH, int _kernelW, int _strideH, int _strideW,
         int _padH, int _padW, bool _relu);
  ~Conv2D(void);
  void forward(void);
  void map(void);
  void unmap(void);
  bool get_parameter(OpParameter para, int*);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
#ifdef USE_CUDNN
  cudnnConvolutionFwdAlgo_t selectForwardAlgorithm(void);
#endif
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  std::map<algo_t,cost_t> algo_cost_mp;
#endif
#ifdef USE_MKL
  std::vector<dnnPrimitive_t> compList;
  std::vector<std::array<void*, dnnResourceNumber>> rsrcList;
  int fwdAlgo; // Placeholder, should never use this in mkl.
#endif
  int outputC, kernelH, kernelW, strideH, strideW, padH, padW;
  bool relu;
  void *biasPtr, *filterPtr;
};

class Matmul : public OpBase {
public:
  Matmul(Model* _model, Tensor _input, int _outputC,
         ActiMode _actiMode);
  ~Matmul(void);
  void forward(void);
  void map(void);
  void unmap(void);
  bool get_parameter(OpParameter para, int*);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int outputC;
  ActiMode actiMode;
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
  void *filterPtr;
};

class Pool2D : public OpBase {
public:
  Pool2D(Model* _model, Tensor _input, OpType _type,
         int _kernelH, int _kernelW,
         int _strideH, int _strideW,
         int _padH, int _padW, bool _relu);
  ~Pool2D(void);
  bool get_parameter(OpParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
#endif
#ifdef USE_MKL
  std::vector<dnnPrimitive_t> compList;
  std::vector<std::array<void*, dnnResourceNumber>> rsrcList;
#endif
  int kernelH, kernelW, strideH, strideW, padH, padW;
  bool relu;
};

class Activation : public OpBase {
public:
  Activation(Model* _model, Tensor _input, OpType _type, bool _inPlace);
  ~Activation(void);
  bool get_parameter(OpParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
  bool inPlace;
};

class BatchNorm : public OpBase {
public:
  BatchNorm(Model* _model, Tensor _input);
  ~BatchNorm(void);
  bool get_parameter(OpParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
#endif
#ifdef USE_MKL
  dnnPrimitive_t comp;
  std::array<void*, dnnResourceNumber> rsrc;
#endif
  DATATYPE *biasPtr, *scalePtr, *runningMean, *runningVar, *saveMean, *saveVar;
};

class Concat : public OpBase {
public:
  Concat(Model* _model, int n, Tensor* _inputs, bool* _needCopy);
  ~Concat(void);
  bool get_parameter(OpParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  bool needCopy[MAX_NUM_INPUTS];
};

class Split : public OpBase {
public:
  Split(Model* _model, Tensor _input, int n, int* channels);
  ~Split(void);
  bool get_parameter(OpParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int channels[MAX_NUM_OUTPUTS];
};

class NoOp : public OpBase {
public:
  NoOp(Model* _model, Tensor _input);
  ~NoOp(void);
  bool get_parameter(OpParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Element : public OpBase {
public:
  Element(Model* _model, OpType _type, Tensor _t1, Tensor _t2);
  ~Element(void);
  bool get_parameter(OpParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor;
  cudnnOpTensorDescriptor_t opDesc;
#endif
};

// keys are (inputN, inputC, inputH, inputW, outputC, kernelH, kernelW,              
//           strideH, strideW, padH, padW, relu)
struct Conv2DKey {
  Conv2DKey(Tensor, int, int, int, int, int, int, int, bool);
  int keys[12];
};

struct Conv2DCompare {
  bool operator()(const Conv2DKey& a, const Conv2DKey& b) const {
    for (int i = 0; i < 12; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// keys are (inputX, inputN, inputC, outputC, acti)
//
struct MatmulKey {
  MatmulKey(Tensor, int, OpBase::ActiMode);
  int keys[5];
};

struct MatmulCompare {
  bool operator()(const MatmulKey& a, const MatmulKey& b) const {
    for (int i = 0; i < 5; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// keys are (inputN, inputC, inputH, inputW, kernelH, kernelW,              
//           strideH, strideW, padH, padW, relu, type)
struct Pool2DKey {
  Pool2DKey(Tensor, OpBase::OpType, int, int, int, int, int, int, bool);
  int keys[12];
};

struct Pool2DCompare {
  bool operator()(const Pool2DKey& a, const Pool2DKey& b) const {
    for (int i = 0; i < 12; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// keys are (inputN, inputC, inputH, inputW, type, inPlace)
struct ActivationKey {
  ActivationKey(Tensor, OpBase::OpType, bool);
  int keys[6];
};

struct ActivationCompare {
  bool operator()(const ActivationKey& a, const ActivationKey& b) const {
    for (int i = 0; i < 6; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// key is (inputN, inputC, inputH, inputW)
struct BatchNormKey {
  BatchNormKey(Tensor);
  int keys[4];
};

struct BatchNormCompare {
  bool operator()(const BatchNormKey& a, const BatchNormKey& b) const {
    for (int i = 0; i < 4; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// keys are (n, inputN, inputH, inputW, bitmask(needCopy), inputC[0...n-1])
struct ConcatKey {
  ConcatKey(int, Tensor*, bool*);
  int keys[CONCAT_KEY_LENGTH];
};

struct ConcatCompare {
  bool operator()(const ConcatKey& a, const ConcatKey& b) const {
    for (int i = 0; i < CONCAT_KEY_LENGTH; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// key is (inputN, input H, inputW, n, outputC[0...,n-1])
struct SplitKey {
  SplitKey(Tensor input, int n, int* channels);
  int keys[SPLIT_KEY_LENGTH];
};

struct SplitCompare {
  bool operator()(const SplitKey& a, const SplitKey& b) const {
    for (int i = 0; i < SPLIT_KEY_LENGTH; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// key is (inputN, inputC, inputH, inputW)
struct NoopKey {
  NoopKey(Tensor input);
  int keys[4];
};

struct NoopCompare {
  bool operator()(const NoopKey& a, const NoopKey& b) const {
    for (int i = 0; i < 4; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// key is (inputN, inputC, inputH, inputW, type)
struct ElementKey {
  ElementKey(Tensor t, OpBase::OpType type);
  int keys[5];
};

struct ElementCompare {
  bool operator()(const ElementKey& a, const ElementKey& b) const {
    for (int i = 0; i < 5; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

class Model {
public:
  Model(bool);
  Op get_or_create_conv2d(Tensor _input, int _outputC,
                          int _kernelH, int _kernelW,
                          int _strideH, int _strideW,
                          int _padH, int _padW, bool _relu);
  Op get_or_create_matmul(Tensor _input, int _outputC,
                          OpBase::ActiMode _actimode);
  Op get_or_create_pool2d(Tensor _input, OpBase::OpType _type,
                          int _kernelH, int _kernelW,
                          int _strideH, int _strideW,
                          int _padH, int _padW, bool _relu);
  Op get_or_create_activation(Tensor _input, OpBase::OpType _type,
                              bool _inPlace);
  Op get_or_create_batchnorm(Tensor _input);
  Op get_or_create_concat(int n, Tensor* _inputs, bool* _needCopy);
  Op get_or_create_split(Tensor _input, int n, int* channels);
  Op get_or_create_noop(Tensor _input);
  Op get_or_create_element(OpBase::OpType type, Tensor t1, Tensor t2);
  void measure_conv2d_cost(Conv2D*);
  void measure_matmul_cost(Matmul*);
  void measure_pool2d_cost(Pool2D*);
  void measure_activation_cost(Activation*);
  void measure_batchnorm_cost(BatchNorm*);
  void measure_concat_cost(Concat*);
  void measure_split_cost(Split*);
  void measure_element_cost(Element*);
  void* allocate_memory(size_t size);
  float measure_oplist_runtime(const std::vector<OpBase*>& list);
public:
  bool isTraining;
  size_t global_unique_id;
  size_t workSpaceSize;
  void* workSpace;
#ifdef USE_CUDNN
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  // Note that actiDesc is set when we construct Model since
  // all relus are identical.
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudaEvent_t startEvent, endEvent;
  // variables for batch norm
  cudnnTensorDescriptor_t scaleTensor;
  // variables for element wise
  cudnnOpTensorDescriptor_t opDesc;
#endif
  std::map<Conv2DKey, Conv2D*, Conv2DCompare> conv2d;
  std::map<MatmulKey, Matmul*, MatmulCompare> matmul;
  std::map<Pool2DKey, Pool2D*, Pool2DCompare> pool2d;
  std::map<ActivationKey, Activation*, ActivationCompare> activation;
  std::map<BatchNormKey, BatchNorm*, BatchNormCompare> batchnorm;
  std::map<ConcatKey, Concat*, ConcatCompare> concat;
  std::map<SplitKey, Split*, SplitCompare> split;
  std::map<NoopKey, NoOp*, NoopCompare> noop;
  std::map<ElementKey, Element*, ElementCompare> element;
  DATATYPE *inputPtr, *biasPtr, *outputPtr, *filterPtr;
  // variables for batch norm
  DATATYPE *scalePtr, *runningMean, *runningVar, *saveMean, *saveVar;
};

string export_op_key(OpBase &ob);

void start_check_power();

double finish_check_power();

inline long long get_current_time()
{
	timespec tmp_time;
	clock_gettime(CLOCK_MONOTONIC, &tmp_time);
	return tmp_time.tv_sec*1000ll+tmp_time.tv_nsec/(1000*1000l);
}
#define TIME_BEFORE_MEASURE 1

#define CHECK_TIME_PERIOD 500

const int idle_time=0; //sec
const int stress_time=2*1000;
const int measure_time=3*1000;
extern int about_to_exit;

struct value_t
{
	double runtime=0.0;
	double power=0.0;
};

extern map<string,value_t> mp;
extern ofstream db_output;

inline vector<string> string_to_vec(const char * s,const char * sp) {
	  vector<string> res;
	  string str=s;
	  char *p = strtok ((char *)str.c_str(),sp);
	  while (p != NULL)
	  {
		res.push_back(p);
	    	p = strtok(NULL, sp);
	  }

	  return res;
}
extern int use_perf_order;
extern vector<double> params;
double cost_func(double runtime,double power);
inline double power_no_idle(double power)
{
	//power-=23;
	//if(power<0) power=0;
	return power;	
}
const int params_num=10;
const int mute=1;
extern int is_inception;
extern int use_enlarge;

#endif
