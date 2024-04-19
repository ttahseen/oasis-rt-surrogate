#include <assert.h>
#include "onnxruntime_c_api.h"
#include "data_file.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

void CheckStatus(OrtStatus* status)
{
  if (status != NULL) {
    const char* msg = g_ort->GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(status);
    exit(1);
  }
}

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

int main() {

  OrtEnv* env = nullptr;
  OrtSession* session = nullptr;

  CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "test", &env));

  OrtSessionOptions* session_options = nullptr;
  CheckStatus(g_ort->CreateSessionOptions(&session_options));

  // set numthreads
  ORT_ABORT_ON_ERROR(g_ort->SetIntraOpNumThreads(session_options, 1));
  ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC));

  const char* model_path = "/home/ucaptp0/oasis-rt-surrogate/trained_models/onnx/rnn_sw/dynamical/649968_combined_inputs.onnx";
  printf("Using Onnxruntime C API\n");

  CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session));


  size_t input_tensor_size = 54 * 3;
  float *input_tensor_values = new float[54 * 3]{0.5};

  const char *input_names[] = {"input_8"};
  const char *output_names[] = { "dense_output" };
  const int64_t input_shape[] = {1, 54, 3};
  const size_t model_input_len = input_tensor_size * sizeof(float);
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  printf("The model_input length %zu\n", model_input_len);

  OrtMemoryInfo* memory_info = nullptr;
  CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  OrtValue *input_tensor = nullptr;
  CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info,
                                                    input_tensor_values,
                                                    model_input_len,
                                                    input_shape,
                                                    input_shape_len,
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                    &input_tensor));

  assert(input_tensor != NULL);
  int is_tensor;
  CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);

  OrtValue* output_tensor = nullptr;

  printf("start to run onnxruntime\n");
  ORT_ABORT_ON_ERROR(g_ort->Run(session,
                         nullptr,
                         input_names,
                         (const OrtValue* const*)&input_tensor,
                         1,
                         output_names,
                         1,
                         &output_tensor));

  assert(output_tensor != NULL);
  printf("finish!!!");

  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);

  return 0;
}