#include <assert.h>
#include "onnxruntime_c_api.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <../include/highfive/H5Easy.hpp>

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

void CheckStatus(OrtStatus* status) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

#define ORT_ABORT_ON_ERROR(expr)                                    \
    do {                                                            \
        OrtStatus* onnx_status = (expr);                            \
        if (onnx_status != NULL) {                                  \
            const char* msg = g_ort->GetErrorMessage(onnx_status);  \
            fprintf(stderr, "%s\n", msg);                           \
            g_ort->ReleaseStatus(onnx_status);                      \
            abort();                                                \
        }                                                           \
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
    
    // DEFINE SIZES OF MODEL INPUT AND MODEL OUTPUT TENSOR
    size_t input_tensor_size = 54 * 3;
    size_t output_tensor_size = 50 * 2;
    
    // LOAD EXAMPLE INPUTS FROM A SINGLE COLUMN
    H5Easy::File file("/home/ucaptp0/oasis-rt-surrogate/inference/data/sw_single_inputs.h5", H5Easy::File::ReadOnly); // SPECIFY INPUTS FILE HERE 
    std::vector<std::vector<double>> input_vector = H5Easy::load<std::vector<std::vector<double>>>(file, "/inputs");
    
    // CHANGE FORMAT OF INPUTS
    float* input_tensor_values = new float[54 * 3];
    size_t index = 0;
    for (const std::vector<double>& row : input_vector) {
        for (const double& element : row) {
            input_tensor_values[index++] = static_cast<float>(element);
        }
    }
    
    // VARS RELATED TO MODEL INPUTS
    const char *input_names[] = {"input_8"};
    const char *output_names[] = { "dense_output" };
    const int64_t input_shape[] = {1, 54, 3};
    const size_t model_input_len = input_tensor_size * sizeof(float);
    const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
    printf("The model_input length %zu\n", model_input_len);
    
    // PUTTING MODEL INPUTS INTO AN ORTVALUE OBJECT
    OrtMemoryInfo* memory_info = nullptr;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue *input_tensor = nullptr;
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        input_tensor_values,
        model_input_len,
        input_shape,
        input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    ));
    
    assert(input_tensor != NULL);
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    
    OrtValue* output_tensor = nullptr;
    

    // RUNNING MODEL INFERENCE
    printf("start to run onnxruntime\n");
    ORT_ABORT_ON_ERROR(g_ort->Run(
        session, // OrtSession *session
        nullptr, // const OrtRunOptions *run_options
        input_names, // const char *const *input_names
        (const OrtValue* const*)&input_tensor, // const OrtValue *const *inputs
        1, // size_t input_len,
        output_names, // const char *const *output_names
        1, // size_t output_names_len
        &output_tensor // OrtValue **outputs
    ));

    assert(output_tensor != NULL);
    printf("finish!!!");
    

    // PUTTING OUTPUT VALUES INTO A 2D VECTOR OF DOUBLES
    float* output_tensor_data = new float[50 * 2];
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
    size_t rows = 50;
    size_t cols = 2;
    std::vector<std::vector<double>> output_tensor_values(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output_tensor_values[i][j] = static_cast<double>(output_tensor_data[i * cols + j]);
        }
    }

    // SAVING MODEL OUTPUTS AS .H5
    H5Easy::File file_output("/home/ucaptp0/oasis-rt-surrogate/inference/data/sw_single_output.h5", H5Easy::File::Overwrite); // SPECIFY OUTPUT FILEPATH
    H5Easy::dump(file_output, "/outputs", output_tensor_values);

    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    
    return 0;
}



