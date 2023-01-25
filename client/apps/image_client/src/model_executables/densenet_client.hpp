/*
Inspiration source: https://github.com/triton-inference-server/client
*/
#include "../triton/triton.hpp"

void setModelInfo(ModelInfo &model_info)
{
  model_info.input_name_ = "data_0";
  model_info.output_name_ = "fc6_1";
  model_info.input_datatype_ = std::string("FP32");
  // The shape of the input
  model_info.max_batch_size_ = 0;
  model_info.input_c_ = 3;
  model_info.input_w_ = 224;
  model_info.input_h_ = 224;
  // The format of the input
  model_info.input_format_ = "FORMAT_NCHW";
  model_info.type1_ = CV_32FC1;
  model_info.type3_ = CV_32FC3;
  model_info.shape_.push_back(model_info.input_c_);
  model_info.shape_.push_back(model_info.input_h_);
  model_info.shape_.push_back(model_info.input_w_);
}

std::string request(cv::Mat request_img)
{
  int topk = 3;
  const std::string model_name = "densenet_onnx";
  std::string url("localhost:8001");

  // Create the inference client for the server.
  TritonClient triton_client;
  tc::Error err;
  err = tc::InferenceServerGrpcClient::Create(
      &triton_client.grpc_client_, url, false);
  if (!err.IsOk())
  {
    std::cerr << "error: unable to create client for inference: " << err
              << std::endl;
    exit(1);
  }

  ModelInfo model_info;
  setModelInfo(model_info);

  // Initialize the inputs with the data.
  tc::InferInput *input;
  err = tc::InferInput::Create(
      &input, model_info.input_name_, model_info.shape_, model_info.input_datatype_);
  if (!err.IsOk())
  {
    std::cerr << "unable to get input: " << err << std::endl;
    exit(1);
  }
  std::shared_ptr<tc::InferInput> input_ptr(input);

  // Preprocess the image into input data according to model
  // requirements
  std::vector<std::vector<uint8_t>> image_data;
  image_data.emplace_back();
  matImgToInputData(
      request_img, model_info.input_c_, model_info.input_h_, model_info.input_w_,
      model_info.input_format_, model_info.type1_, model_info.type3_,
      &(image_data.back()));

  tc::InferRequestedOutput *output;
  err =
      tc::InferRequestedOutput::Create(&output, model_info.output_name_, topk);
  if (!err.IsOk())
  {
    std::cerr << "unable to get output: " << err << std::endl;
    exit(1);
  }
  std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);

  err = input_ptr->AppendRaw(image_data[0]);
  std::vector<tc::InferInput *> inputs = {input_ptr.get()};
  std::vector<const tc::InferRequestedOutput *> outputs = {output_ptr.get()};

  tc::InferResult *result;
  err = triton_client.grpc_client_->Infer(
      &result, tc::InferOptions(model_name), inputs, outputs, tc::Headers());
  if (!err.IsOk())
  {
    std::cerr << "failed sending synchronous infer request: " << err
              << std::endl;
    exit(1);
  }

  return postprocess(result, model_info.output_name_, topk);
}
