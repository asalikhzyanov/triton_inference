/*
Source: https://github.com/triton-inference-server/client
*/

#pragma once
#include <string>
#include <fstream>
#include "common.hpp"
#define GET_TRANSFORMATION_CODE(x) cv::COLOR_##x

namespace
{
  struct ModelInfo
  {
    std::string output_name_;
    std::string input_name_;
    std::string input_datatype_;
    // The shape of the input
    int input_c_;
    int input_h_;
    int input_w_;
    // The format of the input
    std::string input_format_;
    int type1_;
    int type3_;
    int max_batch_size_;
    std::vector<int64_t> shape_;
  };

  void
  preprocess(
      const cv::Mat &img, const std::string &format, int img_type1, int img_type3,
      size_t img_channels, const cv::Size &img_size,
      std::vector<uint8_t> *input_data)
  {
    cv::Mat sample;
    if ((img.channels() == 3) && (img_channels == 1))
    {
      cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGR2GRAY));
    }
    else if ((img.channels() == 4) && (img_channels == 1))
    {
      cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGRA2GRAY));
    }
    else if ((img.channels() == 3) && (img_channels == 3))
    {
      cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGR2RGB));
    }
    else if ((img.channels() == 4) && (img_channels == 3))
    {
      cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGRA2RGB));
    }
    else if ((img.channels() == 1) && (img_channels == 3))
    {
      cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(GRAY2RGB));
    }
    else
    {
      std::cerr << "unexpected number of channels " << img.channels()
                << " in input image, model expects " << img_channels << "."
                << std::endl;
      exit(1);
    }

    cv::Mat sample_resized;
    if (sample.size() != img_size)
    {
      cv::resize(sample, sample_resized, img_size);
    }
    else
    {
      sample_resized = sample;
    }

    cv::Mat sample_type;
    sample_resized.convertTo(
        sample_type, (img_channels == 3) ? img_type3 : img_type1);

    cv::Mat sample_final;
    if (img_channels == 1)
    {
      sample_final = sample_type.mul(cv::Scalar(1 / 127.5));
      sample_final = sample_final - cv::Scalar(1.0);
    }
    else
    {
      sample_final =
          sample_type.mul(cv::Scalar(1 / 127.5, 1 / 127.5, 1 / 127.5));
      sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);
    }

    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample_final.total() * sample_final.elemSize();
    size_t pos = 0;
    input_data->resize(img_byte_size);

    // For NHWC format Mat is already in the correct order but need to
    // handle both cases of data being contigious or not.
    if (format.compare("FORMAT_NHWC") == 0)
    {
      if (sample_final.isContinuous())
      {
        memcpy(&((*input_data)[0]), sample_final.datastart, img_byte_size);
        pos = img_byte_size;
      }
      else
      {
        size_t row_byte_size = sample_final.cols * sample_final.elemSize();
        for (int r = 0; r < sample_final.rows; ++r)
        {
          memcpy(
              &((*input_data)[pos]), sample_final.ptr<uint8_t>(r), row_byte_size);
          pos += row_byte_size;
        }
      }
    }
    else
    {
      // (format.compare("FORMAT_NCHW") == 0)
      //
      // For CHW formats must split out each channel from the matrix and
      // order them as BBBB...GGGG...RRRR. To do this split the channels
      // of the image directly into 'input_data'. The BGR channels are
      // backed by the 'input_data' vector so that ends up with CHW
      // order of the data.
      std::vector<cv::Mat> input_bgr_channels;
      for (size_t i = 0; i < img_channels; ++i)
      {
        input_bgr_channels.emplace_back(
            img_size.height, img_size.width, img_type1, &((*input_data)[pos]));
        pos += input_bgr_channels.back().total() *
               input_bgr_channels.back().elemSize();
      }

      cv::split(sample_final, input_bgr_channels);
    }

    if (pos != img_byte_size)
    {
      std::cerr << "unexpected total size of channels " << pos << ", expecting "
                << img_byte_size << std::endl;
      exit(1);
    }
  }

  void
  fileToInputData(
      const std::string &filename, size_t c, size_t h, size_t w,
      const std::string &format, int type1, int type3,
      std::vector<uint8_t> *input_data)
  {
    // Load the specified image.
    cv::Mat img = cv::imread(filename);
    if (img.empty())
    {
      std::cerr << "error: unable to decode image " << filename << std::endl;
      exit(1);
    }

    // Pre-process the image to match input size expected by the model.
    preprocess(img, format, type1, type3, c, cv::Size(w, h), input_data);
  }

  std::string
  postprocess(
      const tc::InferResult *result,
      const std::string &output_name, const size_t topk)
  {
    if (!result->RequestStatus().IsOk())
    {
      std::cerr << "inference  failed with error: " << result->RequestStatus()
                << std::endl;
      exit(1);
    }

    // Get and validate the shape and datatype
    std::vector<int64_t> shape;
    tc::Error err = result->Shape(output_name, &shape);
    if (!err.IsOk())
    {
      std::cerr << "unable to get shape for " << output_name << std::endl;
      exit(1);
    }

    std::string datatype;
    err = result->Datatype(output_name, &datatype);
    if (!err.IsOk())
    {
      std::cerr << "unable to get datatype for " << output_name << std::endl;
      exit(1);
    }
    // Validate datatype
    if (datatype.compare("BYTES") != 0)
    {
      std::cerr << "received incorrect datatype for " << output_name << ": "
                << datatype << std::endl;
      exit(1);
    }

    std::vector<std::string> result_data;
    err = result->StringData(output_name, &result_data);
    if (!err.IsOk())
    {
      std::cerr << "unable to get data for " << output_name << std::endl;
      exit(1);
    }

    if (result_data.size() != (topk))
    {
      std::cerr << "unexpected number of strings in the result, expected "
                << (topk) << ", got " << result_data.size()
                << std::endl;
      exit(1);
    }
    size_t index = 0;
    std::string res;
    for (size_t c = 0; c < topk; ++c)
    {
      std::istringstream is(result_data[index]);
      int count = 0;
      std::string token;
      while (getline(is, token, ':'))
      {
        if (count == 0)
        {
          res.append("    ");
          res.append(token);
        }
        else if (count == 1)
        {
          res.append(" (");
          res.append(token);
          res.append(")");
        }
        else if (count == 2)
        {
          res.append(" = ");
          res.append(token);
        }
        count++;
      }
      res.append("\n");
      index++;
    }
    return res;
  }

  union TritonClient
  {
    TritonClient()
    {
      new (&grpc_client_) std::unique_ptr<tc::InferenceServerGrpcClient>{};
    }
    ~TritonClient() {}

    std::unique_ptr<tc::InferenceServerGrpcClient> grpc_client_;
  };

}
