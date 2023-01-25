#include <iostream>
#include <string>
#include <vector>
#include <crow.h>
#include <opencv2/opencv.hpp>
#include "model_executables/densenet_client.hpp"

int main()
{
    crow::SimpleApp app;

    CROW_ROUTE(app, "/image")
        .methods("POST"_method)([](const crow::request &req)
                                {
        auto info = crow::json::load(req.body);
        std::string base64_string = info["classification"].s();
        std::string decoded_image = crow::utility::base64decode(base64_string, base64_string.size());
        std::vector<uchar> vec(decoded_image.begin(), decoded_image.end());
        cv::Mat image = cv::imdecode(vec, cv::IMREAD_UNCHANGED);
        if(image.empty())
        {
            return crow::response(404, "Error: Failed to decode image");
        }
        return crow::response(200, request(image).c_str()); });

    app.port(18080).multithreaded().run();
}
