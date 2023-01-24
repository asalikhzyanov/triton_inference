#include <iostream>
#include <string>
#include <vector>
#include <crow.h>
#include <opencv2/opencv.hpp>


CROW_ROUTE(app, "/image")
    .methods("POST"_method)
    ([](const crow::request& req) {
        auto file_data = req.body;
        std::vector<unsigned char> vec(file_data.begin(), file_data.end());

        cv::Mat image = cv::imdecode(vec, cv::IMREAD_COLOR);
        if(image.empty())
            return "Error: Failed to decode image";
        cv::imshow("Image", image);
        cv::waitKey(0);
        return "Image displayed";
    });

int main()
{
    app.port(18080).multithreaded().run();
}
