// serial_grayscale_filter_sca.cu
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

void grayscaleFilterSerial(cv::Mat& inputImage, cv::Mat& outputImage) {
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            cv::Vec3b intensity = inputImage.at<cv::Vec3b>(y, x);
            float grayValue = 0.299f * intensity[2] + 0.587f * intensity[1] + 0.114f * intensity[0];
            outputImage.at<uchar>(y, x) = static_cast<uchar>(grayValue);
        }
    }
}

int main() {
    // Load image using OpenCV
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image\n";
        return -1;
    }

    // Create an empty grayscale image
    cv::Mat grayImage(image.rows, image.cols, CV_8UC1);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Apply grayscale filter
    grayscaleFilterSerial(image, grayImage);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    // Display and save the grayscale image
    cv::imshow("Grayscale Image (Serial)", grayImage);
    cv::imwrite("output_image_serial.jpg", grayImage);
    cv::waitKey(0);

    std::cout << "Tiempo de ejecución (Serial): " << seconds << " segundos\n";

    return 0;
}
