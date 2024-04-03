// grayscale_filter_vec.cu
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>

__global__ void grayscaleFilterCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * 3; // RGB images have 3 channels

        // Convert to grayscale (simple average of R, G, B values)
        unsigned char r = inputImage[rgbOffset];
        unsigned char g = inputImage[rgbOffset + 1];
        unsigned char b = inputImage[rgbOffset + 2];
        outputImage[grayOffset] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

int main() {
    // Load image using OpenCV
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image\n";
        return -1;
    }

    // Get image properties
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    // Allocate memory for input and output images
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    unsigned char* d_inputImage, *d_outputImage;
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, width * height * sizeof(unsigned char));

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    grayscaleFilterCUDA<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    // Copy output image from device to host
    unsigned char* outputImage = new unsigned char[width * height];
    cudaMemcpy(outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Create grayscale image using OpenCV
    cv::Mat grayImage(height, width, CV_8UC1, outputImage);
    
    // Display and save the grayscale image
    cv::imshow("Grayscale Image", grayImage);
    cv::imwrite("output_image.jpg", grayImage);
    cv::waitKey(0);

    // Free device memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    delete[] outputImage;

    std::cout << "Tiempo de ejecución (CUDA): " << seconds << " segundos\n";

    return 0;
}
