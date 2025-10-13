#include "include/blend.cuh"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <iostream>

void init_image(cv::Mat &img1, cv::Mat &img2, const int width, const int height)
{
    int center_x    = width / 2;
    int center_y    = height / 2;
    int line_length = 200;

    // 横线：从 (center_x - line_length/2, center_y) 到 (center_x + line_length/2, center_y)
    cv::line(img1, cv::Point(center_x - line_length / 2, center_y), cv::Point(center_x + line_length / 2, center_y),
             cv::Scalar(255), 2);

    // 竖线：从 (center_x, center_y - line_length/2) 到 (center_x, center_y + line_length/2)
    cv::line(img2, cv::Point(center_x, center_y - line_length / 2), cv::Point(center_x, center_y + line_length / 2),
             cv::Scalar(255), 2);
}

int main(int argc, char *argv[])
{
    // 创建两个 512x512 的灰度图像（黑色背景）
    const int width  = 512;
    const int height = 512;
    cv::Mat   img1   = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat   img2   = cv::Mat::zeros(height, width, CV_8UC1);
    init_image(img1, img2, width, height);

    cv::imshow("Image 1 - Horizontal Line", img1);
    cv::imshow("Image 2 - Vertical Line", img2);

    const int N = width * height;

    // 分配设备内存
    uint8_t *d_src1, *d_src2, *d_dst;
    cudaMalloc(&d_src1, N * sizeof(uint8_t));
    cudaMalloc(&d_src2, N * sizeof(uint8_t));
    cudaMalloc(&d_dst, N * sizeof(uint8_t));

    // 将图像数据拷贝到设备
    cudaMemcpy(d_src1, img1.data, N * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src2, img2.data, N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 调用 CUDA 混合函数
    const double alpha = 0.5;
    const double beta  = 0.5;
    blend_image(d_src1, d_src2, d_dst, alpha, beta, N);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // 创建结果图像并拷贝数据回来
    cv::Mat result(height, width, CV_8UC1);
    cudaMemcpy(result.data, d_dst, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // 显示混合结果
    cv::imshow("Blended Result", result);

    std::cout << "Press any key to close windows..." << std::endl;

    // 等待按键
    cv::waitKey(0);

    // 释放设备内存
    cudaFree(d_src1);
    cudaFree(d_src2);
    cudaFree(d_dst);

    return 0;
}