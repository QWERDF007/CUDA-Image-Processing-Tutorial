#include "include/blur_image_separable.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
    // 检查命令行参数
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image path> " << " <kernel size> " << std::endl;
        return -1;
    }

    // 从参数读取图像路径和滤波核大小
    std::string image_path = argv[1];
    const int   ksz        = std::atoi(argv[2]);

    // 读取图像（灰度模式）
    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    // 检查图像是否成功读取
    if (src.empty())
    {
        std::cerr << "error: failed to read image " << image_path << std::endl;
        return -1;
    }

    const int width  = src.cols;
    const int height = src.rows;
    const int N      = width * height;

    // 显示原始图像
    cv::imshow("Original Image", src);

    // 分配设备内存
    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(uint8_t));
    cudaMalloc(&d_dst, N * sizeof(uint8_t));

    // 将图像数据拷贝到设备
    cudaMemcpy(d_src, src.data, N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 调用 CUDA 均值模糊函数
    blur_image(d_src, d_dst, ksz, ksz, width, height);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // 创建结果图像并拷贝数据回来
    cv::Mat result(height, width, CV_8UC1);
    cudaMemcpy(result.data, d_dst, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // 显示模糊结果
    cv::imshow("Blurred Result", result);

    std::cout << "Press any key to close windows..." << std::endl;

    // 等待按键
    cv::waitKey(0);

    // 释放设备内存
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}