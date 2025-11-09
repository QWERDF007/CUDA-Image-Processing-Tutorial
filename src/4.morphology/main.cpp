#include "include/morphology.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <random>

void addSaltPepperNoise(cv::Mat &src, double amount)
{
    int                                num = amount * src.total();
    std::default_random_engine         gen;
    std::uniform_int_distribution<int> randX(0, src.cols - 1);
    std::uniform_int_distribution<int> randY(0, src.rows - 1);
    for (int i = 0; i < num; i++)
    {
        int x = randX(gen);
        int y = randY(gen);
        if (rand() % 2)
        {
            src.at<uchar>(y, x) = 255; // 盐
            // src.at<uchar>(y, x) = 0;   // 胡椒
        }
    }
}

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

    addSaltPepperNoise(src, 0.05);

    // 检查图像是否成功读取
    if (src.empty())
    {
        std::cerr << "error: failed to read image " << image_path << std::endl;
        return -1;
    }

    // 创建结构元素
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksz, ksz));

    const int width  = src.cols;
    const int height = src.rows;
    const int N      = width * height;
    const int NK     = ksz * ksz;

    // 分配设备内存
    uint8_t *d_src;
    uint8_t *d_kernel;
    uint8_t *d_erode, *d_dilate;
    cudaMalloc(&d_src, N * sizeof(uint8_t));
    cudaMalloc(&d_kernel, NK * sizeof(uint8_t));
    cudaMalloc(&d_erode, N * sizeof(uint8_t));
    cudaMalloc(&d_dilate, N * sizeof(uint8_t));

    // 将数据拷贝到设备
    cudaMemcpy(d_src, src.data, N * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data, NK * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 调用 CUDA 腐蚀、膨胀
    erode(d_src, d_erode, d_kernel, ksz, width, height);
    dilate(d_src, d_dilate, d_kernel, ksz, width, height);

    // cudaDeviceSynchronize();

    cv::Mat erode_result(height, width, CV_8UC1);
    cv::Mat dilate_result(height, width, CV_8UC1);

    // 拷贝结果
    cudaMemcpy(erode_result.data, d_erode, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(dilate_result.data, d_dilate, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cv::imwrite("src.png", src);
    cv::imwrite("erode.png", erode_result);
    cv::imwrite("dilate.png", dilate_result);

    return 0;
}
