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
            // src.at<uchar>(y, x) = 255; // 盐
            src.at<uchar>(y, x) = 0; // 胡椒
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
    uint8_t *d_src, *d_tmp;
    uint8_t *d_kernel;
    uint8_t *d_result;
    uint8_t *d_erode, *d_dilate;
    uint8_t *d_open, *d_close;
    cudaMalloc(&d_src, N * sizeof(uint8_t));
    cudaMalloc(&d_kernel, NK * sizeof(uint8_t));
    cudaMalloc(&d_tmp, N * sizeof(uint8_t));
    cudaMalloc(&d_result, N * sizeof(uint8_t));
    cudaMalloc(&d_erode, N * sizeof(uint8_t));
    cudaMalloc(&d_dilate, N * sizeof(uint8_t));
    cudaMalloc(&d_open, N * sizeof(uint8_t));
    cudaMalloc(&d_close, N * sizeof(uint8_t));

    // 将数据拷贝到设备
    cudaMemcpy(d_src, src.data, N * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data, NK * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 调用 CUDA
    // open(d_src, d_result, d_kernel, d_tmp, ksz, width, height);
    // close(d_src, d_result, d_kernel, d_tmp, ksz, width, height);
    // morphology_gradient(d_src, d_result, d_kernel, d_erode, d_dilate, ksz, width, height);
    // tophat(d_src, d_result, d_kernel, d_tmp, d_open, ksz, width, height);
    blackhat(d_src, d_result, d_kernel, d_tmp, d_close, ksz, width, height);

    // cudaDeviceSynchronize();

    cv::Mat result(height, width, CV_8UC1);

    // 拷贝结果
    cudaMemcpy(result.data, d_result, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cv::imwrite("src.png", src);
    cv::imwrite("res.png", result);

    cudaFree(d_src);
    cudaFree(d_kernel);
    cudaFree(d_tmp);
    cudaFree(d_result);
    cudaFree(d_erode);
    cudaFree(d_dilate);
    cudaFree(d_open);
    cudaFree(d_close);

    return 0;
}
