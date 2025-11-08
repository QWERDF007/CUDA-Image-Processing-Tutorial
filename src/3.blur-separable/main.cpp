#include "include/blur_image_separable.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>

int main(int argc, char *argv[])
{
    // 检查命令行参数
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image size> " << " <kernel size> " << std::endl;
        return -1;
    }

    // 从参数读取图像路径和滤波核大小
    const int img_size = std::atoi(argv[1]);
    const int ksz      = std::atoi(argv[2]);

    // 创建一个 [0, 255] 随机值的图像
    cv::Mat src(img_size, img_size, CV_8UC1);
    cv::randu(src, 0, 256);

    const int width  = src.cols;
    const int height = src.rows;
    const int N      = width * height;

    // 定义预热和迭代次数
    const int warmup = 10;
    const int iters  = 1000;

    // 分配设备内存
    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(uint8_t));
    cudaMalloc(&d_dst, N * sizeof(uint8_t));

    // 将图像数据拷贝到设备
    cudaMemcpy(d_src, src.data, N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 预热
    for (int i = 0; i < warmup; ++i)
    {
        blur_image(d_src, d_dst, ksz, ksz, width, height);
    }
    cudaDeviceSynchronize();

    // 记录 blur_image 耗时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        blur_image(d_src, d_dst, ksz, ksz, width, height);
    }
    cudaDeviceSynchronize();
    auto   end = std::chrono::high_resolution_clock::now();
    double t0  = std::chrono::duration<double, std::milli>(end - start).count();

    // 创建结果图像并拷贝数据回来
    cv::Mat result0(height, width, CV_8UC1);
    cudaMemcpy(result0.data, d_dst, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    uint32_t *d_tmp;
    cudaMalloc(&d_tmp, N * sizeof(uint32_t));
    // 预热
    for (int i = 0; i < warmup; ++i)
    {
        blur_image_separable(d_src, d_dst, d_tmp, ksz, ksz, width, height);
    }
    cudaDeviceSynchronize();

    // 记录 blur_image_separable 耗时
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        blur_image_separable(d_src, d_dst, d_tmp, ksz, ksz, width, height);
    }
    cudaDeviceSynchronize();
    end       = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double, std::milli>(end - start).count();

    // 拷贝 blur_image_separable 的结果
    cv::Mat result1(height, width, CV_8UC1);
    cudaMemcpy(result1.data, d_dst, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // 比较两个 blur 核函数处理的差异
    cv::Mat diff;
    cv::absdiff(result0, result1, diff);
    int diff_cnt = cv::countNonZero(diff);

    std::cout << "          blur_image time: " << t0 << " ms, avg: " << t0 / iters << std::endl;
    std::cout << "blur_image_separable time: " << t1 << " ms, avg: " << t1 / iters << std::endl;
    std::cout << "               diffrenece: " << diff_cnt << std::endl;

    // 释放设备内存
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}