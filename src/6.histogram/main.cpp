#include "include/histogram.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
    // 检查命令行参数
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image path> <bins> " << std::endl;
        return -1;
    }

    // 从参数读取图像路径和滤波核大小
    std::string image_path = argv[1];

    // 直方图 bins 数量
    const int bins = std::atoi(argv[2]);

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

    // 分配设备内存
    uint8_t *d_src;
    int32_t *d_hist;
    cudaMalloc(&d_src, N * sizeof(uint8_t));
    cudaMalloc(&d_hist, bins * sizeof(int32_t));

    // 将数据拷贝到设备
    cudaMemcpy(d_src, src.data, N * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, bins * sizeof(int32_t));

    // 调用 CUDA
    calcHist(d_src, d_hist, bins, height, width);

    // cudaDeviceSynchronize();

    cv::Mat result(bins, 1, CV_32SC1);

    // 拷贝结果
    cudaMemcpy(result.data, d_hist, bins * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_hist);

    cv::Mat      hist;
    float        range[]   = {0, 256}; // 像素值范围
    const float *histRange = {range};

    cv::calcHist(&src,       // 输入图像（可以是多张）
                 1,          // 输入图数量
                 0,          // 使用第 0 个通道（灰度图只有一个）
                 cv::Mat(),  // 不使用 mask
                 hist,       // 输出直方图（float 32）
                 1,          // 直方图维度
                 &bins,      // 每个维度的 bins
                 &histRange, // 每个维度的取值范围
                 true,       // 直方图是否归一化
                 false);     // 累计参数（一般 false）

    hist.convertTo(hist, CV_32SC1);

    std::cout << result.size() << " " << hist.size() << std::endl;
    // 比较两个 calcHist 函数处理的差异
    cv::Mat diff;
    cv::absdiff(result, hist, diff);
    int diff_cnt = cv::countNonZero(diff);
    std::cout << "diffrenece: " << diff_cnt << std::endl;

    return 0;
}
