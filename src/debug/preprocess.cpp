// Owner: Junkyu Kim
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "basic/clahe.h"
#include "basic/color.h"
#include "basic/filter.h"
#include "debug/debug.h"
#include "io/io.h"

namespace la::debug {

    static void expectNear(float actual, float expected) {
        if (!std::isfinite(actual) ||
            std::fabs(actual - expected) > 0.00001f) {
            std::printf(
                "[FAIL] expected %.6f, got %.6f\n",
                expected, actual);
            throw std::runtime_error("filter test failed");
        }
    }

    static void runFilterTests() {
        // Test 1: identity kernel preserves every input pixel.
        cv::Mat src(2, 3, CV_32FC1);
        src.ptr<float>(0)[0] = -1.f;
        src.ptr<float>(0)[1] = 2.f;
        src.ptr<float>(0)[2] = 4.f;
        src.ptr<float>(1)[0] = 8.f;
        src.ptr<float>(1)[1] = 0.f;
        src.ptr<float>(1)[2] = 3.f;

        cv::Mat identity(1, 1, CV_32FC1);
        identity.ptr<float>(0)[0] = 1.f;

        const cv::Mat same = basic::convolve2D(src, identity);

        if (same.rows != src.rows || same.cols != src.cols ||
            same.type() != CV_32FC1) {
            throw std::runtime_error("filter output format mismatch");
        }

        for (int y = 0; y < src.rows; ++y) {
            for (int x = 0; x < src.cols; ++x) {
                expectNear(same.ptr<float>(y)[x],
                    src.ptr<float>(y)[x]);
            }
        }
        std::printf("[PASS] identity convolution\n");

        // Test 2: asymmetric kernel checks true convolution and borders.
        // Input: [1, 2, 4], kernel: [1, 0, -1]
        // With replicate borders, output must be [1, 3, 2].
        cv::Mat row(1, 3, CV_32FC1);
        row.ptr<float>(0)[0] = 1.f;
        row.ptr<float>(0)[1] = 2.f;
        row.ptr<float>(0)[2] = 4.f;

        cv::Mat kernel(1, 3, CV_32FC1);
        kernel.ptr<float>(0)[0] = 1.f;
        kernel.ptr<float>(0)[1] = 0.f;
        kernel.ptr<float>(0)[2] = -1.f;

        const cv::Mat horizontal = basic::convolve2D(row, kernel);
        expectNear(horizontal.ptr<float>(0)[0], 1.f);
        expectNear(horizontal.ptr<float>(0)[1], 3.f);
        expectNear(horizontal.ptr<float>(0)[2], 2.f);
        std::printf("[PASS] horizontal convolution and borders\n");

        // Test 3: verify the same calculation in the vertical direction.
        cv::Mat column(3, 1, CV_32FC1);
        cv::Mat verticalKernel(3, 1, CV_32FC1);

        for (int i = 0; i < 3; ++i) {
            column.ptr<float>(i)[0] = row.ptr<float>(0)[i];
            verticalKernel.ptr<float>(i)[0] = kernel.ptr<float>(0)[i];
        }

        const cv::Mat vertical =
            basic::convolve2D(column, verticalKernel);

        expectNear(vertical.ptr<float>(0)[0], 1.f);
        expectNear(vertical.ptr<float>(1)[0], 3.f);
        expectNear(vertical.ptr<float>(2)[0], 2.f);
        std::printf("[PASS] vertical convolution and borders\n");

        // Test 4: Gaussian blur preserves a constant image,
        // even when its kernel is larger than the image.
        cv::Mat constant(3, 4, CV_32FC1);
        for (int y = 0; y < constant.rows; ++y) {
            for (int x = 0; x < constant.cols; ++x) {
                constant.ptr<float>(y)[x] = 0.4f;
            }
        }

        const cv::Mat blurred = basic::gaussianBlur(constant, 2.f);

        if (blurred.rows != constant.rows ||
            blurred.cols != constant.cols ||
            blurred.type() != CV_32FC1) {
            throw std::runtime_error("Gaussian output format mismatch");
        }

        for (int y = 0; y < blurred.rows; ++y) {
            for (int x = 0; x < blurred.cols; ++x) {
                expectNear(blurred.ptr<float>(y)[x], 0.4f);
            }
        }
        std::printf("[PASS] Gaussian constant image and borders\n");
        std::printf("All filter tests passed.\n");
    }

    int runPreprocess(const Args& args) {
        runFilterTests();

        if (args.empty()) {
            std::printf("usage: --debug preprocess <photo.jpg>\n");
            return 0;
        }

        const cv::Mat bgr = io::loadColor(args[0]);
        const cv::Mat gray = basic::bgrToGray(bgr);
        io::save("debug_preprocess_gray.png",
            basic::grayToBgr8(gray));

        const cv::Mat blurred = basic::gaussianBlur(gray, 2.f);
        io::save("debug_preprocess_blur.png",
            basic::grayToBgr8(blurred));

        const cv::Mat eq = basic::clahe(gray);
        io::save("debug_preprocess_clahe.png",
            basic::grayToBgr8(eq));

        std::printf("saved debug_preprocess_*.png\n");
        return 0;
    }

}  // namespace la::debug