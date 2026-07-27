// test_rolling_median.cpp — unit tests for RollingMedianOp (滚动中位数)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingMedian.h"

using namespace quantcore;

class RollingMedianOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    // Helper: median of values in [start, end]
    static double median(const double* data, std::size_t start, std::size_t end) {
        std::size_t count = end - start + 1;
        std::vector<double> buf(count);
        for (std::size_t j = 0; j < count; ++j) buf[j] = data[start + j];
        std::sort(buf.begin(), buf.end());
        std::size_t mid = count / 2;
        if (count % 2 == 1) return buf[mid];
        return (buf[mid - 1] + buf[mid]) / 2.0;
    }

    std::vector<double> input_;
};

TEST_F(RollingMedianOpTest, OpCode) {
    RollingMedianOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_MEDIAN);
}

TEST_F(RollingMedianOpTest, Name) {
    RollingMedianOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_median");
}

TEST_F(RollingMedianOpTest, WindowAccess) {
    RollingMedianOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar — odd window
// ============================================================

TEST_F(RollingMedianOpTest, ScalarWindow3) {
    RollingMedianOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), median(input_.data(), 0, 2));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), median(input_.data(), 1, 3));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), median(input_.data(), 7, 9));
}

TEST_F(RollingMedianOpTest, ScalarWindow5) {
    RollingMedianOp op(5);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 3, 5)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 4, 5), median(input_.data(), 0, 4));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 5, 5), median(input_.data(), 1, 5));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 5), median(input_.data(), 5, 9));
}

// ============================================================
// evaluateScalar — even window
// ============================================================

TEST_F(RollingMedianOpTest, ScalarWindow4) {
    RollingMedianOp op(4);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 2, 4)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 4), median(input_.data(), 0, 3));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 4, 4), median(input_.data(), 1, 4));
}

TEST_F(RollingMedianOpTest, ScalarWindow2) {
    RollingMedianOp op(2);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 2)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 1, 2), median(input_.data(), 0, 1));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 2), median(input_.data(), 1, 2));
}

TEST_F(RollingMedianOpTest, ScalarUnsorted) {
    double data[] = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
    RollingMedianOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), median(data, 0, 2));  // median of {3,1,4}
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), median(data, 1, 3));  // median of {1,4,1}
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 4, 3), median(data, 2, 4));  // median of {4,1,5}
}

TEST_F(RollingMedianOpTest, ScalarWindow1) {
    RollingMedianOp op(1);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingMedianOpTest, BatchNoNulls) {
    RollingMedianOp op(3);
    std::vector<double> output(input_.size());
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    for (std::size_t i = 0; i < input_.size(); ++i) {
        double expected = op.evaluateScalar(input_.data(), i, 3);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(output[i], expected) << "i=" << i;
    }
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingMedianOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 100;
    RollingMedianOp op(7);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i % 13) * 2.3;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 7);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 7);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}
