// test_rolling_ema.cpp — unit tests for RollingEmaOp (指数移动平均)
// Matches pandas.Series.ewm(span=n, adjust=False).mean()
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingEma.h"

using namespace quantcore;

class RollingRollingEmaOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }
    std::vector<double> input_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(RollingRollingEmaOpTest, OpCode) {
    RollingEmaOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_EMA);
}

TEST_F(RollingRollingEmaOpTest, Name) {
    RollingEmaOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_ema");
    static_assert(RollingEmaOp::kOpCode == RollingOpCode::ROLLING_EMA);
}

TEST_F(RollingRollingEmaOpTest, WindowAccess) {
    RollingEmaOp op(5);
    EXPECT_EQ(op.window(), 5u);
}

// ============================================================
// evaluateScalar — basic (verified against pandas ewm(span, adjust=False).mean())
// ============================================================

TEST_F(RollingRollingEmaOpTest, ScalarSpan3) {
    // span=3, alpha = 2/(3+1) = 0.5
    // i=0,1: i+1 < 3 → NaN (not enough history)
    // i=2: EMA[2] = 0.5*3 + 0.5*(0.5*2+0.5*1) = 2.25
    // i=3: EMA[3] = 0.5*4 + 0.5*2.25 = 3.125, etc.
    RollingEmaOp op(3);

    double alpha = 0.5, omAlpha = 0.5;

    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));

    double ema = 1.0;
    ema = alpha * 2.0 + omAlpha * ema;  // EMA[1] = 1.5
    ema = alpha * 3.0 + omAlpha * ema;  // EMA[2] = 2.25
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), ema);

    for (std::size_t i = 3; i < 10; ++i) {
        ema = alpha * input_[i] + omAlpha * ema;
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 3), ema) << "i=" << i;
    }
}

TEST_F(RollingRollingEmaOpTest, ScalarSpan5) {
    // span=5, alpha = 2/(5+1) = 1/3
    // i=0..3: i+1 < 5 → NaN
    RollingEmaOp op(5);
    double alpha = 2.0 / 6.0, omAlpha = 1.0 - alpha;

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), i, 5))) << "i=" << i;
    }

    double ema = 1.0;
    for (std::size_t k = 1; k <= 4; ++k) {
        ema = alpha * input_[k] + omAlpha * ema;
    }
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 4, 5), ema);

    for (std::size_t i = 5; i < 10; ++i) {
        ema = alpha * input_[i] + omAlpha * ema;
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 5), ema) << "i=" << i;
    }
}

TEST_F(RollingRollingEmaOpTest, ScalarSpan1) {
    // span=1, alpha = 2/(1+1) = 1.0
    // With alpha=1, EMA[i] = input[i] (no smoothing)
    RollingEmaOp op(1);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

TEST_F(RollingRollingEmaOpTest, ScalarConstantSeries) {
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingEmaOp op(3);
    // i=0,1: i+1 < 3 → NaN
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 1, 3)));
    for (std::size_t i = 2; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 3), 5.0);
    }
}

TEST_F(RollingRollingEmaOpTest, ScalarMatchesPandas) {
    // span=4, alpha = 2/5 = 0.4
    // i=0..2: i+1 < 4 → NaN
    // i=3: EMA = 0.4*40 + 0.6*(0.4*30+0.6*(0.4*20+0.6*10)) = 28.24
    // i=4: EMA = 0.4*50 + 0.6*28.24 = 36.944
    RollingEmaOp op(4);
    double data[] = {10.0, 20.0, 30.0, 40.0, 50.0};

    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 0, 4)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 1, 4)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 4)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 4), 28.24);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 4, 4), 36.944);
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(RollingRollingEmaOpTest, ScalarNaNPropagation) {
    // When a NaN appears in the sequence, it contaminates all subsequent EMA values.
    // Also verify boundary NaN (i+1 < window) is independent of data NaN.
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0, 5.0};
    RollingEmaOp op(3);

    // i=0,1: i+1 < 3 → boundary NaN
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 1, 3)));

    // i=2: first valid position; 0.5*NaN + 0.5*1.5 = NaN (data NaN)
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 3)));

    // i=3,4: NaN drags forward through the recurrence
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 3, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 4, 3)));
}

TEST_F(RollingRollingEmaOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    double data[] = {1.0, inf, 3.0};
    RollingEmaOp op(3);

    // i=0: i+1 < 3 → boundary NaN
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 0, 3)));
    // i=1: i+1 < 3 → boundary NaN (even though data[1]=Inf, boundary wins)
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 1, 3)));
    // i=2: first valid position. EMA[2] = 0.5*3.0 + 0.5*EMA[1],
    //      where EMA[1] = 0.5*Inf + 0.5*1.0 = Inf, so EMA[2] = Inf
    EXPECT_TRUE(std::isinf(op.evaluateScalar(data, 2, 3)));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingRollingEmaOpTest, BatchNoNulls) {
    RollingEmaOp op(3);
    std::size_t n = input_.size();
    std::vector<double> output(n);
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    // Verify against scalar reference for all positions
    for (std::size_t i = 0; i < n; ++i) {
        double expected = op.evaluateScalar(input_.data(), i, 3);
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
        } else {
            EXPECT_DOUBLE_EQ(output[i], expected) << "i=" << i;
        }
    }
}

TEST_F(RollingRollingEmaOpTest, BatchWithNulls) {
    double input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double output[7] = {};
    uint64_t mask[1] = {(1ULL << 3) | (1ULL << 5)};  // pos 3,5 null (engine skips evaluateScalar)

    RollingEmaOp op(3);
    op.evaluate(ColView<double>(input, 7, mask), output);

    // i=0,1: i+1 < 3 → boundary NaN from evaluateScalar
    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    // i=2: first valid EMA
    EXPECT_DOUBLE_EQ(output[2], op.evaluateScalar(input, 2, 3));
    // i=3: null → 0.0
    EXPECT_DOUBLE_EQ(output[3], 0.0);
    // i=4: valid EMA
    EXPECT_DOUBLE_EQ(output[4], op.evaluateScalar(input, 4, 3));
    // i=5: null → 0.0
    EXPECT_DOUBLE_EQ(output[5], 0.0);
    // i=6: valid EMA
    EXPECT_DOUBLE_EQ(output[6], op.evaluateScalar(input, 6, 3));
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingRollingEmaOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 200;
    RollingEmaOp op(10);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i) * 1.2;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 10);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 10);
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        } else {
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
        }
    }
}

// ============================================================
// Statelessness
// ============================================================

TEST_F(RollingRollingEmaOpTest, MultipleInstancesIdentical) {
    RollingEmaOp op1(4), op2(4);
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (std::size_t i = 0; i < 6; ++i) {
        double v1 = op1.evaluateScalar(data, i, 4);
        double v2 = op2.evaluateScalar(data, i, 4);
        if (std::isnan(v1)) {
            EXPECT_TRUE(std::isnan(v2)) << "i=" << i;
        } else {
            EXPECT_DOUBLE_EQ(v1, v2) << "i=" << i;
        }
    }
}
