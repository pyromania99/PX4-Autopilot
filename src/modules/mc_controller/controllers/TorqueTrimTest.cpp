/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file TorqueTrimTest.cpp
 *
 * The torque trim is one scalar integral per axis, so these are properties:
 *  - OFF by default is exact passthrough, and yaw is never touched;
 *  - it INTEGRATES the law's torque at the configured rate, in the law's direction;
 *  - CONVERGENCE against a constant body-fixed bias, for any rotation the loop puts
 *    between torque and attitude error - the reason it integrates torque at all. The same
 *    test with the attitude error integrated instead diverges past 90 deg, and is asserted
 *    to, so the design argument cannot silently stop being true;
 *  - ANTI-WINDUP at the output clamp and on allocator saturation, and the limit.
 */

#include <gtest/gtest.h>

#include "TorqueTrim.hpp"

#include <mathlib/mathlib.h>
#include <parameters/param.h>

#include <cmath>

using namespace matrix;

namespace
{

/// updateParams() is protected; a test needs to re-cache after changing a parameter.
class TestTrim : public TorqueTrim
{
public:
	TestTrim() : TorqueTrim(nullptr) {}
	using TorqueTrim::updateParams;
};

void setF(const char *n, float v) { param_set_no_notification(param_find(n), &v); }

class TorqueTrimTest : public ::testing::Test
{
public:
	void SetUp() override
	{
		param_control_autosave(false);
		setF("MC_ATT_TRIM_I", 0.f);
		setF("MC_ATT_TRIM_LIM", 0.1f);
	}

	void TearDown() override
	{
		setF("MC_ATT_TRIM_I", 0.f);
		setF("MC_ATT_TRIM_LIM", 0.1f);
	}
};

constexpr float kDt = 0.0025f;

TEST_F(TorqueTrimTest, DisabledByDefaultIsExactPassthrough)
{
	TestTrim trim;
	const Vector3f law(0.05f, -0.03f, 0.2f);

	for (int i = 0; i < 1000; i++) {
		EXPECT_EQ(trim.apply(law, kDt), law);
	}

	EXPECT_FALSE(trim.enabled());
	EXPECT_EQ(trim.trim().norm(), 0.f);
}

TEST_F(TorqueTrimTest, IntegratesTheLawTorqueAndNeverTouchesYaw)
{
	setF("MC_ATT_TRIM_I", 1.f);
	TestTrim trim;
	const Vector3f law(0.04f, -0.02f, 0.3f);

	const int steps = 200;
	Vector3f out{};

	for (int i = 0; i < steps; i++) {
		out = trim.apply(law, kDt);
	}

	EXPECT_NEAR(trim.trim()(0), 1.f * 0.04f * kDt * steps, 1e-6f);
	EXPECT_NEAR(trim.trim()(1), 1.f * -0.02f * kDt * steps, 1e-6f);
	EXPECT_NEAR(out(0), law(0) + trim.trim()(0), 1e-7f);
	EXPECT_NEAR(out(1), law(1) + trim.trim()(1), 1e-7f);
	EXPECT_EQ(out(2), law(2));
}

/**
 * The design argument. Quasi-static loop: whatever torque the vehicle needs beyond the
 * trim, the law holds, so tau_law = -(bias + trim), and that does not depend on how the
 * loop maps torque to attitude error - so the trim converges however spin rotates that
 * map. The attitude error the loop settles at is the held torque ROTATED by phi; an
 * integral of that converges at 45 deg and diverges past 90 deg, which is asserted too so
 * the reason for this design cannot silently stop being true.
 */
TEST_F(TorqueTrimTest, ConvergesOnABodyFixedBiasWhereAnAttitudeIntegralWouldNot)
{
	setF("MC_ATT_TRIM_I", 2.f);
	setF("MC_ATT_TRIM_LIM", 0.5f);
	const Vector2f bias(-0.015f, 0.008f);
	const int steps = 4000;	// 10 s at 400 Hz

	TestTrim trim;

	for (int i = 0; i < steps; i++) {
		const Vector2f held = -(bias + trim.trim());
		trim.apply(Vector3f(held(0), held(1), 0.f), kDt);
	}

	EXPECT_NEAR(trim.trim()(0), -bias(0), 1e-5f);
	EXPECT_NEAR(trim.trim()(1), -bias(1), 1e-5f);

	auto attitude_integral_residual = [&](float phi_deg) {
		const float phi = math::radians(phi_deg);
		const float c = cosf(phi);
		const float s = sinf(phi);
		Vector2f u{};

		for (int i = 0; i < steps; i++) {
			const Vector2f held = -(bias + u);
			const Vector2f error(c * held(0) - s * held(1), s * held(0) + c * held(1));
			u += 2.f * error * kDt;
		}

		return (u + bias).norm();
	};

	EXPECT_LT(attitude_integral_residual(45.f), 1e-5f);
	EXPECT_GT(attitude_integral_residual(120.f), bias.norm());
}

TEST_F(TorqueTrimTest, StopsAtTheLimit)
{
	setF("MC_ATT_TRIM_I", 5.f);
	setF("MC_ATT_TRIM_LIM", 0.05f);
	TestTrim trim;

	for (int i = 0; i < 4000; i++) {
		trim.apply(Vector3f(0.5f, -0.5f, 0.f), kDt);
	}

	EXPECT_FLOAT_EQ(trim.trim()(0), 0.05f);
	EXPECT_FLOAT_EQ(trim.trim()(1), -0.05f);
}

TEST_F(TorqueTrimTest, FreezesWhileTheAllocatorSaturates)
{
	setF("MC_ATT_TRIM_I", 1.f);
	TestTrim trim;
	const Vector3f law(0.04f, 0.04f, 0.f);

	for (int i = 0; i < 100; i++) { trim.apply(law, kDt); }

	const Vector2f before = trim.trim();
	trim.setSaturated(true);

	for (int i = 0; i < 100; i++) { trim.apply(law, kDt); }

	EXPECT_EQ(trim.trim(), before);

	trim.setSaturated(false);
	trim.apply(law, kDt);
	EXPECT_GT(trim.trim()(0), before(0));
}

/// Integrating into a clamped output buys nothing and has to be unwound; out of it must work.
TEST_F(TorqueTrimTest, DoesNotWindIntoTheOutputClamp)
{
	setF("MC_ATT_TRIM_I", 1.f);
	setF("MC_ATT_TRIM_LIM", 0.5f);
	TestTrim trim;

	for (int i = 0; i < 4000; i++) { trim.apply(Vector3f(1.2f, 0.f, 0.f), kDt); }

	EXPECT_EQ(trim.trim()(0), 0.f) << "law alone already past the clamp";

	// Opposite sign: integrating out of nothing toward the other side is allowed.
	trim.apply(Vector3f(-0.5f, 0.f, 0.f), kDt);
	EXPECT_LT(trim.trim()(0), 0.f);
}

TEST_F(TorqueTrimTest, ResetDropsTheTrimAndBadDtIsIgnored)
{
	setF("MC_ATT_TRIM_I", 1.f);
	TestTrim trim;

	trim.apply(Vector3f(0.05f, 0.f, 0.f), 0.5f);	// scheduling stall, not an interval
	trim.apply(Vector3f(0.05f, 0.f, 0.f), NAN);
	EXPECT_EQ(trim.trim()(0), 0.f);

	trim.apply(Vector3f(0.05f, 0.f, 0.f), kDt);
	EXPECT_GT(trim.trim()(0), 0.f);

	trim.reset();
	EXPECT_EQ(trim.trim().norm(), 0.f);

	// A NaN law torque passes through and never reaches the integral.
	const Vector3f out = trim.apply(Vector3f(NAN, 0.f, 0.f), kDt);
	EXPECT_FALSE(PX4_ISFINITE(out(0)));
	EXPECT_EQ(trim.trim()(0), 0.f);
}

} // namespace
