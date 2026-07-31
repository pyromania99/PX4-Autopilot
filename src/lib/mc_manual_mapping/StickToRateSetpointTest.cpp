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
 * Stage 1 differential test.
 *
 * `referenceAcroMapping()` is a VERBATIM copy of the pre-extraction acro stick
 * mapping from MulticopterRateControl::Run() (MulticopterRateControl.cpp:160-169
 * at git 154dc85809).
 *
 * Disposition: PERMANENT regression guard.
 */

#include <gtest/gtest.h>

#include "StickToRateSetpoint.hpp"

#include <mathlib/math/Functions.hpp>
#include <mathlib/math/Limits.hpp>
#include <parameters/param.h>
#include <px4_platform_common/defines.h>

#include <random>

using namespace matrix;

namespace
{

/// Verbatim pre-extraction implementation, with the params passed in explicitly.
void referenceAcroMapping(const manual_control_setpoint_s &manual_control_setpoint, const Vector3f &acro_rate_max,
			  float expo, float supexpo, float expo_y, float supexpoy,
			  Vector3f &rates_setpoint, Vector3f &thrust_setpoint)
{
	const Vector3f man_rate_sp{
		math::superexpo(manual_control_setpoint.roll, expo, supexpo),
		math::superexpo(-manual_control_setpoint.pitch, expo, supexpo),
		math::superexpo(manual_control_setpoint.yaw, expo_y, supexpoy)};

	rates_setpoint = man_rate_sp.emult(acro_rate_max);
	thrust_setpoint(2) = -(manual_control_setpoint.throttle + 1.f) * .5f;
	thrust_setpoint(0) = thrust_setpoint(1) = 0.f;
}

/**
 * ModuleParams::updateParams() is protected and cascades to children, which is how
 * the owning module drives it. Parent the object under this harness so a param
 * change can be pushed through the real cascade path.
 */
class ParamHarness : public ModuleParams
{
public:
	ParamHarness() : ModuleParams(nullptr) {}
	using ModuleParams::updateParams;
};

float paramFloat(const char *name)
{
	float v = 0.f;
	param_get(param_find(name), &v);
	return v;
}

} // namespace

TEST(StickToRateSetpointTest, MatchesPreExtractionReference)
{
	param_control_autosave(false);

	ParamHarness harness;
	StickToRateSetpoint stick_to_rate{&harness};
	harness.updateParams();

	const Vector3f acro_rate_max{
		math::radians(paramFloat("MC_ACRO_R_MAX")),
		math::radians(paramFloat("MC_ACRO_P_MAX")),
		math::radians(paramFloat("MC_ACRO_Y_MAX"))};

	// The class must derive the same limits from the same parameters.
	EXPECT_NEAR(stick_to_rate.acroRateMax()(0), acro_rate_max(0), 1e-6f);
	EXPECT_NEAR(stick_to_rate.acroRateMax()(1), acro_rate_max(1), 1e-6f);
	EXPECT_NEAR(stick_to_rate.acroRateMax()(2), acro_rate_max(2), 1e-6f);

	const float expo = paramFloat("MC_ACRO_EXPO");
	const float supexpo = paramFloat("MC_ACRO_SUPEXPO");
	const float expo_y = paramFloat("MC_ACRO_EXPO_Y");
	const float supexpoy = paramFloat("MC_ACRO_SUPEXPOY");

	std::mt19937 rng{20260729u};
	std::uniform_real_distribution<float> stick{-1.f, 1.f};

	for (int step = 0; step < 5000; step++) {
		manual_control_setpoint_s manual{};
		manual.roll = stick(rng);
		manual.pitch = stick(rng);
		manual.yaw = stick(rng);
		manual.throttle = stick(rng);

		Vector3f rate_extracted{};
		Vector3f thrust_extracted{};
		Vector3f rate_reference{};
		Vector3f thrust_reference{};

		stick_to_rate.update(manual, rate_extracted, thrust_extracted);
		referenceAcroMapping(manual, acro_rate_max, expo, supexpo, expo_y, supexpoy,
				     rate_reference, thrust_reference);

		for (int i = 0; i < 3; i++) {
			ASSERT_FLOAT_EQ(rate_extracted(i), rate_reference(i))
					<< "rate_setpoint(" << i << ") diverged at step " << step;
			ASSERT_FLOAT_EQ(thrust_extracted(i), thrust_reference(i))
					<< "thrust_setpoint(" << i << ") diverged at step " << step;
		}
	}
}

TEST(StickToRateSetpointTest, CentredSticksGiveZeroRates)
{
	param_control_autosave(false);

	ParamHarness harness;
	StickToRateSetpoint stick_to_rate{&harness};
	harness.updateParams();

	manual_control_setpoint_s manual{};	// all sticks centred, throttle at -1
	Vector3f rate{};
	Vector3f thrust{};

	stick_to_rate.update(manual, rate, thrust);

	EXPECT_NEAR(rate(0), 0.f, 1e-6f);
	EXPECT_NEAR(rate(1), 0.f, 1e-6f);
	EXPECT_NEAR(rate(2), 0.f, 1e-6f);

	// throttle == 0 maps to -0.5 collective; x/y thrust must be exactly zero
	EXPECT_FLOAT_EQ(thrust(0), 0.f);
	EXPECT_FLOAT_EQ(thrust(1), 0.f);
	EXPECT_FLOAT_EQ(thrust(2), -0.5f);
}

TEST(StickToRateSetpointTest, FullStickReachesConfiguredMaxRate)
{
	param_control_autosave(false);

	ParamHarness harness;
	StickToRateSetpoint stick_to_rate{&harness};
	harness.updateParams();

	manual_control_setpoint_s manual{};
	manual.roll = 1.f;
	manual.pitch = -1.f;	// pitch is negated in the mapping
	manual.yaw = 1.f;

	Vector3f rate{};
	Vector3f thrust{};
	stick_to_rate.update(manual, rate, thrust);

	// superexpo(1) == 1 for any expo/superexpo shape, so full stick == the configured limit
	EXPECT_NEAR(rate(0), stick_to_rate.acroRateMax()(0), 1e-5f);
	EXPECT_NEAR(rate(1), stick_to_rate.acroRateMax()(1), 1e-5f);
	EXPECT_NEAR(rate(2), stick_to_rate.acroRateMax()(2), 1e-5f);
}
