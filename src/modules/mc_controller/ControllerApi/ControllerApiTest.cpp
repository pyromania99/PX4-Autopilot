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
 * Stage 3: interface freeze gate.
 *
 * Proves the abstract class is implementable exactly as specified and that the
 * defaults fail closed. Everything downstream depends on these types, so this
 * test guards the frozen contract.
 *
 * Disposition: PERMANENT.
 */

#include <gtest/gtest.h>

#include "MulticopterControllerBase.hpp"

using namespace mc_ctrl;

namespace
{

/// Minimal conforming controller: proves the interface is actually implementable.
class NullController : public MulticopterControllerBase
{
public:
	explicit NullController(ModuleParams *parent) : MulticopterControllerBase(parent) {}

	const char *name() const override { return "null"; }
	uint8_t supportedLevels() const override { return kAllLevels; }
	void reset() override { reset_calls++; }

	bool update(const ControllerState &, const ControllerCommand &command, float,
		    ControllerOutput &output) override
	{
		if (command.reset_integrals) {
			integral.setZero();
		}

		output.torque = matrix::Vector3f{0.f, 0.f, 0.f};
		output.thrust = matrix::Vector3f{0.f, 0.f, -0.5f};
		output.valid = true;
		return true;
	}

	int reset_calls{0};
	matrix::Vector3f integral{1.f, 1.f, 1.f};
};

/// A controller that only handles the inner loop, to check mask composition.
class RateOnlyController : public MulticopterControllerBase
{
public:
	explicit RateOnlyController(ModuleParams *parent) : MulticopterControllerBase(parent) {}
	const char *name() const override { return "rate_only"; }
	uint8_t supportedLevels() const override { return levelBit(ControlLevel::BodyRate); }
	void reset() override {}
	bool update(const ControllerState &, const ControllerCommand &, float, ControllerOutput &) override
	{
		return false;
	}
};

} // namespace

TEST(ControllerApiTest, OutputDefaultsFailClosed)
{
	ControllerOutput out;

	// A controller that forgets to write its output must not command zero torque -
	// it must be rejected by the output stage.
	EXPECT_FALSE(out.valid);
	EXPECT_FALSE(PX4_ISFINITE(out.torque(0)));
	EXPECT_FALSE(PX4_ISFINITE(out.thrust(2)));
	EXPECT_FALSE(outputIsFinite(out));
}

TEST(ControllerApiTest, OutputResetRestoresFailClosedState)
{
	ControllerOutput out;
	out.torque = matrix::Vector3f{0.1f, 0.2f, 0.3f};
	out.thrust = matrix::Vector3f{0.f, 0.f, -0.5f};
	out.valid = true;
	ASSERT_TRUE(outputIsFinite(out));

	out.reset();
	EXPECT_FALSE(out.valid);
	EXPECT_FALSE(outputIsFinite(out));
}

TEST(ControllerApiTest, OutputFiniteCheckTorqueAndThrust)
{
	ControllerOutput out;
	out.torque = matrix::Vector3f{0.f, 0.f, 0.f};
	out.thrust = matrix::Vector3f{0.f, 0.f, -0.5f};
	EXPECT_TRUE(outputIsFinite(out));

	out.torque(1) = NAN;
	EXPECT_FALSE(outputIsFinite(out));

	out.torque(1) = INFINITY;
	EXPECT_FALSE(outputIsFinite(out));
}

TEST(ControllerApiTest, ThrustIsMandatoryNotJustTorque)
{
	// land_detector and mc_hover_thrust_estimator consume vehicle_thrust_setpoint, so a
	// controller that fills only torque must be rejected rather than quietly published.
	ControllerOutput out;
	out.torque = matrix::Vector3f{0.f, 0.f, 0.f};
	EXPECT_FALSE(outputIsFinite(out)) << "thrust left NaN must be rejected";

	out.thrust = matrix::Vector3f{0.f, 0.f, -0.5f};
	EXPECT_TRUE(outputIsFinite(out));
}

TEST(ControllerApiTest, LevelBitsAreDistinctAndComposable)
{
	EXPECT_EQ(levelBit(ControlLevel::None), 1u);
	EXPECT_EQ(levelBit(ControlLevel::BodyRate), 2u);
	EXPECT_EQ(levelBit(ControlLevel::Attitude), 4u);
	EXPECT_EQ(levelBit(ControlLevel::Trajectory), 8u);

	EXPECT_EQ(kAllLevels, 2u | 4u | 8u);
	EXPECT_EQ(kAllLevels & levelBit(ControlLevel::None), 0u) << "None must not be a supportable level";
}

TEST(ControllerApiTest, SupportedLevelsGovernsSupportsLevel)
{
	NullController full{nullptr};
	EXPECT_TRUE(full.supportsLevel(ControlLevel::Trajectory));
	EXPECT_TRUE(full.supportsLevel(ControlLevel::Attitude));
	EXPECT_TRUE(full.supportsLevel(ControlLevel::BodyRate));

	RateOnlyController inner{nullptr};
	EXPECT_TRUE(inner.supportsLevel(ControlLevel::BodyRate));
	EXPECT_FALSE(inner.supportsLevel(ControlLevel::Attitude));
	EXPECT_FALSE(inner.supportsLevel(ControlLevel::Trajectory));
}

TEST(ControllerApiTest, ResetIntegralsContractIsReachable)
{
	NullController c{nullptr};
	ControllerState state;
	ControllerCommand command;
	ControllerOutput out;

	command.reset_integrals = false;
	ASSERT_TRUE(c.update(state, command, 0.004f, out));
	EXPECT_GT(c.integral.norm(), 0.f);

	command.reset_integrals = true;
	ASSERT_TRUE(c.update(state, command, 0.004f, out));
	EXPECT_FLOAT_EQ(c.integral.norm(), 0.f);
}

TEST(ControllerApiTest, ControllerCommandDefaultsAreUncommanded)
{
	ControllerCommand cmd;

	EXPECT_EQ(cmd.level, ControlLevel::None);
	EXPECT_FALSE(PX4_ISFINITE(cmd.position_sp(0)));
	EXPECT_FALSE(PX4_ISFINITE(cmd.velocity_sp(1)));
	EXPECT_FALSE(PX4_ISFINITE(cmd.yaw_sp));
	EXPECT_FALSE(PX4_ISFINITE(cmd.yawspeed_sp));
	EXPECT_FALSE(cmd.reset_integrals);
}

TEST(ControllerApiTest, StateDefaultsAreConservative)
{
	ControllerState s;

	// Default-constructed state must not claim the vehicle is flying or that
	// unavailable estimates are usable.
	EXPECT_TRUE(s.landed);
	EXPECT_TRUE(s.maybe_landed);
	EXPECT_FALSE(s.armed);
	EXPECT_FALSE(s.spooled_up);
	EXPECT_FALSE(s.position_valid_xy);
	EXPECT_FALSE(s.hover_thrust_valid);
	EXPECT_FALSE(s.resets.any());
}

TEST(ControllerApiTest, EkfResetsClearAndAny)
{
	EkfResets r;
	EXPECT_FALSE(r.any());

	r.heading = true;
	r.delta_heading = 0.3f;
	EXPECT_TRUE(r.any());

	r.clear();
	EXPECT_FALSE(r.any());
	EXPECT_FLOAT_EQ(r.delta_heading, 0.f);
}

TEST(ControllerApiTest, LevelNamesAreStable)
{
	EXPECT_STREQ(levelName(ControlLevel::None), "none");
	EXPECT_STREQ(levelName(ControlLevel::BodyRate), "body_rate");
	EXPECT_STREQ(levelName(ControlLevel::Attitude), "attitude");
	EXPECT_STREQ(levelName(ControlLevel::Trajectory), "trajectory");
}

TEST(ControllerApiTest, StateStaysWithinRealTimeCopyBudget)
{
	// Copied every gyro cycle on the rate_ctrl work queue.
	EXPECT_LE(sizeof(ControllerState), 512u) << "actual: " << sizeof(ControllerState);
	EXPECT_LE(sizeof(ControllerCommand), 512u) << "actual: " << sizeof(ControllerCommand);
}
