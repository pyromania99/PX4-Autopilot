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
 * @file TrajectoryStageTest.cpp
 *
 * Two things carry most of the weight here:
 *
 *  - the INTEGRATOR, which is new. Its failure modes are all silent: winding up on the
 *    ground and leaping at takeoff, integrating an estimator jump as though it were a
 *    tracking error, or stepping the collective every time the hover thrust estimate
 *    updates. Each has a test.
 *  - the COLLECTIVE, because this stage changed the formula EigenController used to
 *    carry. The properties asserted are the derivation itself (the WHOLE demand divided
 *    by cos(tilt), not just the hover term) and the absence of any boost outside the
 *    valid cone.
 *
 * With both integral gains at zero the stage must reproduce the PD law byte for byte -
 * that is what makes the refactor a refactor, and it is the first test below.
 */

#include <gtest/gtest.h>

#include "TrajectoryStage.hpp"

#include <geo/geo.h>
#include <parameters/param.h>
#include <px4_platform_common/defines.h>

using namespace matrix;
using namespace mc_ctrl;

namespace
{

class ParamHarness : public ModuleParams
{
public:
	ParamHarness() : ModuleParams(nullptr) {}
	using ModuleParams::updateParams;
};

void setParam(const char *name, float value)
{
	param_set_no_notification(param_find(name), &value);
}

/// Restore the shipped defaults so one test's gain override cannot leak into the next.
void resetParams()
{
	setParam("MC_OL_XY_I", 0.0f);
	setParam("MC_OL_Z_I", 0.0f);
	setParam("MC_OL_I_LIM", 9.81f);
}

/// Gains a caller would pass in. Chosen distinct per axis so a transposition shows up.
const Vector3f kPosP{0.5f, 0.5f, 10.0f};
const Vector3f kPosD{1.0f, 1.0f, 5.3f};

/// Hovering, level, stationary, at the origin, 100 m up.
ControllerState hoverState()
{
	ControllerState state{};
	state.q = Quatf(1.f, 0.f, 0.f, 0.f);
	state.position = Vector3f(0.f, 0.f, -100.f);
	state.velocity.setZero();
	state.acceleration.setZero();
	state.heading = 0.f;
	state.position_valid_xy = true;
	state.position_valid_z = true;
	state.velocity_valid_xy = true;
	state.velocity_valid_z = true;
	state.landed = false;
	state.maybe_landed = false;
	state.armed = true;
	state.spooled_up = true;
	state.hover_thrust = 0.5f;
	state.hover_thrust_valid = true;
	state.freshness.dt = 0.0025f;
	state.freshness.dt_attitude = 0.004f;
	state.freshness.dt_position = 0.02f;
	state.freshness.attitude_new = true;
	state.freshness.position_new = true;
	return state;
}

/// Hold position at the state's own location, so the PD is at equilibrium.
ControllerCommand hoverCommand(const ControllerState &state)
{
	ControllerCommand command{};
	command.level = ControlLevel::Trajectory;
	command.position_sp = state.position;
	command.velocity_sp = Vector3f(NAN, NAN, NAN);
	command.acceleration_sp = Vector3f(NAN, NAN, NAN);
	command.yaw_sp = 0.f;
	command.axis_position = true;
	command.axis_altitude = true;
	command.thrust_min = 0.12f;
	command.thrust_max = 1.f;
	command.tilt_limit = math::radians(45.f);
	command.automatic = true;
	return command;
}

/// Roll the vehicle, leaving pitch and yaw alone, so Dcm(2,2) == cos(roll).
Quatf rolled(float radians)
{
	return Quatf(Eulerf(radians, 0.f, 0.f));
}

class TrajectoryStageTest : public ::testing::Test
{
public:
	void SetUp() override
	{
		// Without this the first param write blocks forever: autosave schedules onto
		// wq:lp_default, which the gtest harness does not start.
		param_control_autosave(false);
		resetParams();
		_harness = new ParamHarness();
		_stage = new TrajectoryStage(_harness);
	}

	void TearDown() override
	{
		delete _stage;
		delete _harness;
		resetParams();
	}

	/// Apply a parameter change and push it into the stage.
	void reconfigure(const char *name, float value)
	{
		setParam(name, value);
		_harness->updateParams();
	}

	ParamHarness *_harness{nullptr};
	TrajectoryStage *_stage{nullptr};
};

// ---------------------------------------------------------------------------------------
// Stage 1, with the integrator off: the refactor must be a refactor.
// ---------------------------------------------------------------------------------------

TEST_F(TrajectoryStageTest, GainsAtZeroReproduceThePdLaw)
{
	ControllerState state = hoverState();
	state.position = Vector3f(1.f, -2.f, -97.f);
	state.velocity = Vector3f(0.3f, 0.4f, -0.5f);

	ControllerCommand command = hoverCommand(state);
	command.position_sp = Vector3f(4.f, 1.f, -100.f);

	const Vector3f acceleration = _stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);

	for (int i = 0; i < 3; i++) {
		const float position_error = command.position_sp(i) - state.position(i);
		const float velocity_error = -state.velocity(i);
		EXPECT_NEAR(acceleration(i), kPosP(i) * position_error + kPosD(i) * velocity_error, 1e-5f) << "axis " << i;
	}

	EXPECT_FLOAT_EQ(_stage->integral()(0), 0.f);
	EXPECT_FLOAT_EQ(_stage->integral()(1), 0.f);
	EXPECT_FLOAT_EQ(_stage->integral()(2), 0.f);
}

TEST_F(TrajectoryStageTest, UnsetPositionAxisDampsAbsoluteVelocity)
{
	ControllerState state = hoverState();
	state.velocity = Vector3f(2.f, 0.f, 0.f);

	ControllerCommand command = hoverCommand(state);
	command.position_sp(0) = NAN;

	const Vector3f acceleration = _stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);

	// No position error contribution, only the damping term.
	EXPECT_NEAR(acceleration(0), kPosD(0) * -2.f, 1e-5f);
}

TEST_F(TrajectoryStageTest, VelocityOnlyAxisTracksItsVelocitySetpoint)
{
	ControllerState state = hoverState();
	state.velocity = Vector3f(0.f, 0.f, 1.f);

	ControllerCommand command = hoverCommand(state);
	command.position_sp(2) = NAN;
	command.velocity_sp(2) = -1.5f;

	const Vector3f acceleration = _stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);

	EXPECT_NEAR(acceleration(2), kPosD(2) * (-1.5f - 1.f), 1e-4f);
}

TEST_F(TrajectoryStageTest, AccelerationFeedforwardIsAdded)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.acceleration_sp = Vector3f(0.7f, NAN, NAN);

	const Vector3f acceleration = _stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);

	// At equilibrium the PD contributes nothing, so the feedforward is the whole answer.
	EXPECT_NEAR(acceleration(0), 0.7f, 1e-5f);
	EXPECT_NEAR(acceleration(1), 0.f, 1e-5f);
}

TEST_F(TrajectoryStageTest, LateralDemandIsBoundedByWhatTheTiltLimitDelivers)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(500.f, 500.f, 0.f);
	command.tilt_limit = math::radians(45.f);

	const Vector3f acceleration = _stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);

	// g*tan(45 deg) == g.
	EXPECT_NEAR(Vector2f(acceleration).norm(), CONSTANTS_ONE_G, 1e-3f);
}

// ---------------------------------------------------------------------------------------
// The integrator.
// ---------------------------------------------------------------------------------------

TEST_F(TrajectoryStageTest, IntegralAccumulatesAgainstAStandingPositionError)
{
	reconfigure("MC_OL_Z_I", 1.0f);

	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp(2) = state.position(2) - 2.f;	// 2 m below the setpoint

	float previous = 0.f;

	for (int i = 0; i < 10; i++) {
		_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
		const float now = _stage->integral()(2);
		EXPECT_LT(now, previous) << "a negative (upward) error must integrate downward, step " << i;
		previous = now;
	}

	// 10 steps of I * e * dt = 1.0 * -2 * 0.02.
	EXPECT_NEAR(_stage->integral()(2), -0.4f, 1e-4f);
}

TEST_F(TrajectoryStageTest, IntegralIsBoundedByTheLimit)
{
	reconfigure("MC_OL_Z_I", 50.0f);
	reconfigure("MC_OL_I_LIM", 1.5f);

	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp(2) = state.position(2) + 100.f;

	for (int i = 0; i < 200; i++) {
		_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	}

	EXPECT_NEAR(_stage->integral()(2), 1.5f, 1e-5f);
}

TEST_F(TrajectoryStageTest, IntegralOnlyAdvancesOnAFreshPositionSample)
{
	reconfigure("MC_OL_Z_I", 1.0f);

	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp(2) = state.position(2) + 2.f;

	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	const float after_fresh = _stage->integral()(2);
	EXPECT_GT(after_fresh, 0.f);

	// A gyro-rate cycle with no new position sample must not advance it: dt_position
	// describes an interval this stage has already consumed.
	state.freshness.position_new = false;

	for (int i = 0; i < 20; i++) {
		_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	}

	EXPECT_FLOAT_EQ(_stage->integral()(2), after_fresh);
}

TEST_F(TrajectoryStageTest, IntegralSkipsTheCycleAnEstimatorResetLandsOn)
{
	reconfigure("MC_OL_XY_I", 1.0f);
	reconfigure("MC_OL_Z_I", 1.0f);

	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(3.f, 0.f, 3.f);

	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	const Vector3f before = _stage->integral();

	// A jump in the estimate is not a tracking error; integrating it would wind the
	// integrator by the size of the discontinuity.
	state.resets.xy = true;
	state.resets.z = true;
	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);

	EXPECT_FLOAT_EQ(_stage->integral()(0), before(0));
	EXPECT_FLOAT_EQ(_stage->integral()(2), before(2));
}

TEST_F(TrajectoryStageTest, IntegralDrainsOnAnAxisThatStopsCommandingPosition)
{
	reconfigure("MC_OL_XY_I", 1.0f);

	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(3.f, 0.f, 0.f);

	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	EXPECT_GT(_stage->integral()(0), 0.f);

	// Switching to a velocity-only mode leaves no hold target to converge onto.
	command.position_sp(0) = NAN;
	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);

	EXPECT_FLOAT_EQ(_stage->integral()(0), 0.f);
}

TEST_F(TrajectoryStageTest, ResetIntegralZeroesItWithoutTouchingTheGains)
{
	reconfigure("MC_OL_Z_I", 1.0f);

	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp(2) = state.position(2) + 5.f;

	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	EXPECT_GT(_stage->integral()(2), 0.f);

	_stage->resetIntegral();
	EXPECT_FLOAT_EQ(_stage->integral()(2), 0.f);

	// Still integrating afterwards - resetIntegral() is not a disable.
	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	EXPECT_GT(_stage->integral()(2), 0.f);
}

TEST_F(TrajectoryStageTest, HoverThrustChangeLeavesTheCollectiveContinuous)
{
	reconfigure("MC_OL_Z_I", 1.0f);

	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);

	// Build a nonzero integral against a standing altitude error...
	command.position_sp(2) = state.position(2) - 2.f;

	for (int i = 0; i < 50; i++) {
		_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	}

	// ...then remove the error, so the integral holds steady and is the only thing left.
	command.position_sp(2) = state.position(2);
	Vector3f acceleration = _stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	const Vector3f before = _stage->computeThrustSetpoint(state, command, acceleration);
	const float integral_before = _stage->integral()(2);

	// The estimator revises hover thrust. Same integral would mean a different collective,
	// so the stage has to absorb the change.
	state.hover_thrust = 0.62f;
	acceleration = _stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	const Vector3f after = _stage->computeThrustSetpoint(state, command, acceleration);

	EXPECT_NE(_stage->integral()(2), integral_before) << "the integral must actually have been rescaled";
	EXPECT_NEAR(after(2), before(2), 1e-4f);
}

// ---------------------------------------------------------------------------------------
// The collective.
// ---------------------------------------------------------------------------------------

TEST_F(TrajectoryStageTest, CollectiveCompensatesBankAngle)
{
	ControllerState state = hoverState();
	state.q = rolled(math::radians(30.f));
	const ControllerCommand command = hoverCommand(state);

	// At equilibrium the acceleration setpoint is zero, so this is the pure hover term.
	const Vector3f thrust = _stage->computeThrustSetpoint(state, command, Vector3f{});

	EXPECT_NEAR(thrust(2), -0.5f / cosf(math::radians(30.f)), 1e-4f);
	EXPECT_FLOAT_EQ(thrust(0), 0.f);
	EXPECT_FLOAT_EQ(thrust(1), 0.f);
}

TEST_F(TrajectoryStageTest, CollectiveDividesTheWholeDemandNotJustTheHoverTerm)
{
	ControllerState state = hoverState();
	state.q = rolled(math::radians(30.f));
	const ControllerCommand command = hoverCommand(state);

	const float a_z = -2.f;	// NED down-positive, so this is a climb
	const Vector3f thrust = _stage->computeThrustSetpoint(state, command, Vector3f(0.f, 0.f, a_z));

	const float cos_tilt = cosf(math::radians(30.f));
	const float expected = (state.hover_thrust * (CONSTANTS_ONE_G - a_z)) / (CONSTANTS_ONE_G * cos_tilt);
	EXPECT_NEAR(thrust(2), -expected, 1e-4f);

	// The superseded form left the acceleration term undivided. Assert the difference is
	// real, so this test fails if that formula is ever restored.
	const float undivided = state.hover_thrust / cos_tilt - a_z * state.hover_thrust / CONSTANTS_ONE_G;
	EXPECT_GT(fabsf(expected - undivided), 1e-3f);
}

TEST_F(TrajectoryStageTest, CollectiveIsNotBoostedOutsideTheValidCone)
{
	ControllerState state = hoverState();
	const ControllerCommand command = hoverCommand(state);

	const float a_z = -2.f;
	const float undivided = fabsf(a_z * (state.hover_thrust / CONSTANTS_ONE_G) - state.hover_thrust);

	// Past ~84 deg the vertical channel is unavailable. Dividing would command up to 10x
	// the collective sideways; inverted it would drive the vehicle downward harder.
	for (const float roll_deg : {87.f, 100.f, 180.f}) {
		state.q = rolled(math::radians(roll_deg));
		const Vector3f thrust = _stage->computeThrustSetpoint(state, command, Vector3f(0.f, 0.f, a_z));
		EXPECT_NEAR(thrust(2), -undivided, 1e-4f) << "roll " << roll_deg << " deg";
	}
}

TEST_F(TrajectoryStageTest, CollectiveRespectsTheCommandedThrustEnvelope)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.thrust_min = 0.2f;
	command.thrust_max = 0.8f;

	const Vector3f hard_climb = _stage->computeThrustSetpoint(state, command, Vector3f(0.f, 0.f, -50.f));
	EXPECT_FLOAT_EQ(hard_climb(2), -0.8f);

	const Vector3f hard_drop = _stage->computeThrustSetpoint(state, command, Vector3f(0.f, 0.f, 50.f));
	EXPECT_FLOAT_EQ(hard_drop(2), -0.2f);
}

// ---------------------------------------------------------------------------------------
// Bookkeeping the laws depend on.
// ---------------------------------------------------------------------------------------

TEST_F(TrajectoryStageTest, StageValidityTracksRunsAndInvalidation)
{
	ControllerState state = hoverState();
	const ControllerCommand command = hoverCommand(state);

	EXPECT_FALSE(_stage->stageValid()) << "nothing has run yet";

	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	EXPECT_TRUE(_stage->stageValid());

	_stage->invalidateStage();
	EXPECT_FALSE(_stage->stageValid());

	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	EXPECT_TRUE(_stage->stageValid());

	_stage->reset();
	EXPECT_FALSE(_stage->stageValid());
}

TEST_F(TrajectoryStageTest, ResetDropsTheIntegralAndTheCachedSetpoints)
{
	reconfigure("MC_OL_Z_I", 1.0f);

	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp(2) = state.position(2) + 5.f;

	_stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);
	_stage->reset();

	EXPECT_FLOAT_EQ(_stage->integral()(2), 0.f);

	vehicle_local_position_setpoint_s sp{};
	_stage->fillLocalPositionSetpoint(sp, Quatf(1.f, 0.f, 0.f, 0.f), Vector3f(0.f, 0.f, -0.5f));
	EXPECT_FALSE(PX4_ISFINITE(sp.x));
	EXPECT_FALSE(PX4_ISFINITE(sp.z));
}

TEST_F(TrajectoryStageTest, LocalPositionSetpointReportsWhatTheStageUsed)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = Vector3f(3.f, 4.f, -80.f);
	command.velocity_sp = Vector3f(1.f, 0.f, -0.5f);

	const Vector3f acceleration = _stage->computeAccelerationSetpoint(state, command, kPosP, kPosD);

	vehicle_local_position_setpoint_s sp{};
	_stage->fillLocalPositionSetpoint(sp, Quatf(1.f, 0.f, 0.f, 0.f), Vector3f(0.f, 0.f, -0.5f));

	EXPECT_FLOAT_EQ(sp.x, 3.f);
	EXPECT_FLOAT_EQ(sp.y, 4.f);
	EXPECT_FLOAT_EQ(sp.z, -80.f);
	EXPECT_FLOAT_EQ(sp.vx, 1.f);
	EXPECT_FLOAT_EQ(sp.vz, -0.5f);
	EXPECT_NEAR(sp.acceleration[0], acceleration(0), 1e-5f);
	EXPECT_NEAR(sp.acceleration[2], acceleration(2), 1e-5f);

	// Level attitude, so body-FRD thrust passes into NED unchanged.
	EXPECT_NEAR(sp.thrust[2], -0.5f, 1e-5f);

	// No law that shares this stage controls heading.
	EXPECT_FALSE(PX4_ISFINITE(sp.yaw));
	EXPECT_FALSE(PX4_ISFINITE(sp.yawspeed));
}

// ---------------------------------------------------------------------------------------
// The heading both laws anchor their attitude setpoint on
// ---------------------------------------------------------------------------------------

/// The quaternion is the source, so the answer is exactly Eulerf(q).psi().
TEST_F(TrajectoryStageTest, CurrentHeadingComesFromTheAttitudeQuaternion)
{
	ControllerState state = hoverState();

	for (const float heading : {0.f, 0.7f, -2.4f, 3.0f}) {
		state.q = Quatf(Eulerf(0.2f, -0.3f, heading));
		EXPECT_NEAR(TrajectoryStage::currentHeading(state), heading, 1e-5f);
	}
}

/**
 * The whole point of the helper. state.heading is only republished on an EKF2 fusion step,
 * so between those the quaternion is the fresher of two otherwise identical angles. If this
 * ever preferred state.heading again, both laws would anchor their attitude setpoint on a
 * frame the vehicle had already rotated out of.
 */
TEST_F(TrajectoryStageTest, CurrentHeadingPrefersTheQuaternionOverAStaleStateHeading)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(0.f, 0.f, math::radians(30.f)));
	state.heading = 0.f;			// the value a slower fusion step left behind
	state.freshness.position_new = false;

	EXPECT_NEAR(TrajectoryStage::currentHeading(state), math::radians(30.f), 1e-5f);
}

/// The failsafe controller shares this helper, so a degenerate quaternion must still
/// produce a finite angle rather than a NaN that reaches the allocator.
TEST_F(TrajectoryStageTest, CurrentHeadingFallsBackWhenTheQuaternionIsUnusable)
{
	ControllerState state = hoverState();

	state.q = Quatf(NAN, NAN, NAN, NAN);
	state.heading = math::radians(45.f);
	EXPECT_NEAR(TrajectoryStage::currentHeading(state), math::radians(45.f), 1e-5f);

	state.q = Quatf(0.f, 0.f, 0.f, 0.f);
	EXPECT_NEAR(TrajectoryStage::currentHeading(state), math::radians(45.f), 1e-5f);

	// Neither source available: zero, not NaN.
	state.heading = NAN;
	EXPECT_TRUE(PX4_ISFINITE(TrajectoryStage::currentHeading(state)));
	EXPECT_NEAR(TrajectoryStage::currentHeading(state), 0.f, 1e-6f);
}

} // namespace
