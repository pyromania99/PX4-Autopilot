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
 * @file CascadedPdControllerTest.cpp
 *
 * There is deliberately no differential test against the stock cascade here. The
 * previous controller at MC_CTRL_ALG=1 wrapped the stock objects and could be compared
 * to them; this one is an independent control law and any such comparison would only
 * measure how far the two have been allowed to diverge, which is the whole point.
 *
 * What is tested instead is the set of properties the law is supposed to have, plus the
 * fail-closed behaviour that matters because this class is also the framework's
 * always-allocated failsafe.
 */

#include <gtest/gtest.h>

#include "CascadedPdController.hpp"

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
	setParam("MC_PD_XY_P", 0.50f);
	setParam("MC_PD_XY_D", 1.00f);
	setParam("MC_PD_Z_P", 4.0f);
	setParam("MC_PD_Z_D", 4.0f);
	setParam("MC_PD_ATT_P", 0.6f);
	setParam("MC_PD_ATT_D", 0.15f);
	setParam("MC_PD_YAWR_D", 0.0f);
}

/// Hovering, level, stationary, at the origin, 100 m up.
ControllerState hoverState()
{
	ControllerState state{};
	state.q = Quatf(1.f, 0.f, 0.f, 0.f);
	state.angular_velocity.setZero();
	state.angular_accel.setZero();
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

/// Tilt angle of an attitude setpoint away from level, in radians.
float tiltOf(const Quatf &q)
{
	const Vector3f body_z = Dcmf(q).col(2);
	return acosf(math::constrain(body_z.dot(Vector3f(0.f, 0.f, 1.f)), -1.f, 1.f));
}

class CascadedPdControllerTest : public ::testing::Test
{
public:
	void SetUp() override
	{
		// Without this the first param write blocks forever: autosave schedules onto
		// wq:lp_default, which the gtest harness does not start. Same reason
		// CommandFrontEndTest and VehicleStateProviderTest disable it.
		param_control_autosave(false);
		resetParams();
		_harness = new ParamHarness();
		_controller = new CascadedPdController(_harness);
	}

	void TearDown() override
	{
		delete _controller;
		delete _harness;
		resetParams();
	}

	/// Apply a parameter change and push it into the controller.
	void reconfigure(const char *name, float value)
	{
		setParam(name, value);
		_harness->updateParams();
	}

	ParamHarness *_harness{nullptr};
	CascadedPdController *_controller{nullptr};
};

// ---------------------------------------------------------------------------
// Stage 3: the geometric attitude error
// ---------------------------------------------------------------------------

/**
 * The law depends on Dcm::vee() being the standard vee, matching the
 * [-S(1,2), S(0,2), -S(0,1)] the prototype writes by hand. If matrix ever changed that
 * convention the controller would silently invert two axes, so pin it here rather than
 * in a comment.
 */
TEST_F(CascadedPdControllerTest, VeeMatchesTheStandardConvention)
{
	const Vector3f v(0.3f, -0.7f, 1.1f);
	const Vector3f recovered = Dcmf(v.hat()).vee();
	EXPECT_NEAR(recovered(0), v(0), 1e-6f);
	EXPECT_NEAR(recovered(1), v(1), 1e-6f);
	EXPECT_NEAR(recovered(2), v(2), 1e-6f);

	// And that hat() really is the skew-symmetric form the formula assumes.
	const Dcmf S(v.hat());
	EXPECT_NEAR(S(2, 1), v(0), 1e-6f);
	EXPECT_NEAR(S(0, 2), v(1), 1e-6f);
	EXPECT_NEAR(S(1, 0), v(2), 1e-6f);
}

/**
 * At the Attitude level the controller keeps the pilot's tilt but rebuilds the heading
 * from the vehicle, so a pure roll offset must produce a roll torque of the expected
 * sign and nothing else worth speaking of.
 */
TEST_F(CascadedPdControllerTest, RollErrorProducesOpposingRollTorque)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::Attitude;
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);

	// Vehicle is rolled +10 deg, commanded level.
	state.q = Quatf(Eulerf(math::radians(10.f), 0.f, 0.f));
	state.heading = 0.f;
	command.attitude_sp = Quatf(1.f, 0.f, 0.f, 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	ASSERT_TRUE(output.valid);

	// Rolled positive, so the correction must be negative.
	EXPECT_LT(output.torque(0), -1e-3f);
	EXPECT_NEAR(output.torque(1), 0.f, 1e-4f);
	EXPECT_NEAR(output.torque(2), 0.f, 1e-6f);

	// Mirror image: rolled -10 deg gives the opposite torque, same magnitude.
	const float torque_positive_roll = output.torque(0);
	state.q = Quatf(Eulerf(math::radians(-10.f), 0.f, 0.f));
	output.reset();
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(output.torque(0), -torque_positive_roll, 1e-5f);
}

/**
 * The implied-rate-setpoint form must be algebraically identical to the law it stands in
 * for: tau == -Kp*e_R - Kd*w. Checked with a non-zero body rate so both terms are live.
 */
TEST_F(CascadedPdControllerTest, TorqueEqualsProportionalPlusRateDamping)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::Attitude;
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);
	command.attitude_sp = Quatf(1.f, 0.f, 0.f, 0.f);

	state.q = Quatf(Eulerf(math::radians(6.f), math::radians(-4.f), 0.f));
	state.angular_velocity = Vector3f(0.2f, -0.1f, 0.05f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	// Recompute e_R independently of the controller.
	const Dcmf R(state.q);
	const Dcmf R_des(output.attitude_setpoint);
	const Vector3f e_R = 0.5f * Dcmf(R_des.transpose() * R - R.transpose() * R_des).vee();

	const float kp = 0.6f;	// MC_PD_ATT_P
	const float kd = 0.15f;	// MC_PD_ATT_D

	EXPECT_NEAR(output.torque(0), -kp * e_R(0) - kd * state.angular_velocity(0), 1e-5f);
	EXPECT_NEAR(output.torque(1), -kp * e_R(1) - kd * state.angular_velocity(1), 1e-5f);
}

// ---------------------------------------------------------------------------
// Yaw is not controlled
// ---------------------------------------------------------------------------

/**
 * The defining property of this controller. A 90 deg heading offset from the commanded
 * yaw must produce no yaw torque at all, because R_des is rebuilt around the vehicle's
 * own heading and the yaw axis has no proportional term.
 */
TEST_F(CascadedPdControllerTest, LargeHeadingErrorProducesNoYawTorque)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(0.f, 0.f, math::radians(90.f)));
	state.heading = math::radians(90.f);

	ControllerCommand command = hoverCommand(state);
	command.yaw_sp = 0.f;	// asking for north while pointing east

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	ASSERT_TRUE(output.valid);

	EXPECT_NEAR(output.torque(2), 0.f, 1e-6f);
	EXPECT_NEAR(output.rate_setpoint(2), 0.f, 1e-6f);

	// The attitude setpoint follows the vehicle's heading, not the command's.
	const Eulerf euler(Quatf(output.attitude_setpoint));
	EXPECT_NEAR(euler.psi(), math::radians(90.f), 1e-3f);
}

/// With MC_PD_YAWR_D raised, the only yaw action available is damping the yaw rate.
TEST_F(CascadedPdControllerTest, YawTorqueIsPureRateDamping)
{
	reconfigure("MC_PD_YAWR_D", 0.2f);

	ControllerState state = hoverState();
	state.angular_velocity = Vector3f(0.f, 0.f, 1.5f);
	ControllerCommand command = hoverCommand(state);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	EXPECT_NEAR(output.torque(2), -0.2f * 1.5f, 1e-5f);

	// Doubling the yaw rate doubles the torque: no proportional term hiding anywhere.
	state.angular_velocity(2) = 3.0f;
	output.reset();
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(output.torque(2), -0.2f * 3.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Stages 1 and 2: position PD, thrust and tilt
// ---------------------------------------------------------------------------

/// At the setpoint with no velocity, the collective is the hover thrust and nothing tilts.
TEST_F(CascadedPdControllerTest, HoverEquilibriumCommandsHoverThrustAndNoTorque)
{
	const ControllerState state = hoverState();
	const ControllerCommand command = hoverCommand(state);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	ASSERT_TRUE(output.valid);

	EXPECT_NEAR(output.thrust(0), 0.f, 1e-6f);
	EXPECT_NEAR(output.thrust(1), 0.f, 1e-6f);
	EXPECT_NEAR(output.thrust(2), -state.hover_thrust, 1e-4f);

	EXPECT_NEAR(output.torque(0), 0.f, 1e-5f);
	EXPECT_NEAR(output.torque(1), 0.f, 1e-5f);
	EXPECT_NEAR(output.torque(2), 0.f, 1e-6f);

	EXPECT_NEAR(tiltOf(Quatf(output.attitude_setpoint)), 0.f, 1e-4f);
}

/**
 * A position error north must tilt the vehicle nose-down (pitch negative in NED), by the
 * angle the PD actually asked for.
 *
 * The exact angle matters: asserting only the sign let a bug through where every tilt
 * came out pinned at the tilt limit, because limitTilt() was handed an unnormalized
 * vector and acosf() returned NaN.
 */
TEST_F(CascadedPdControllerTest, NorthPositionErrorTiltsForward)
{
	const ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp(0) += 5.f;

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	// a_north = MC_PD_XY_P * 5 = 2.5 m/s^2, well inside the 45 deg limit.
	const float expected_tilt = atan2f(0.5f * 5.f, CONSTANTS_ONE_G);
	ASSERT_LT(expected_tilt, command.tilt_limit);

	const Eulerf euler(Quatf(output.attitude_setpoint));
	EXPECT_NEAR(euler.theta(), -expected_tilt, 1e-4f);
	EXPECT_NEAR(euler.phi(), 0.f, 1e-4f);
	EXPECT_NEAR(tiltOf(Quatf(output.attitude_setpoint)), expected_tilt, 1e-4f);
}

/// However big the demand, the commanded attitude must respect the framework's tilt limit.
TEST_F(CascadedPdControllerTest, TiltIsClampedToTheCommandedLimit)
{
	const ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.tilt_limit = math::radians(25.f);
	command.position_sp(0) += 1000.f;	// far beyond anything the gains can serve

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	EXPECT_NEAR(tiltOf(Quatf(output.attitude_setpoint)), math::radians(25.f), 1e-3f);
}

/// Damping acts on the estimator velocity, so descending at the setpoint still commands
/// more thrust than hover.
TEST_F(CascadedPdControllerTest, DampingUsesEstimatorVelocity)
{
	ControllerState state = hoverState();
	state.velocity(2) = 2.f;	// NED: falling at 2 m/s
	const ControllerCommand command = hoverCommand(state);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	// a_z = -MC_PD_Z_D * v_z = -8 m/s^2, i.e. accelerate upward, so more collective.
	EXPECT_LT(output.thrust(2), -state.hover_thrust);
}

/// Collective stays inside the framework's normalized envelope even under a demand that
/// would otherwise blow straight through it.
TEST_F(CascadedPdControllerTest, ThrustIsClampedToTheCommandedEnvelope)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.thrust_min = 0.15f;
	command.thrust_max = 0.75f;

	// Huge climb demand.
	command.position_sp(2) -= 500.f;
	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(output.thrust(2), -command.thrust_max, 1e-5f);

	// Huge descent demand.
	command.position_sp(2) += 1000.f;
	state.freshness.position_new = true;
	output.reset();
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(output.thrust(2), -command.thrust_min, 1e-5f);
}

// ---------------------------------------------------------------------------
// Levels
// ---------------------------------------------------------------------------

TEST_F(CascadedPdControllerTest, DeclaresFullStackSupport)
{
	EXPECT_EQ(_controller->supportedLevels(), kAllLevels);
	EXPECT_TRUE(_controller->supportsLevel(ControlLevel::Trajectory));
	EXPECT_TRUE(_controller->supportsLevel(ControlLevel::Attitude));
	EXPECT_TRUE(_controller->supportsLevel(ControlLevel::BodyRate));
	EXPECT_FALSE(_controller->supportsLevel(ControlLevel::None));

	// The law is cheap enough to run entirely on the gyro-rate queue.
	EXPECT_FALSE(_controller->hasOuterStage());
}

/// In Acro the D gains double as rate gains: tau == Kd * (rate_sp - w).
TEST_F(CascadedPdControllerTest, BodyRateLevelIsPureRateDamping)
{
	reconfigure("MC_PD_YAWR_D", 0.1f);

	ControllerState state = hoverState();
	state.angular_velocity = Vector3f(0.1f, 0.2f, -0.3f);

	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::BodyRate;
	command.rate_sp = Vector3f(0.5f, -0.5f, 1.0f);
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.6f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	ASSERT_TRUE(output.valid);

	const Vector3f kd(0.15f, 0.15f, 0.1f);
	const Vector3f expected = kd.emult(command.rate_sp - state.angular_velocity);
	EXPECT_NEAR(output.torque(0), expected(0), 1e-6f);
	EXPECT_NEAR(output.torque(1), expected(1), 1e-6f);
	EXPECT_NEAR(output.torque(2), expected(2), 1e-6f);

	EXPECT_NEAR(output.thrust(2), -0.6f, 1e-6f);
}

TEST_F(CascadedPdControllerTest, NoneLevelRefusesToProduceOutput)
{
	const ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::None;

	ControllerOutput output{};
	EXPECT_FALSE(_controller->update(state, command, 0.0025f, output));
	EXPECT_FALSE(output.valid);
}

// ---------------------------------------------------------------------------
// Fail-closed behaviour: this class is the framework's failsafe
// ---------------------------------------------------------------------------

/**
 * An unset axis must contribute no error rather than poisoning the whole vector. This is
 * how Altitude mode (no XY setpoint) works, and a NaN escaping here would reach the
 * allocator with nothing behind it to catch.
 */
TEST_F(CascadedPdControllerTest, PartialSetpointsStillProduceFiniteOutput)
{
	const ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);

	// Altitude mode: climb rate only, no horizontal setpoint at all.
	command.position_sp = Vector3f(NAN, NAN, state.position(2));
	command.velocity_sp = Vector3f(NAN, NAN, -1.f);
	command.axis_position = false;
	command.axis_climb_rate = true;

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	ASSERT_TRUE(output.valid);
	EXPECT_TRUE(outputIsFinite(output));
}

/// Every setpoint NaN, every estimate NaN: still finite, still bounded.
TEST_F(CascadedPdControllerTest, FullyUnsetSetpointStillProducesFiniteOutput)
{
	ControllerState state = hoverState();
	state.position = Vector3f(NAN, NAN, NAN);
	state.velocity = Vector3f(NAN, NAN, NAN);
	state.heading = NAN;
	state.position_valid_xy = false;
	state.position_valid_z = false;

	ControllerCommand command = hoverCommand(state);
	command.position_sp = Vector3f(NAN, NAN, NAN);
	command.velocity_sp = Vector3f(NAN, NAN, NAN);
	command.acceleration_sp = Vector3f(NAN, NAN, NAN);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_TRUE(outputIsFinite(output));

	// With no error anywhere it should be asking for level flight at hover thrust.
	EXPECT_NEAR(output.thrust(2), -state.hover_thrust, 1e-4f);
}

/// Torque is normalized; the allocator must never see a value outside [-1, 1].
TEST_F(CascadedPdControllerTest, TorqueIsSaturatedToTheNormalizedRange)
{
	reconfigure("MC_PD_ATT_P", 10.f);

	ControllerState state = hoverState();
	// Inverted, and spinning hard on every axis.
	state.q = Quatf(Eulerf(math::radians(170.f), 0.f, 0.f));
	state.angular_velocity = Vector3f(50.f, -50.f, 50.f);

	const ControllerCommand command = hoverCommand(state);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	for (int i = 0; i < 3; i++) {
		EXPECT_LE(output.torque(i), 1.f);
		EXPECT_GE(output.torque(i), -1.f);
	}

	EXPECT_TRUE(outputIsFinite(output));
}

/**
 * reset_integrals is a hard contract. There are no integrators to zero, but the cached
 * attitude setpoint must not survive it - otherwise a mode change commands the previous
 * mode's attitude for one cycle.
 */
TEST_F(CascadedPdControllerTest, ResetIntegralsDropsTheCachedPositionStage)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp(0) += 20.f;

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	ASSERT_GT(tiltOf(Quatf(output.attitude_setpoint)), math::radians(1.f));

	// No fresh position sample, so without a reset the cached tilted setpoint is reused.
	state.freshness.position_new = false;
	output.reset();
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_GT(tiltOf(Quatf(output.attitude_setpoint)), math::radians(1.f));

	// With the reset, the position stage must re-run even on a stale sample, and it now
	// sees the vehicle back at its setpoint.
	command.reset_integrals = true;
	command.position_sp = state.position;
	output.reset();
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(tiltOf(Quatf(output.attitude_setpoint)), 0.f, 1e-3f);
}

/// reset() must leave nothing behind for the next arm.
TEST_F(CascadedPdControllerTest, ResetClearsCachedState)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp(1) += 30.f;

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	ASSERT_GT(tiltOf(Quatf(output.attitude_setpoint)), math::radians(1.f));

	_controller->reset();

	vehicle_local_position_setpoint_s sp{};
	_controller->getLocalPositionSetpoint(sp);
	EXPECT_FALSE(PX4_ISFINITE(sp.x));
	EXPECT_FALSE(PX4_ISFINITE(sp.y));
	EXPECT_FALSE(PX4_ISFINITE(sp.z));

	// yaw is never controlled, so it is never reported.
	EXPECT_FALSE(PX4_ISFINITE(sp.yaw));
}

/**
 * MC_PD_ATT_D == 0 must not silently disable roll and pitch. The inner loop is factored
 * as Kd*(w_sp - w), which degenerates at Kd == 0, so the gain is floored in updateParams().
 * Yaw is deliberately NOT floored - zero there is the default and means what it says.
 */
TEST_F(CascadedPdControllerTest, ZeroAttitudeDampingIsFlooredButZeroYawDampingIsNot)
{
	reconfigure("MC_PD_ATT_D", 0.f);

	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(math::radians(10.f), 0.f, 0.f));
	state.angular_velocity = Vector3f(0.f, 0.f, 2.f);

	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::Attitude;
	command.attitude_sp = Quatf(1.f, 0.f, 0.f, 0.f);
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	// Roll still corrects, at the floored gain: tau = -Kp*e_R - 0.01*w.
	const Dcmf R(state.q);
	const Dcmf R_des(output.attitude_setpoint);
	const Vector3f e_R = 0.5f * Dcmf(R_des.transpose() * R - R.transpose() * R_des).vee();
	EXPECT_NEAR(output.torque(0), -0.6f * e_R(0) - 0.01f * state.angular_velocity(0), 1e-5f);
	EXPECT_LT(output.torque(0), -1e-3f);

	// Yaw damping really is off despite a 2 rad/s spin.
	EXPECT_NEAR(output.torque(2), 0.f, 1e-9f);
}

/// A PD law has no integrators, but the topic must keep arriving for its consumers.
TEST_F(CascadedPdControllerTest, RateControlStatusReportsZeroIntegrators)
{
	rate_ctrl_status_s status{};
	status.rollspeed_integ = 1.f;
	status.pitchspeed_integ = 2.f;
	status.yawspeed_integ = 3.f;

	_controller->getRateControlStatus(status);

	EXPECT_EQ(status.rollspeed_integ, 0.f);
	EXPECT_EQ(status.pitchspeed_integ, 0.f);
	EXPECT_EQ(status.yawspeed_integ, 0.f);
}

} // namespace
