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
 * @file EigenControllerTest.cpp
 *
 * Properties the eigen-dynamics law is supposed to have, not agreement with anything.
 * Three groups carry most of the weight:
 *
 *  - the CROSS-COUPLING, which is the entire reason this controller exists and which a
 *    conventional per-axis rate law would not have;
 *  - the GYROSCOPIC term, the one place the port deliberately departs from the Isaac Sim
 *    prototype, pinned here against a direct omega x (I*omega) evaluation;
 *  - the FRAME CONVERSION, because the prototype is ENU/FLU and PX4 is NED/FRD, and a
 *    sign error there is a controller that accelerates away from its setpoint.
 */

#include <gtest/gtest.h>

#include "EigenController.hpp"

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
	setParam("MC_EIG_XY_P", 0.50f);
	setParam("MC_EIG_XY_D", 1.00f);
	setParam("MC_EIG_Z_P", 10.0f);
	setParam("MC_EIG_Z_D", 5.3f);
	setParam("MC_EIG_ATT_P", 7.0f);
	setParam("MC_EIG_ATT_D", 1.0f);
	setParam("MC_EIG_WN", 18.0f);
	setParam("MC_EIG_B", 10.2f);
	setParam("MC_EIG_ALPHA", 0.5f);
	setParam("MC_EIG_BETA", 0.6f);
	setParam("MC_EIG_IXX", 0.01f);
	setParam("MC_EIG_IYY", 0.01f);
	setParam("MC_EIG_IZZ", 0.02f);
	setParam("MC_EIG_TRQ_MAX", 1.0f);
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

/// Acro-style command, so a rate setpoint reaches the inner loop untouched.
ControllerCommand rateCommand(const Vector3f &rate_sp)
{
	ControllerCommand command{};
	command.level = ControlLevel::BodyRate;
	command.rate_sp = rate_sp;
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);
	command.thrust_min = 0.f;
	command.thrust_max = 1.f;
	command.manual = true;
	return command;
}

/// Tilt angle of an attitude setpoint away from level, in radians.
float tiltOf(const Quatf &q)
{
	const Vector3f body_z = Dcmf(q).col(2);
	return acosf(math::constrain(body_z.dot(Vector3f(0.f, 0.f, 1.f)), -1.f, 1.f));
}

class EigenControllerTest : public ::testing::Test
{
public:
	void SetUp() override
	{
		// Without this the first param write blocks forever: autosave schedules onto
		// wq:lp_default, which the gtest harness does not start.
		param_control_autosave(false);
		resetParams();
		_harness = new ParamHarness();
		_controller = new EigenController(_harness);
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
	EigenController *_controller{nullptr};
};

// ---------------------------------------------------------------------------
// The cross-coupling: what makes this an eigen controller
// ---------------------------------------------------------------------------

/**
 * A pure ROLL rate error must produce a PITCH torque, and vice versa with the opposite
 * sign. That antisymmetry is the imaginary part of the eigenvalue pair -wn +/- j*b, and it
 * is the whole reason this law exists - a conventional per-axis rate controller would
 * produce nothing on the other axis.
 */
TEST_F(EigenControllerTest, RateErrorIsCrossCoupledAntisymmetrically)
{
	ControllerState state = hoverState();

	// Pure roll rate demand, vehicle at rest.
	ControllerOutput roll_output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, roll_output));

	// Own axis: wn * error, scaled by inertia.
	EXPECT_NEAR(roll_output.torque(0), 0.01f * 18.0f, 1e-5f);
	// Cross axis: -b * roll error. Non-zero is the point; the sign is the law.
	EXPECT_NEAR(roll_output.torque(1), 0.01f * -10.2f, 1e-5f);
	EXPECT_NEAR(roll_output.torque(2), 0.f, 1e-9f);

	// Pure pitch rate demand. Own axis identical, cross axis flipped.
	_controller->reset();
	ControllerOutput pitch_output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(0.f, 1.f, 0.f)), 0.0025f, pitch_output));

	EXPECT_NEAR(pitch_output.torque(1), 0.01f * 18.0f, 1e-5f);
	EXPECT_NEAR(pitch_output.torque(0), 0.01f * 10.2f, 1e-5f);

	// Antisymmetry: roll->pitch and pitch->roll are equal and opposite.
	EXPECT_NEAR(roll_output.torque(1), -pitch_output.torque(0), 1e-6f);
}

/**
 * MC_EIG_B == 0 collapses the law to two independent axes. Worth pinning because it is the
 * escape hatch documented in the parameter, and because it isolates the coupling.
 */
TEST_F(EigenControllerTest, ZeroCrossGainDecouplesTheAxes)
{
	reconfigure("MC_EIG_B", 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, output));

	EXPECT_NEAR(output.torque(0), 0.01f * 18.0f, 1e-5f);
	EXPECT_NEAR(output.torque(1), 0.f, 1e-9f);
}

// ---------------------------------------------------------------------------
// The gyroscopic term: where the port departs from the prototype
// ---------------------------------------------------------------------------

/**
 * With the rate error, the assumed drag and the yaw feedforward all zeroed, the only term
 * left is the gyroscopic one, which must equal omega x (I*omega) exactly for an asymmetric
 * inertia. This is the assertion that makes MC_EIG_IZZ load-bearing.
 */
TEST_F(EigenControllerTest, GyroscopicTermIsTheTrueRigidBodyCoupling)
{
	reconfigure("MC_EIG_ALPHA", 0.f);
	reconfigure("MC_EIG_BETA", 0.f);
	reconfigure("MC_EIG_IXX", 0.01f);
	reconfigure("MC_EIG_IYY", 0.02f);
	reconfigure("MC_EIG_IZZ", 0.04f);

	const Vector3f omega(0.5f, -0.3f, 0.7f);
	ControllerState state = hoverState();
	state.angular_velocity = omega;

	// Rate setpoint == measured rate, so every feedback term vanishes.
	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(omega), 0.0025f, output));

	const Vector3f inertia(0.01f, 0.02f, 0.04f);
	const Vector3f expected = omega % inertia.emult(omega);

	EXPECT_NEAR(output.torque(0), expected(0), 1e-7f);
	EXPECT_NEAR(output.torque(1), expected(1), 1e-7f);
	EXPECT_NEAR(output.torque(2), expected(2), 1e-7f);
}

/**
 * Documents the correction rather than the code: for an ordinary planar quadrotor
 * (Ixx == Iyy, Izz ~ 2*Ixx) the true roll coupling is +Ixx*q*r, whereas the prototype's
 * hardcoded A_r gives -Ixx*q*r. Opposite signs - the prototype adds to the coupling it
 * meant to cancel. If someone ever "restores fidelity" by reverting to the hardcoded form,
 * this fails and says why.
 */
TEST_F(EigenControllerTest, GyroscopicTermOpposesThePrototypeForAPlanarQuad)
{
	reconfigure("MC_EIG_ALPHA", 0.f);
	reconfigure("MC_EIG_BETA", 0.f);
	reconfigure("MC_EIG_IXX", 0.01f);
	reconfigure("MC_EIG_IYY", 0.01f);
	reconfigure("MC_EIG_IZZ", 0.02f);

	const Vector3f omega(0.f, -0.3f, 0.7f);
	ControllerState state = hoverState();
	state.angular_velocity = omega;

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(omega), 0.0025f, output));

	const float prototype_roll_term = -0.01f * omega(1) * omega(2);
	EXPECT_NEAR(output.torque(0), -prototype_roll_term, 1e-7f);
	EXPECT_GT(fabsf(prototype_roll_term), 1e-4f);	// the two really do differ
}

/**
 * The assumed aerodynamic damping is cancelled, so it ADDS energy: with no rate error the
 * torque is +alpha*I*omega on roll and pitch.
 */
TEST_F(EigenControllerTest, AssumedDampingIsCancelledNotApplied)
{
	reconfigure("MC_EIG_BETA", 0.f);
	// Symmetric inertia so the gyroscopic term drops out of roll and pitch.
	reconfigure("MC_EIG_IZZ", 0.01f);

	const Vector3f omega(0.4f, 0.f, 0.f);
	ControllerState state = hoverState();
	state.angular_velocity = omega;

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(omega), 0.0025f, output));

	EXPECT_NEAR(output.torque(0), 0.01f * 0.5f * 0.4f, 1e-7f);
	EXPECT_GT(output.torque(0), 0.f);
}

// ---------------------------------------------------------------------------
// Yaw
// ---------------------------------------------------------------------------

/**
 * At Trajectory level the yaw rate setpoint is always zero, and on a symmetric frame the
 * gyroscopic yaw term is zero too, so yaw torque is exactly zero however large the heading
 * error. This is the free-running-heading design point, inherited from MC_CTRL_ALG=1.
 */
TEST_F(EigenControllerTest, TrajectoryLevelProducesNoYawTorque)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(0.f, 0.f, math::radians(150.f)));
	state.heading = math::radians(150.f);

	ControllerCommand command = hoverCommand(state);
	command.yaw_sp = 0.f;	// 150 deg of heading error, deliberately ignored

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(output.torque(2), 0.f, 1e-9f);
}

/**
 * The yaw channel is pure feedforward - cancelling the assumed drag and imposing the same
 * eigenvalue leaves no feedback on the measured yaw rate. So a commanded yaw rate produces
 * Izz*beta*rate_sp, and a measured yaw rate on its own produces nothing.
 */
TEST_F(EigenControllerTest, YawTorqueIsPureFeedforward)
{
	ControllerState state = hoverState();

	ControllerOutput commanded{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(0.f, 0.f, 1.f)), 0.0025f, commanded));
	EXPECT_NEAR(commanded.torque(2), 0.02f * 0.6f * 1.f, 1e-7f);

	// Spinning at 1 rad/s with nothing commanded: no restoring torque at all.
	_controller->reset();
	state.angular_velocity = Vector3f(0.f, 0.f, 1.f);
	ControllerOutput spinning{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(0.f, 0.f, 0.f)), 0.0025f, spinning));
	EXPECT_NEAR(spinning.torque(2), 0.f, 1e-9f);
}

/**
 * Unlike MC_CTRL_ALG=1, the yaw stick does something at the Attitude level: yaw_sp_move_rate
 * reaches the feedforward term.
 */
TEST_F(EigenControllerTest, YawStickReachesTheOutputAtAttitudeLevel)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::Attitude;
	command.attitude_sp = Quatf(1.f, 0.f, 0.f, 0.f);
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);
	command.yaw_sp_move_rate = 0.8f;

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(output.torque(2), 0.02f * 0.6f * 0.8f, 1e-7f);
	EXPECT_NEAR(output.rate_setpoint(2), 0.8f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Frame conversion: NED world, FRD body
// ---------------------------------------------------------------------------

/**
 * A setpoint to the NORTH must pitch the nose DOWN (negative pitch in FRD) and leave roll
 * alone. The prototype's ENU/FLU form would get this wrong, so it is re-derived and pinned
 * here rather than trusted.
 */
TEST_F(EigenControllerTest, NorthPositionErrorPitchesNoseDown)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(10.f, 0.f, 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	const Eulerf attitude_sp(output.attitude_setpoint);
	EXPECT_LT(attitude_sp.theta(), -1e-3f);
	EXPECT_NEAR(attitude_sp.phi(), 0.f, 1e-6f);

	// Magnitude: kp * 10 m of error, inverted through the small-angle hover relation.
	EXPECT_NEAR(attitude_sp.theta(), -(0.5f * 10.f) / CONSTANTS_ONE_G, 1e-4f);
}

/**
 * A setpoint to the EAST must roll RIGHT (positive roll in FRD) and leave pitch alone.
 */
TEST_F(EigenControllerTest, EastPositionErrorRollsRight)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(0.f, 10.f, 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	const Eulerf attitude_sp(output.attitude_setpoint);
	EXPECT_NEAR(attitude_sp.phi(), (0.5f * 10.f) / CONSTANTS_ONE_G, 1e-4f);
	EXPECT_NEAR(attitude_sp.theta(), 0.f, 1e-6f);
}

/**
 * The demand is resolved into the BODY frame, so the same north error at a 90 deg heading
 * becomes a roll to the left rather than a pitch forward. This is the term that would
 * survive the two tests above while still being wrong.
 */
TEST_F(EigenControllerTest, TiltIsResolvedAgainstTheCurrentHeading)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(0.f, 0.f, M_PI_2_F));
	state.heading = M_PI_2_F;

	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(10.f, 0.f, 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	const Eulerf attitude_sp(output.attitude_setpoint);
	// Pointing east, target to the north: that is off the left wing, so roll negative.
	EXPECT_NEAR(attitude_sp.phi(), -(0.5f * 10.f) / CONSTANTS_ONE_G, 1e-4f);
	EXPECT_NEAR(attitude_sp.theta(), 0.f, 1e-4f);
}

/**
 * Damping acts on the estimator velocity, not on a derivative of the position error. At the
 * setpoint but moving north, the vehicle must pitch back to arrest itself.
 */
TEST_F(EigenControllerTest, DampingUsesEstimatorVelocity)
{
	ControllerState state = hoverState();
	state.velocity = Vector3f(2.f, 0.f, 0.f);

	ControllerCommand command = hoverCommand(state);	// position_sp == position

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	const Eulerf attitude_sp(output.attitude_setpoint);
	// kd * (0 - 2) = -2 m/s^2 north, so nose up.
	EXPECT_NEAR(attitude_sp.theta(), (1.0f * 2.f) / CONSTANTS_ONE_G, 1e-4f);
}

// ---------------------------------------------------------------------------
// Equilibrium and envelopes
// ---------------------------------------------------------------------------

TEST_F(EigenControllerTest, HoverEquilibriumCommandsHoverThrustAndNoTorque)
{
	ControllerState state = hoverState();

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, hoverCommand(state), 0.0025f, output));
	ASSERT_TRUE(output.valid);

	EXPECT_NEAR(output.thrust(2), -state.hover_thrust, 1e-6f);
	EXPECT_NEAR(output.thrust(0), 0.f, 1e-9f);
	EXPECT_NEAR(output.thrust(1), 0.f, 1e-9f);

	EXPECT_NEAR(output.torque(0), 0.f, 1e-6f);
	EXPECT_NEAR(output.torque(1), 0.f, 1e-6f);
	EXPECT_NEAR(output.torque(2), 0.f, 1e-9f);
}

/**
 * Two clamps stack here: lateral acceleration is bounded by g*tan(tilt), and the resulting
 * angle is bounded again because the small-angle inversion overshoots (tan(45 deg) rad is
 * 57 deg, not 45). The second clamp is what actually holds the framework's limit.
 */
TEST_F(EigenControllerTest, TiltIsClampedToTheCommandedLimit)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(500.f, 300.f, 0.f);
	command.tilt_limit = math::radians(20.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	EXPECT_LE(tiltOf(output.attitude_setpoint), math::radians(20.f) + 1e-3f);
	EXPECT_GT(tiltOf(output.attitude_setpoint), math::radians(19.f));
}

TEST_F(EigenControllerTest, ThrustIsClampedToTheCommandedEnvelope)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.thrust_max = 0.6f;
	command.position_sp = state.position + Vector3f(0.f, 0.f, -50.f);	// 50 m above

	ControllerOutput climb{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, climb));
	EXPECT_NEAR(climb.thrust(2), -0.6f, 1e-6f);

	// And the floor, on a hard descent demand.
	_controller->reset();
	command.thrust_min = 0.2f;
	command.position_sp = state.position + Vector3f(0.f, 0.f, 50.f);

	ControllerOutput descend{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, descend));
	EXPECT_NEAR(descend.thrust(2), -0.2f, 1e-6f);
}

/**
 * Only the hover term is divided by cos(tilt), matching the prototype's base_thrust. At
 * 30 deg of bank the collective must rise by 1/cos(30 deg).
 */
TEST_F(EigenControllerTest, CollectiveCompensatesBankAngle)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(math::radians(30.f), 0.f, 0.f));

	ControllerCommand command = hoverCommand(state);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(-output.thrust(2), 0.5f / cosf(math::radians(30.f)), 1e-4f);
}

// ---------------------------------------------------------------------------
// Physical scaling
// ---------------------------------------------------------------------------

/**
 * MC_EIG_TRQ_MAX is the only thing standing between physical N m and the dimensionless
 * setpoint control_allocator wants, so halving it must double the normalized output.
 */
TEST_F(EigenControllerTest, TorqueMaxRescalesTheNormalizedOutput)
{
	const ControllerState state = hoverState();
	const ControllerCommand command = rateCommand(Vector3f(0.5f, 0.f, 0.f));

	ControllerOutput at_one{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, at_one));

	reconfigure("MC_EIG_TRQ_MAX", 0.5f);
	_controller->reset();

	ControllerOutput at_half{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, at_half));

	EXPECT_NEAR(at_half.torque(0), 2.f * at_one.torque(0), 1e-6f);
	EXPECT_NEAR(at_half.torque(1), 2.f * at_one.torque(1), 1e-6f);
}

TEST_F(EigenControllerTest, InertiaScalesTorqueLinearly)
{
	const ControllerState state = hoverState();
	const ControllerCommand command = rateCommand(Vector3f(0.5f, 0.f, 0.f));

	ControllerOutput baseline{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, baseline));

	reconfigure("MC_EIG_IXX", 0.02f);
	_controller->reset();

	ControllerOutput doubled{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, doubled));

	EXPECT_NEAR(doubled.torque(0), 2.f * baseline.torque(0), 1e-6f);
	// Pitch is untouched: its scale is Iyy.
	EXPECT_NEAR(doubled.torque(1), baseline.torque(1), 1e-6f);
}

TEST_F(EigenControllerTest, TorqueIsSaturatedToTheNormalizedRange)
{
	reconfigure("MC_EIG_TRQ_MAX", 0.001f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(5.f, -5.f, 0.f)), 0.0025f, output));

	EXPECT_FLOAT_EQ(output.torque(0), 1.f);
	EXPECT_FLOAT_EQ(output.torque(1), -1.f);
}

// ---------------------------------------------------------------------------
// The angle-error derivative
// ---------------------------------------------------------------------------

/**
 * The first cycle after a reset must produce a zero derivative rather than
 * (error - 0) / dt. Verified by showing the first output does not depend on MC_EIG_ATT_D at
 * all, which is only true if the term is seeded.
 */
TEST_F(EigenControllerTest, AngleErrorDerivativeIsSeededOnTheFirstCycle)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::Attitude;
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);
	command.attitude_sp = Quatf(Eulerf(math::radians(30.f), 0.f, 0.f));

	ControllerOutput small_d{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, small_d));

	reconfigure("MC_EIG_ATT_D", 50.f);
	_controller->reset();

	ControllerOutput large_d{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, large_d));

	EXPECT_NEAR(large_d.rate_setpoint(0), small_d.rate_setpoint(0), 1e-6f);
}

/**
 * The derivative is refreshed on a new attitude sample and HELD in between - at gyro rate
 * the attitude usually has not moved, and differentiating an unchanged error against a tiny
 * dt would inject estimator quantization straight into the rate setpoint.
 */
TEST_F(EigenControllerTest, AngleErrorDerivativeIsHeldBetweenAttitudeSamples)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::Attitude;
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);
	command.attitude_sp = Quatf(1.f, 0.f, 0.f, 0.f);

	// Cycle 1 seeds at zero error.
	ControllerOutput seed{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, seed));

	// Cycle 2: the vehicle has rolled, and the sample is fresh - the derivative engages.
	state.q = Quatf(Eulerf(math::radians(2.f), 0.f, 0.f));
	ControllerOutput moved{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, moved));

	const float error_rate = (-math::radians(2.f) - 0.f) / state.freshness.dt_attitude;
	EXPECT_NEAR(moved.rate_setpoint(0), 7.0f * -math::radians(2.f) + 1.0f * error_rate, 1e-4f);

	// Cycle 3: same attitude, stale sample. The held derivative must be reused, not
	// recomputed as zero.
	state.freshness.attitude_new = false;
	ControllerOutput held{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, held));
	EXPECT_NEAR(held.rate_setpoint(0), moved.rate_setpoint(0), 1e-6f);
}

// ---------------------------------------------------------------------------
// Levels, resets, fail-closed
// ---------------------------------------------------------------------------

TEST_F(EigenControllerTest, DeclaresFullStackSupport)
{
	EXPECT_EQ(_controller->supportedLevels(), kAllLevels);
	EXPECT_TRUE(_controller->supportsLevel(ControlLevel::Trajectory));
	EXPECT_TRUE(_controller->supportsLevel(ControlLevel::Attitude));
	EXPECT_TRUE(_controller->supportsLevel(ControlLevel::BodyRate));
	EXPECT_FALSE(_controller->hasOuterStage());
	EXPECT_STREQ(_controller->name(), "eigen");
}

TEST_F(EigenControllerTest, BodyRateLevelPassesTheRateSetpointStraightThrough)
{
	ControllerOutput output{};
	const Vector3f rate_sp(0.3f, -0.2f, 0.1f);
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(rate_sp), 0.0025f, output));

	EXPECT_NEAR(output.rate_setpoint(0), rate_sp(0), 1e-6f);
	EXPECT_NEAR(output.rate_setpoint(1), rate_sp(1), 1e-6f);
	EXPECT_NEAR(output.rate_setpoint(2), rate_sp(2), 1e-6f);
	EXPECT_NEAR(output.thrust(2), -0.5f, 1e-6f);
}

TEST_F(EigenControllerTest, NoneLevelRefusesToProduceOutput)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::None;

	ControllerOutput output{};
	EXPECT_FALSE(_controller->update(state, command, 0.0025f, output));
	EXPECT_FALSE(output.valid);
}

/**
 * Altitude mode and the velocity-only offboard setpoints arrive with position_sp NaN on
 * some axes. Those must contribute nothing rather than poisoning the sum.
 */
TEST_F(EigenControllerTest, PartialSetpointsStillProduceFiniteOutput)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = Vector3f(NAN, NAN, state.position(2) - 5.f);
	command.velocity_sp = Vector3f(1.f, NAN, NAN);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_TRUE(output.torque.isAllFinite());
	EXPECT_TRUE(output.thrust.isAllFinite());
	EXPECT_TRUE(outputIsFinite(output));
}

TEST_F(EigenControllerTest, FullyUnsetSetpointStillProducesFiniteOutput)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = Vector3f(NAN, NAN, NAN);
	command.velocity_sp = Vector3f(NAN, NAN, NAN);
	command.tilt_limit = NAN;

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_TRUE(outputIsFinite(output));
}

/**
 * The hard contract. Nothing integrates here, but the cached position stage and the
 * angle-error history must both be dropped, or a mode change differentiates a stale error
 * into a torque spike on the first cycle back.
 */
TEST_F(EigenControllerTest, ResetIntegralsDropsCachedStageAndErrorHistory)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.level = ControlLevel::Attitude;
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);
	command.attitude_sp = Quatf(1.f, 0.f, 0.f, 0.f);

	ControllerOutput seed{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, seed));

	// A large jump in error, but arriving on the cycle that also demands a reset: the
	// derivative must be re-seeded, so the output matches a pure proportional response.
	state.q = Quatf(Eulerf(math::radians(20.f), 0.f, 0.f));
	command.reset_integrals = true;

	ControllerOutput after_reset{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, after_reset));
	EXPECT_NEAR(after_reset.rate_setpoint(0), 7.0f * -math::radians(20.f), 1e-4f);
}

TEST_F(EigenControllerTest, ResetClearsCachedState)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(20.f, 0.f, 0.f);

	ControllerOutput first{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, first));
	ASSERT_GT(tiltOf(first.attitude_setpoint), 1e-3f);

	_controller->reset();

	// Back at equilibrium, and with the stale stage dropped the controller must not carry
	// the previous tilt across.
	ControllerOutput second{};
	ASSERT_TRUE(_controller->update(state, hoverCommand(state), 0.0025f, second));
	EXPECT_NEAR(tiltOf(second.attitude_setpoint), 0.f, 1e-5f);
	EXPECT_NEAR(second.thrust(2), -state.hover_thrust, 1e-6f);
}

/**
 * The position stage is gated on a fresh sample so it runs at position rate, but it must
 * still run once on entry even if no fresh sample has arrived yet - otherwise the first
 * cycle of a new mode commands whatever was cached.
 */
TEST_F(EigenControllerTest, PositionStageRunsOnEntryWithoutAFreshSample)
{
	ControllerState state = hoverState();
	state.freshness.position_new = false;

	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(10.f, 0.f, 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_LT(Eulerf(output.attitude_setpoint).theta(), -1e-3f);
}

/**
 * A zero inertia is a nonsensical rigid body, not a request to disable an axis. Floored so
 * it cannot silently zero both the axis torque and the other axes' gyroscopic compensation.
 */
TEST_F(EigenControllerTest, ZeroInertiaIsFloored)
{
	reconfigure("MC_EIG_IXX", 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, output));
	EXPECT_TRUE(outputIsFinite(output));
	EXPECT_GT(output.torque(0), 0.f);
}

} // namespace
