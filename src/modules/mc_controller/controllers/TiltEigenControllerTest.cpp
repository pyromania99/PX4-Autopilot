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
 * @file TiltEigenControllerTest.cpp
 *
 * Properties the tilt-axis eigen law is supposed to have. Four groups carry the weight,
 * and the first two are what justify this controller existing alongside MC_CTRL_ALG=3:
 *
 *  - THE CHART CHANGE IS FAITHFUL. The tilt error reduces to the Euler roll/pitch error to
 *    first order, with the same sign and magnitude, which is the claim that lets
 *    MC_EIG_ATT_P's tuned value transfer to MC_TEIG_ATT_P unchanged.
 *  - THE CHART CHANGE BUYS SOMETHING. Finite and correctly directed output at 90 deg of
 *    pitch and when inverted, where MC_CTRL_ALG=3 is singular; an exact tilt limit rather
 *    than a linearization that overshoots; and no dt dependence anywhere, which is what
 *    removing the angle-error derivative was for.
 *  - THE CROSS-COUPLING SURVIVED. Stage 3 was supposed to be untouched by any of this;
 *    these are the same assertions EigenControllerTest makes, and they must still hold.
 *  - THE FRAME CONVERSION, because a sign error there is a controller that accelerates
 *    away from its setpoint.
 */

#include <gtest/gtest.h>

#include "TiltEigenController.hpp"

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
	setParam("MC_TEIG_XY_P", 0.50f);
	setParam("MC_TEIG_XY_D", 1.00f);
	setParam("MC_TEIG_Z_P", 10.0f);
	setParam("MC_TEIG_Z_D", 5.3f);
	setParam("MC_TEIG_ATT_P", 7.0f);
	setParam("MC_TEIG_WN", 18.0f);
	setParam("MC_TEIG_B", 10.2f);
	setParam("MC_TEIG_ALPHA", 0.5f);
	setParam("MC_TEIG_BETA", 0.6f);
	setParam("MC_TEIG_IXX", 0.01f);
	setParam("MC_TEIG_IYY", 0.01f);
	setParam("MC_TEIG_IZZ", 0.02f);
	setParam("MC_TEIG_TRQ_MAX", 1.0f);
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

/// Stabilized-style command carrying an attitude setpoint.
ControllerCommand attitudeCommand(const Quatf &attitude_sp, const float yaw_move_rate = 0.f)
{
	ControllerCommand command{};
	command.level = ControlLevel::Attitude;
	command.attitude_sp = attitude_sp;
	command.yaw_sp_move_rate = yaw_move_rate;
	command.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);
	command.thrust_min = 0.f;
	command.thrust_max = 1.f;
	command.manual = true;
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

class TiltEigenControllerTest : public ::testing::Test
{
public:
	void SetUp() override
	{
		// Without this the first param write blocks forever: autosave schedules onto
		// wq:lp_default, which the gtest harness does not start.
		param_control_autosave(false);
		resetParams();
		_harness = new ParamHarness();
		_controller = new TiltEigenController(_harness);
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
	TiltEigenController *_controller{nullptr};
};

// ---------------------------------------------------------------------------
// The chart change is faithful: tilt error IS the Euler error, to first order
// ---------------------------------------------------------------------------

/**
 * THE GAIN-TRANSFER CLAIM. MC_TEIG_ATT_P ships with MC_EIG_ATT_P's tuned value on the
 * grounds that the two act on the same quantity for small errors. That is only true if the
 * tilt error vector reduces to (roll_error, pitch_error) with the SAME SIGN and the SAME
 * MAGNITUDE - a sign flip here would be an unstable attitude loop, and a factor of two
 * would be a controller tuned twice as hot as the log says.
 *
 * Driven at Attitude level so the setpoint is exact and no position PD intervenes: the
 * vehicle is level, the setpoint is a small pure roll (then a small pure pitch), so the
 * rate setpoint must come out as att_p * that angle on that axis and ~0 on the other.
 */
TEST_F(TiltEigenControllerTest, TiltErrorMatchesEulerErrorAtSmallAngles)
{
	const ControllerState state = hoverState();	// level
	const float angle = math::radians(3.f);
	const float att_p = 7.0f;

	ControllerOutput roll_output{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(Eulerf(angle, 0.f, 0.f))), 0.0025f, roll_output));

	// Same sign, same magnitude as att_p * roll_error would give in the Euler law.
	EXPECT_NEAR(roll_output.rate_setpoint(0), att_p * angle, 1e-4f);
	EXPECT_NEAR(roll_output.rate_setpoint(1), 0.f, 1e-5f);

	_controller->reset();

	ControllerOutput pitch_output{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(Eulerf(0.f, angle, 0.f))), 0.0025f, pitch_output));

	EXPECT_NEAR(pitch_output.rate_setpoint(1), att_p * angle, 1e-4f);
	EXPECT_NEAR(pitch_output.rate_setpoint(0), 0.f, 1e-5f);
}

/**
 * The same identity from the other side: with the vehicle ROTATED and the setpoint level,
 * the error must point the other way. Guards against an error that is computed against the
 * wrong operand, which the setpoint-only test above cannot see because at a level state the
 * two differ only in sign.
 */
TEST_F(TiltEigenControllerTest, TiltErrorReversesWhenTheVehicleIsTheOneRolled)
{
	ControllerState state = hoverState();
	const float angle = math::radians(3.f);
	state.q = Quatf(Eulerf(angle, 0.f, 0.f));	// vehicle rolled right, setpoint level

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(1.f, 0.f, 0.f, 0.f)), 0.0025f, output));

	// Error is -angle, so the commanded rate rolls back to the left.
	EXPECT_NEAR(output.rate_setpoint(0), -7.0f * angle, 1e-4f);
	EXPECT_NEAR(output.rate_setpoint(1), 0.f, 1e-5f);
}

/**
 * The tilt error is a rotation ANGLE, so a 90 deg error must read as pi/2 - not as the
 * sine of it, and not as whatever an Euler decomposition would report. This is the test
 * that would fail if the axis-angle extraction used asin() (correct only on half the
 * sphere) or skipped the theta/sin(theta) rescale (correct only near zero).
 */
TEST_F(TiltEigenControllerTest, TiltErrorIsAnAngleNotItsSine)
{
	const ControllerState state = hoverState();	// level

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(Eulerf(math::radians(90.f), 0.f, 0.f))),
					0.0025f, output));

	// att_p * pi/2, not att_p * sin(pi/2) == att_p.
	EXPECT_NEAR(output.rate_setpoint(0), 7.0f * M_PI_F / 2.f, 1e-3f);
}

// ---------------------------------------------------------------------------
// The chart change buys something: no singularity, exact limit, no dt
// ---------------------------------------------------------------------------

/**
 * THE HEADLINE. At 90 deg of pitch the Euler decomposition MC_CTRL_ALG=3 relies on is
 * singular - roll and yaw are no longer separable and phi() is meaningless. The reduced
 * attitude error has no such point: the vehicle's thrust axis is horizontal, the setpoint's
 * is vertical, and the rotation between them is a perfectly ordinary 90 deg about the pitch
 * axis.
 *
 * Assert it is finite, of the right magnitude, and pointing the right way: nose-up 90 deg
 * needs a nose-DOWN rate to recover.
 */
TEST_F(TiltEigenControllerTest, NoSingularityAtNinetyDegreesOfPitch)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(0.f, math::radians(90.f), 0.f));	// straight up

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(1.f, 0.f, 0.f, 0.f)), 0.0025f, output));

	ASSERT_TRUE(output.rate_setpoint.isAllFinite());
	ASSERT_TRUE(output.torque.isAllFinite());

	// Error is a 90 deg rotation about -pitch, so the commanded pitch rate is -att_p*pi/2.
	EXPECT_NEAR(output.rate_setpoint(1), -7.0f * M_PI_F / 2.f, 1e-3f);
	EXPECT_NEAR(output.rate_setpoint(0), 0.f, 1e-4f);
}

/**
 * Fully inverted is the degenerate case of the axis-angle extraction: every axis in the
 * body x-y plane closes the error equally well, so the axis is genuinely undefined and a
 * naive implementation divides by a zero sine.
 *
 * The requirement is not a particular axis - it is that the vehicle does not sit there
 * commanding nothing. Assert finite output and a non-trivial recovery demand.
 */
TEST_F(TiltEigenControllerTest, InvertedAttitudeStillCommandsARecovery)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(M_PI_F, 0.f, 0.f));	// upside down

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(1.f, 0.f, 0.f, 0.f)), 0.0025f, output));

	ASSERT_TRUE(output.rate_setpoint.isAllFinite());
	ASSERT_TRUE(output.torque.isAllFinite());

	// A real rotation is demanded, not a shrug. pi radians of error at att_p = 7 is a
	// large rate demand, which is the correct response to being inverted.
	EXPECT_GT(Vector2f(output.rate_setpoint(0), output.rate_setpoint(1)).norm(), 7.0f * 3.f);
}

/**
 * NO dt DEPENDENCE ANYWHERE. Removing the angle-error derivative was supposed to make the
 * law a pure function of state and command; this pins that, because a derivative creeping
 * back in is exactly the regression that would reintroduce the body-frame differencing
 * problem the design avoids.
 *
 * Two identical cycles with wildly different timesteps, and with the attitude sample marked
 * stale on the second, must produce identical output.
 */
TEST_F(TiltEigenControllerTest, OutputIsIndependentOfTheTimestep)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(math::radians(5.f), math::radians(-4.f), 0.f));

	ControllerOutput fast{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(1.f, 0.f, 0.f, 0.f)), 0.001f, fast));

	_controller->reset();

	state.freshness.attitude_new = false;
	state.freshness.dt_attitude = 0.02f;

	ControllerOutput slow{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(1.f, 0.f, 0.f, 0.f)), 0.02f, slow));

	EXPECT_NEAR(fast.rate_setpoint(0), slow.rate_setpoint(0), 1e-9f);
	EXPECT_NEAR(fast.rate_setpoint(1), slow.rate_setpoint(1), 1e-9f);
	EXPECT_NEAR(fast.torque(0), slow.torque(0), 1e-9f);
	EXPECT_NEAR(fast.torque(1), slow.torque(1), 1e-9f);
}

/**
 * THE REASON THE D TERM WENT. The tilt error lives in the body frame, and that frame spins
 * with the vehicle. A derivative taken by differencing it would pick up an omega_z x e term
 * proportional to yaw rate - invisible at hover, dominant under the rotor-failure spin this
 * law exists for.
 *
 * The P-only error is a function of ATTITUDE alone, so spinning the vehicle at 8 rad/s must
 * not move the attitude-derived part of the rate setpoint at all. (The torque does change,
 * correctly, through the gyroscopic term - that is stage 3's business and asserted
 * separately.)
 */
TEST_F(TiltEigenControllerTest, TiltErrorIsUnaffectedByYawRate)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(math::radians(6.f), math::radians(3.f), 0.f));

	ControllerOutput still{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(1.f, 0.f, 0.f, 0.f)), 0.0025f, still));

	_controller->reset();

	state.angular_velocity = Vector3f(0.f, 0.f, 8.f);	// hard yaw spin

	ControllerOutput spinning{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(1.f, 0.f, 0.f, 0.f)), 0.0025f, spinning));

	// The rate SETPOINT is attitude-derived and must be untouched by the spin.
	EXPECT_NEAR(still.rate_setpoint(0), spinning.rate_setpoint(0), 1e-6f);
	EXPECT_NEAR(still.rate_setpoint(1), spinning.rate_setpoint(1), 1e-6f);
}

/**
 * THE EXACT TILT LIMIT. MC_CTRL_ALG=3 bounds lateral acceleration at g*tan(limit) and then
 * has to clamp a second time, because inverting the linearized hover relation asks for
 * tan(limit) RADIANS - 57 deg at a 45 deg limit. Here the direction is bounded by
 * limitTilt(), so a demand far beyond the envelope lands exactly ON the limit.
 */
TEST_F(TiltEigenControllerTest, TiltIsClampedExactlyToTheCommandedLimit)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.tilt_limit = math::radians(30.f);
	command.position_sp = state.position + Vector3f(500.f, 0.f, 0.f);	// far beyond the envelope

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	// Exactly the limit, not the limit plus a linearization error.
	EXPECT_NEAR(tiltOf(output.attitude_setpoint), math::radians(30.f), 1e-3f);
}

/**
 * The limit is honoured across the range, not just at one angle - and in particular a 45 deg
 * limit produces 45 deg, which is the case where MC_CTRL_ALG=3's linearization overshoots
 * most visibly.
 */
TEST_F(TiltEigenControllerTest, TiltLimitHoldsAcrossTheRange)
{
	for (const float limit_deg : {15.f, 30.f, 45.f, 60.f}) {
		_controller->reset();

		ControllerState state = hoverState();
		ControllerCommand command = hoverCommand(state);
		command.tilt_limit = math::radians(limit_deg);
		command.position_sp = state.position + Vector3f(500.f, 300.f, 0.f);

		ControllerOutput output{};
		ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

		EXPECT_NEAR(tiltOf(output.attitude_setpoint), math::radians(limit_deg), 1e-3f)
				<< "limit " << limit_deg << " deg";
	}
}

// ---------------------------------------------------------------------------
// The cross-coupling survived: stage 3 was supposed to be untouched
// ---------------------------------------------------------------------------

/**
 * A pure ROLL rate error must produce a PITCH torque, and vice versa with the opposite
 * sign. Same assertion and same expected numbers as EigenControllerTest, because the change
 * of attitude chart is not supposed to have reached this stage at all - if these numbers
 * moved, something in stage 3 was edited that should not have been.
 */
TEST_F(TiltEigenControllerTest, RateErrorIsCrossCoupledAntisymmetrically)
{
	ControllerState state = hoverState();

	ControllerOutput roll_output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, roll_output));

	// Own axis: wn * error, scaled by inertia.
	EXPECT_NEAR(roll_output.torque(0), 0.01f * 18.0f, 1e-5f);
	// Cross axis: -b * roll error. Non-zero is the point; the sign is the law.
	EXPECT_NEAR(roll_output.torque(1), 0.01f * -10.2f, 1e-5f);
	EXPECT_NEAR(roll_output.torque(2), 0.f, 1e-9f);

	_controller->reset();
	ControllerOutput pitch_output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(0.f, 1.f, 0.f)), 0.0025f, pitch_output));

	EXPECT_NEAR(pitch_output.torque(1), 0.01f * 18.0f, 1e-5f);
	EXPECT_NEAR(pitch_output.torque(0), 0.01f * 10.2f, 1e-5f);

	EXPECT_NEAR(roll_output.torque(1), -pitch_output.torque(0), 1e-6f);
}

/// MC_TEIG_B == 0 collapses the law to two independent axes - the documented escape hatch.
TEST_F(TiltEigenControllerTest, ZeroCrossGainDecouplesTheAxes)
{
	reconfigure("MC_TEIG_B", 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, output));

	EXPECT_NEAR(output.torque(0), 0.01f * 18.0f, 1e-5f);
	EXPECT_NEAR(output.torque(1), 0.f, 1e-9f);
}

/**
 * The gyroscopic term must be the TRUE omega x (I*omega), not the prototype's hardcoded
 * approximation - which for a planar quadrotor has the opposite sign on roll and adds to
 * the coupling it was meant to cancel. Pinned against a direct evaluation.
 */
TEST_F(TiltEigenControllerTest, GyroscopicTermIsTheTrueRigidBodyCoupling)
{
	// Asymmetric inertias so a sign or index error cannot hide behind Ixx == Iyy.
	reconfigure("MC_TEIG_IXX", 0.011f);
	reconfigure("MC_TEIG_IYY", 0.017f);
	reconfigure("MC_TEIG_IZZ", 0.023f);
	// Isolate the gyroscopic term: no rate error, no assumed damping to cancel.
	reconfigure("MC_TEIG_ALPHA", 0.f);
	reconfigure("MC_TEIG_BETA", 0.f);

	const Vector3f omega(0.7f, -0.4f, 1.3f);
	ControllerState state = hoverState();
	state.angular_velocity = omega;

	// rate_sp == omega, so every feedback term is exactly zero.
	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(omega), 0.0025f, output));

	const Vector3f inertia(0.011f, 0.017f, 0.023f);
	const Vector3f expected = omega % inertia.emult(omega);

	EXPECT_NEAR(output.torque(0), expected(0), 1e-7f);
	EXPECT_NEAR(output.torque(1), expected(1), 1e-7f);
	EXPECT_NEAR(output.torque(2), expected(2), 1e-7f);
}

/**
 * MC_TEIG_ALPHA is the damping the airframe is ASSUMED to have, which the feedback
 * linearization cancels - so it enters as +alpha*rate and adds energy. Getting this sign
 * backwards would look like extra damping and quietly destabilize the real vehicle.
 */
TEST_F(TiltEigenControllerTest, AssumedDampingIsCancelledNotApplied)
{
	reconfigure("MC_TEIG_B", 0.f);		// isolate the own-axis path

	ControllerState state = hoverState();
	state.angular_velocity = Vector3f(0.5f, 0.f, 0.f);

	// Command the rate the vehicle already has: the wn term is zero, so only alpha is left.
	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(0.5f, 0.f, 0.f)), 0.0025f, output));

	// +alpha * p * Ixx, positive: it ADDS to the motion.
	EXPECT_NEAR(output.torque(0), 0.01f * 0.5f * 0.5f, 1e-6f);
	EXPECT_GT(output.torque(0), 0.f);
}

/// Yaw carries no feedback term, so a zero yaw rate setpoint means zero yaw torque.
TEST_F(TiltEigenControllerTest, TrajectoryLevelProducesNoYawTorque)
{
	ControllerState state = hoverState();
	// Yawing hard, with a symmetric airframe so the gyroscopic term vanishes too.
	state.angular_velocity = Vector3f(0.f, 0.f, 2.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, hoverCommand(state), 0.0025f, output));

	EXPECT_NEAR(output.torque(2), 0.f, 1e-9f);
	EXPECT_NEAR(output.rate_setpoint(2), 0.f, 1e-9f);
}

/// Yaw torque is pure feedforward: proportional to the SETPOINT, blind to the measurement.
TEST_F(TiltEigenControllerTest, YawTorqueIsPureFeedforward)
{
	ControllerState state = hoverState();

	ControllerOutput commanded{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(0.f, 0.f, 1.f)), 0.0025f, commanded));
	EXPECT_NEAR(commanded.torque(2), 0.02f * 0.6f, 1e-6f);

	// Same command, but now the vehicle is already yawing: the output must not move.
	_controller->reset();
	state.angular_velocity = Vector3f(0.f, 0.f, 1.f);

	ControllerOutput matched{};
	ASSERT_TRUE(_controller->update(state, rateCommand(Vector3f(0.f, 0.f, 1.f)), 0.0025f, matched));
	EXPECT_NEAR(matched.torque(2), commanded.torque(2), 1e-6f);
}

/// The yaw stick still reaches the output at Attitude level, unlike MC_CTRL_ALG=1.
TEST_F(TiltEigenControllerTest, YawStickReachesTheOutputAtAttitudeLevel)
{
	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(hoverState(), attitudeCommand(Quatf(1.f, 0.f, 0.f, 0.f), 1.5f),
					0.0025f, output));

	EXPECT_NEAR(output.rate_setpoint(2), 1.5f, 1e-6f);
	EXPECT_NEAR(output.torque(2), 0.02f * 0.6f * 1.5f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Frame conversion: NED/FRD signs
// ---------------------------------------------------------------------------

/// A north position error must pitch the nose DOWN. A sign error here flies away.
TEST_F(TiltEigenControllerTest, NorthPositionErrorPitchesNoseDown)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(5.f, 0.f, 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	// Nose down is negative pitch, so the desired body-z leans south (negative x in NED).
	const Vector3f body_z_sp = Dcmf(output.attitude_setpoint).col(2);
	EXPECT_LT(body_z_sp(0), -0.01f);
	EXPECT_NEAR(body_z_sp(1), 0.f, 1e-4f);

	// And the pitch rate demand follows it down.
	EXPECT_LT(output.rate_setpoint(1), -0.01f);
	EXPECT_NEAR(output.rate_setpoint(0), 0.f, 1e-4f);
}

/// An east position error must roll RIGHT.
TEST_F(TiltEigenControllerTest, EastPositionErrorRollsRight)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(0.f, 5.f, 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	const Vector3f body_z_sp = Dcmf(output.attitude_setpoint).col(2);
	EXPECT_LT(body_z_sp(1), -0.01f);
	EXPECT_NEAR(body_z_sp(0), 0.f, 1e-4f);

	EXPECT_GT(output.rate_setpoint(0), 0.01f);
	EXPECT_NEAR(output.rate_setpoint(1), 0.f, 1e-4f);
}

/**
 * The tilt is resolved against the vehicle's CURRENT heading. Nose east, wanting to go
 * north, means going left - which is a roll, not a pitch. Here that falls out of R^T rather
 * than out of an explicit sin/cos of the heading.
 */
TEST_F(TiltEigenControllerTest, TiltIsResolvedAgainstTheCurrentHeading)
{
	ControllerState state = hoverState();
	state.heading = math::radians(90.f);
	state.q = Quatf(Eulerf(0.f, 0.f, state.heading));

	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(5.f, 0.f, 0.f);	// north

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	// North is to the vehicle's left: roll left, no pitch.
	EXPECT_LT(output.rate_setpoint(0), -0.01f);
	EXPECT_NEAR(output.rate_setpoint(1), 0.f, 1e-3f);
}

/// Damping acts on the estimator velocity, not on a derivative of the position error.
TEST_F(TiltEigenControllerTest, DampingUsesEstimatorVelocity)
{
	ControllerState state = hoverState();
	state.velocity = Vector3f(2.f, 0.f, 0.f);	// drifting north, on setpoint

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, hoverCommand(state), 0.0025f, output));

	// Position error is zero, so any response at all is the D term: pitch up to brake.
	EXPECT_GT(output.rate_setpoint(1), 0.01f);
}

// ---------------------------------------------------------------------------
// Equilibrium, saturation, scaling
// ---------------------------------------------------------------------------

/// At equilibrium: hover collective, level setpoint, no torque.
TEST_F(TiltEigenControllerTest, HoverEquilibriumCommandsHoverThrustAndNoTorque)
{
	ControllerState state = hoverState();

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, hoverCommand(state), 0.0025f, output));

	EXPECT_NEAR(output.thrust(2), -0.5f, 1e-4f);
	EXPECT_NEAR(output.torque(0), 0.f, 1e-6f);
	EXPECT_NEAR(output.torque(1), 0.f, 1e-6f);
	EXPECT_NEAR(output.torque(2), 0.f, 1e-9f);
	EXPECT_NEAR(tiltOf(output.attitude_setpoint), 0.f, 1e-4f);
}

/// Banked, the collective grows as 1/cos(tilt) to keep the vertical component.
TEST_F(TiltEigenControllerTest, CollectiveCompensatesBankAngle)
{
	ControllerState state = hoverState();
	state.q = Quatf(Eulerf(math::radians(30.f), 0.f, 0.f));

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, hoverCommand(state), 0.0025f, output));

	EXPECT_NEAR(output.thrust(2), -0.5f / cosf(math::radians(30.f)), 1e-3f);
}

/// The collective respects the commanded envelope.
TEST_F(TiltEigenControllerTest, ThrustIsClampedToTheCommandedEnvelope)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.thrust_max = 0.6f;
	command.position_sp = state.position + Vector3f(0.f, 0.f, -50.f);	// climb hard

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));
	EXPECT_NEAR(output.thrust(2), -0.6f, 1e-6f);

	_controller->reset();
	command.thrust_min = 0.3f;
	command.position_sp = state.position + Vector3f(0.f, 0.f, 50.f);	// descend hard

	ControllerOutput descend{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, descend));
	EXPECT_NEAR(descend.thrust(2), -0.3f, 1e-6f);
}

/// MC_TEIG_TRQ_MAX rescales the normalized output without touching the physical torque.
TEST_F(TiltEigenControllerTest, TorqueMaxRescalesTheNormalizedOutput)
{
	ControllerOutput unit{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, unit));

	reconfigure("MC_TEIG_TRQ_MAX", 2.f);
	_controller->reset();

	ControllerOutput halved{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, halved));

	EXPECT_NEAR(halved.torque(0), unit.torque(0) / 2.f, 1e-6f);
	EXPECT_NEAR(halved.torque(1), unit.torque(1) / 2.f, 1e-6f);
}

/// Inertia scales torque linearly - it is a physical property, not a bare gain.
TEST_F(TiltEigenControllerTest, InertiaScalesTorqueLinearly)
{
	ControllerOutput baseline{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, baseline));

	reconfigure("MC_TEIG_IXX", 0.02f);
	_controller->reset();

	ControllerOutput doubled{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, doubled));

	EXPECT_NEAR(doubled.torque(0), baseline.torque(0) * 2.f, 1e-6f);
}

/// The normalized output is saturated, never handed to control_allocator out of range.
TEST_F(TiltEigenControllerTest, TorqueIsSaturatedToTheNormalizedRange)
{
	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(500.f, 0.f, 0.f)), 0.0025f, output));

	EXPECT_LE(output.torque(0), 1.f);
	EXPECT_GE(output.torque(0), -1.f);
	EXPECT_NEAR(output.torque(0), 1.f, 1e-6f);
	EXPECT_NEAR(output.torque(1), -1.f, 1e-6f);
}

/// A zero inertia is a nonsensical rigid body, not a "disable this axis" request.
TEST_F(TiltEigenControllerTest, ZeroInertiaIsFloored)
{
	reconfigure("MC_TEIG_IXX", 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(Vector3f(1.f, 0.f, 0.f)), 0.0025f, output));

	ASSERT_TRUE(output.torque.isAllFinite());
	EXPECT_GT(output.torque(0), 0.f);
}

// ---------------------------------------------------------------------------
// Framework contract
// ---------------------------------------------------------------------------

TEST_F(TiltEigenControllerTest, DeclaresFullStackSupport)
{
	EXPECT_EQ(_controller->supportedLevels(), kAllLevels);
	EXPECT_STREQ(_controller->name(), "tilt_eigen");
	EXPECT_FALSE(_controller->hasOuterStage());
}

/// Acro: the commanded rates reach the inner loop untouched by any attitude stage.
TEST_F(TiltEigenControllerTest, BodyRateLevelPassesTheRateSetpointStraightThrough)
{
	const Vector3f rate_sp(0.3f, -0.2f, 0.1f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(hoverState(), rateCommand(rate_sp), 0.0025f, output));

	EXPECT_NEAR(output.rate_setpoint(0), rate_sp(0), 1e-6f);
	EXPECT_NEAR(output.rate_setpoint(1), rate_sp(1), 1e-6f);
	EXPECT_NEAR(output.rate_setpoint(2), rate_sp(2), 1e-6f);
}

TEST_F(TiltEigenControllerTest, NoneLevelRefusesToProduceOutput)
{
	ControllerCommand command{};
	command.level = ControlLevel::None;

	ControllerOutput output{};
	EXPECT_FALSE(_controller->update(hoverState(), command, 0.0025f, output));
	EXPECT_FALSE(output.valid);
}

/// Altitude-style: velocity setpoint only, position NaN. Must still produce finite output.
TEST_F(TiltEigenControllerTest, PartialSetpointsStillProduceFiniteOutput)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = Vector3f(NAN, NAN, state.position(2));
	command.velocity_sp = Vector3f(1.f, 0.f, NAN);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	EXPECT_TRUE(output.torque.isAllFinite());
	EXPECT_TRUE(output.thrust.isAllFinite());
	EXPECT_TRUE(output.rate_setpoint.isAllFinite());
}

/// Everything NaN. NaN downstream of control_allocator means "stop that motor".
TEST_F(TiltEigenControllerTest, FullyUnsetSetpointStillProducesFiniteOutput)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = Vector3f(NAN, NAN, NAN);
	command.velocity_sp = Vector3f(NAN, NAN, NAN);
	command.acceleration_sp = Vector3f(NAN, NAN, NAN);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	EXPECT_TRUE(output.torque.isAllFinite());
	EXPECT_TRUE(output.thrust.isAllFinite());
}

/**
 * reset_integrals is the framework's one hard contract. There are no integrators here, but
 * the cached thrust direction must be dropped so a mode change cannot carry a stale tilt
 * across - and the position stage must re-run on the next cycle even without a fresh
 * position sample.
 */
TEST_F(TiltEigenControllerTest, ResetIntegralsDropsTheCachedStage)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(5.f, 0.f, 0.f);

	ControllerOutput first{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, first));
	ASSERT_GT(tiltOf(first.attitude_setpoint), 1e-3f);	// there really is a lean to drop

	// Now on setpoint, stale position sample, and a reset: the cached lean must go.
	state.freshness.position_new = false;
	command.position_sp = state.position;
	command.reset_integrals = true;

	ControllerOutput second{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, second));

	EXPECT_NEAR(tiltOf(second.attitude_setpoint), 0.f, 1e-4f);
}

/// reset() drops every cached stage output.
TEST_F(TiltEigenControllerTest, ResetClearsCachedState)
{
	ControllerState state = hoverState();
	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(5.f, 0.f, 0.f);

	ControllerOutput first{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, first));

	_controller->reset();

	// Stale position sample: without the reset having cleared _position_stage_valid, the
	// stage would be skipped and the old lean would survive.
	state.freshness.position_new = false;
	command.position_sp = state.position;

	ControllerOutput second{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, second));

	EXPECT_NEAR(tiltOf(second.attitude_setpoint), 0.f, 1e-4f);
}

/// Entering Trajectory level without a fresh position sample must still run the stage once.
TEST_F(TiltEigenControllerTest, PositionStageRunsOnEntryWithoutAFreshSample)
{
	ControllerState state = hoverState();
	state.freshness.position_new = false;

	ControllerCommand command = hoverCommand(state);
	command.position_sp = state.position + Vector3f(5.f, 0.f, 0.f);

	ControllerOutput output{};
	ASSERT_TRUE(_controller->update(state, command, 0.0025f, output));

	// The stage ran: a north error produced a lean.
	EXPECT_GT(tiltOf(output.attitude_setpoint), 1e-3f);
}

/**
 * The attitude-level path takes only the DIRECTION out of the pilot's quaternion, so two
 * setpoints with the same tilt and different headings must produce the same rate demand.
 * This is what "no yaw control" means here, and unlike MC_CTRL_ALG=3 it holds without a
 * heading having been injected and cancelled.
 */
TEST_F(TiltEigenControllerTest, AttitudeLevelIgnoresTheCommandedHeading)
{
	const ControllerState state = hoverState();
	const float roll = math::radians(10.f);

	ControllerOutput north{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(Eulerf(roll, 0.f, 0.f))), 0.0025f, north));

	_controller->reset();

	// Same tilt magnitude, wildly different commanded heading.
	ControllerOutput turned{};
	ASSERT_TRUE(_controller->update(state, attitudeCommand(Quatf(Eulerf(roll, 0.f, math::radians(140.f)))),
					0.0025f, turned));

	// The tilt DIRECTION differs because the heading rotates it, so compare magnitude:
	// what must hold is that no yaw rate is commanded in either case.
	EXPECT_NEAR(north.rate_setpoint(2), 0.f, 1e-9f);
	EXPECT_NEAR(turned.rate_setpoint(2), 0.f, 1e-9f);

	const float north_norm = Vector2f(north.rate_setpoint(0), north.rate_setpoint(1)).norm();
	const float turned_norm = Vector2f(turned.rate_setpoint(0), turned.rate_setpoint(1)).norm();
	EXPECT_NEAR(north_norm, turned_norm, 1e-4f);
}

} // namespace
