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
 * Stage 5b gate: setpoint sourcing.
 *
 * Unlike Stages 1 and 6, there is NO differential oracle available here: the
 * stock logic is embedded in MulticopterPositionControl::Run() rather than an
 * extractable function, so it cannot be compiled as a verbatim twin. These tests
 * therefore assert the documented stock SEMANTICS (failsafe timing boundaries,
 * ramp monotonicity, reset-applied-once, on-ground override, NaN handling).
 * True equivalence only arrives at the Stage 8 SITL A/B.
 *
 * Disposition: PERMANENT.
 */

#include <gtest/gtest.h>

#include "CommandFrontEnd.hpp"

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

vehicle_control_mode_s posctlMode()
{
	vehicle_control_mode_s v{};
	v.flag_armed = true;
	v.flag_control_manual_enabled = true;
	v.flag_control_position_enabled = true;
	v.flag_control_velocity_enabled = true;
	v.flag_control_altitude_enabled = true;
	v.flag_control_climb_rate_enabled = true;
	v.flag_control_attitude_enabled = true;
	v.flag_control_rates_enabled = true;
	v.flag_control_allocation_enabled = true;
	v.flag_multicopter_position_control_enabled = true;
	return v;
}

vehicle_control_mode_s stabMode()
{
	vehicle_control_mode_s v{};
	v.flag_armed = true;
	v.flag_control_manual_enabled = true;
	v.flag_control_attitude_enabled = true;
	v.flag_control_rates_enabled = true;
	v.flag_control_allocation_enabled = true;
	return v;
}

vehicle_control_mode_s acroMode()
{
	vehicle_control_mode_s v{};
	v.flag_armed = true;
	v.flag_control_manual_enabled = true;
	v.flag_control_rates_enabled = true;
	v.flag_control_allocation_enabled = true;
	return v;
}

ControllerState flyingState(uint64_t t_us)
{
	ControllerState s;
	s.timestamp_sample = t_us;
	s.q = Quatf(1.f, 0.f, 0.f, 0.f);
	s.position = Vector3f(0.f, 0.f, -5.f);
	s.velocity = Vector3f(0.f, 0.f, 0.f);
	s.acceleration = Vector3f(0.f, 0.f, 0.f);
	s.angular_velocity = Vector3f(0.1f, 0.2f, 0.3f);
	s.position_valid_xy = s.position_valid_z = true;
	s.velocity_valid_xy = s.velocity_valid_z = true;
	s.heading = 0.f;
	s.unaided_heading = 0.f;
	s.landed = false;
	s.maybe_landed = false;
	s.armed = true;
	s.spooled_up = true;
	s.freshness.dt = 0.004f;
	s.freshness.dt_attitude = 0.004f;
	s.freshness.dt_position = 0.01f;
	s.freshness.position_new = true;
	s.freshness.attitude_new = true;
	return s;
}

/**
 * Drive the front-end through the takeoff state machine until it reports flight.
 *
 * Necessary because the on-ground override (empty setpoint + 100 m/s^2 downward
 * acceleration) is active until TakeoffState::flight is reached - a controller
 * must never see position corrections before the vehicle is airborne. Tests that
 * want pass-through behaviour have to take off first, exactly like the real
 * vehicle.
 *
 * @return the timestamp after takeoff completes
 */
uint64_t takeOff(CommandFrontEnd &fe, uint64_t t, const vehicle_control_mode_s &mode)
{
	vehicle_constraints_s c{};
	c.want_takeoff = true;
	c.speed_up = NAN;
	c.speed_down = NAN;
	fe.setVehicleConstraints(c);
	fe.setControlMode(mode);

	CommandFrontEnd::Publications pubs;

	for (int i = 0; i < 3000; i++) {
		ControllerState s;
		s.timestamp_sample = t;
		s.q = Quatf(1.f, 0.f, 0.f, 0.f);
		s.position = Vector3f(0.f, 0.f, -5.f);
		s.velocity = Vector3f(0.f, 0.f, 0.f);
		s.acceleration = Vector3f(0.f, 0.f, 0.f);
		s.position_valid_xy = s.position_valid_z = true;
		s.velocity_valid_xy = s.velocity_valid_z = true;
		s.landed = (i < 5);
		s.maybe_landed = (i < 5);
		s.armed = true;
		s.spooled_up = true;
		s.freshness.dt_position = 0.01f;
		s.freshness.dt_attitude = 0.004f;
		s.freshness.dt = 0.004f;

		trajectory_setpoint_s sp{};
		sp.timestamp = t;
		// Position hold on all three axes. This used to leave x and y wholly uncommanded,
		// which no flight task emits and which the setpoint-validity check now rejects for
		// the same reason stock's _inputValid() does - half a horizontal vector is not a
		// setpoint - so the helper would have taken off on the failsafe path instead.
		sp.position[0] = sp.position[1] = 0.f;
		sp.position[2] = -5.f;
		sp.velocity[0] = sp.velocity[1] = sp.velocity[2] = NAN;
		sp.acceleration[0] = sp.acceleration[1] = sp.acceleration[2] = NAN;
		sp.jerk[0] = sp.jerk[1] = sp.jerk[2] = NAN;
		sp.yaw = 0.f;
		sp.yawspeed = NAN;
		fe.setTrajectorySetpoint(sp);

		fe.update(s, t, pubs);
		t += 10000;

		if (fe.takeoffState() >= TakeoffState::flight) {
			break;
		}
	}

	return t;
}

trajectory_setpoint_s posSetpoint(uint64_t t_us, float n, float e, float d)
{
	trajectory_setpoint_s sp{};
	sp.timestamp = t_us;
	sp.position[0] = n;
	sp.position[1] = e;
	sp.position[2] = d;
	sp.velocity[0] = sp.velocity[1] = sp.velocity[2] = NAN;
	sp.acceleration[0] = sp.acceleration[1] = sp.acceleration[2] = NAN;
	sp.jerk[0] = sp.jerk[1] = sp.jerk[2] = NAN;
	sp.yaw = 0.f;
	sp.yawspeed = NAN;
	return sp;
}

} // namespace

TEST(CommandFrontEndTest, TrajectorySetpointPassesThroughWithNaNSemantics)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = takeOff(fe, 10000000, posctlMode());
	auto state = flyingState(t);
	fe.setControlMode(posctlMode());
	fe.setTrajectorySetpoint(posSetpoint(t, 3.f, 4.f, -5.f));

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(state, t, pubs);

	ASSERT_GE(fe.takeoffState(), TakeoffState::flight);
	EXPECT_EQ(cmd.level, ControlLevel::Trajectory);
	EXPECT_FLOAT_EQ(cmd.position_sp(0), 3.f);
	EXPECT_FLOAT_EQ(cmd.position_sp(1), 4.f);
	EXPECT_FLOAT_EQ(cmd.position_sp(2), -5.f);

	// Uncommanded axes must stay NaN, not become zero.
	EXPECT_FALSE(PX4_ISFINITE(cmd.velocity_sp(0)));
	EXPECT_FALSE(PX4_ISFINITE(cmd.acceleration_sp(2)));
	EXPECT_FALSE(PX4_ISFINITE(cmd.yawspeed_sp));
	EXPECT_FLOAT_EQ(cmd.yaw_sp, 0.f);
}

TEST(CommandFrontEndTest, StaleSetpointFallsBackThenFailsafe)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = takeOff(fe, 10000000, posctlMode());
	auto state = flyingState(t);
	fe.setControlMode(posctlMode());

	// Establish a valid setpoint.
	fe.setTrajectorySetpoint(posSetpoint(t, 1.f, 1.f, -5.f));
	CommandFrontEnd::Publications pubs;
	fe.update(state, t, pubs);

	// Re-enter position control with a setpoint that predates it: within the
	// 200 ms window the last valid setpoint is reused.
	fe.setControlMode(stabMode());
	fe.update(state, t, pubs);            // leaves Trajectory, re-arms the latch
	fe.setControlMode(posctlMode());

	uint64_t t2 = t + 150000;             // 150 ms later -> inside the window
	state = flyingState(t2);
	fe.setTrajectorySetpoint(posSetpoint(t, 1.f, 1.f, -5.f));  // old timestamp
	const auto &cmd_recent = fe.update(state, t2, pubs);
	EXPECT_FLOAT_EQ(cmd_recent.position_sp(0), 1.f) << "should reuse the last valid setpoint";

	// Beyond the window -> failsafe: velocity is commanded to zero, position is
	// released so the vehicle stops rather than chasing a stale target.
	fe.setControlMode(stabMode());
	fe.update(state, t2, pubs);
	fe.setControlMode(posctlMode());

	uint64_t t3 = t + 400000;             // 400 ms -> outside the window
	state = flyingState(t3);
	fe.setTrajectorySetpoint(posSetpoint(t, 1.f, 1.f, -5.f));
	const auto &cmd_stale = fe.update(state, t3, pubs);

	EXPECT_FALSE(PX4_ISFINITE(cmd_stale.position_sp(0))) << "failsafe must release position";
	EXPECT_FLOAT_EQ(cmd_stale.velocity_sp(0), 0.f) << "failsafe: stop and wait";
	EXPECT_FLOAT_EQ(cmd_stale.velocity_sp(1), 0.f);
}

TEST(CommandFrontEndTest, FailsafeBlindLandsWhenHorizontalVelocityUnavailable)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = takeOff(fe, 10000000, posctlMode());
	CommandFrontEnd::Publications pubs;

	// Force the failsafe path: leave position control (re-arming the "no setpoint
	// since entry" latch), then re-enter with only a stale setpoint and beyond the
	// 200 ms last-valid window.
	auto state = flyingState(t);
	state.velocity(0) = NAN;              // cannot stop horizontally
	state.velocity(1) = NAN;
	fe.setControlMode(stabMode());
	fe.update(state, t, pubs);

	t += 500000;
	state = flyingState(t);
	state.velocity(0) = NAN;
	state.velocity(1) = NAN;
	fe.setControlMode(posctlMode());
	fe.setTrajectorySetpoint(posSetpoint(t - 500000, 1.f, 1.f, -5.f));
	const auto &cmd = fe.update(state, t, pubs);

	// Descend at land speed rather than trying to hold position.
	EXPECT_GT(cmd.velocity_sp(2), 0.f) << "should descend (NED: +z is down)";
	EXPECT_FLOAT_EQ(cmd.acceleration_sp(0), 0.f);
}

TEST(CommandFrontEndTest, FailsafeBlindDescentWhenVerticalVelocityUnavailable)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = takeOff(fe, 10000000, posctlMode());
	CommandFrontEnd::Publications pubs;

	auto state = flyingState(t);
	state.velocity(2) = NAN;
	fe.setControlMode(stabMode());
	fe.update(state, t, pubs);

	t += 500000;
	state = flyingState(t);
	state.velocity(2) = NAN;
	fe.setControlMode(posctlMode());
	fe.setTrajectorySetpoint(posSetpoint(t - 500000, 1.f, 1.f, -5.f));
	const auto &cmd = fe.update(state, t, pubs);

	EXPECT_FALSE(PX4_ISFINITE(cmd.velocity_sp(2)));
	EXPECT_NEAR(cmd.acceleration_sp(2), 0.3f, 1e-6f) << "a bit below hover thrust";
}

/**
 * Stock refuses to fly a setpoint whose estimate has gone away and drops into the failsafe
 * ladder (PositionControl::_inputValid()). Without that check the stage simply found a
 * non-finite position, contributed no position error, and flew on velocity damping - a
 * silent downgrade out of position hold with nothing published to say so.
 */
TEST(CommandFrontEndTest, PositionSetpointWithoutAPositionEstimateFailsafes)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = takeOff(fe, 10000000, posctlMode());
	fe.setControlMode(posctlMode());
	fe.setTrajectorySetpoint(posSetpoint(t, 10.f, 20.f, -5.f));

	CommandFrontEnd::Publications pubs;

	// Sanity: with the estimate present this is an ordinary position hold.
	const auto &held = fe.update(flyingState(t), t, pubs);
	ASSERT_FLOAT_EQ(held.position_sp(0), 10.f);

	// The EKF drops the horizontal solution. Same setpoint, still fresh.
	t += 10000;
	auto blind = flyingState(t);
	blind.position(0) = blind.position(1) = NAN;
	blind.position_valid_xy = false;
	fe.setTrajectorySetpoint(posSetpoint(t, 10.f, 20.f, -5.f));

	const auto &cmd = fe.update(blind, t, pubs);

	EXPECT_FALSE(PX4_ISFINITE(cmd.position_sp(0))) << "must not keep commanding a position it cannot measure";
	// The failsafe answer with a usable velocity estimate: stop and wait.
	EXPECT_FLOAT_EQ(cmd.velocity_sp(0), 0.f);
	EXPECT_FLOAT_EQ(cmd.velocity_sp(1), 0.f);
}

TEST(CommandFrontEndTest, HalfCommandedHorizontalPairIsRejected)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = takeOff(fe, 10000000, posctlMode());
	fe.setControlMode(posctlMode());

	// x commanded, y not. The horizontal setpoint is controlled as a vector, so this is
	// not a setpoint at all - stock's second _inputValid() rule.
	trajectory_setpoint_s sp = posSetpoint(t, 10.f, 20.f, -5.f);
	sp.position[1] = NAN;
	fe.setTrajectorySetpoint(sp);

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(flyingState(t), t, pubs);

	// Rejected whole, not patched up: what gets flown is the last setpoint that passed,
	// which takeOff() left holding the origin. x == 10 would mean the broken one was used.
	EXPECT_FLOAT_EQ(cmd.position_sp(0), 0.f) << "the whole setpoint is rejected, not just the missing half";
	EXPECT_FLOAT_EQ(cmd.position_sp(1), 0.f);

	// Past the 200 ms fallback window the ladder runs out and the failsafe takes over.
	t += 300_ms;
	sp.timestamp = t;
	fe.setTrajectorySetpoint(sp);
	const auto &later = fe.update(flyingState(t), t, pubs);

	EXPECT_FALSE(PX4_ISFINITE(later.position_sp(0)));
	EXPECT_FLOAT_EQ(later.velocity_sp(0), 0.f) << "failsafe: stop and wait";
}

TEST(CommandFrontEndTest, AltitudeModeSetpointWithoutAHorizontalEstimateStaysValid)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = takeOff(fe, 10000000, posctlMode());

	auto altctl = posctlMode();
	altctl.flag_control_position_enabled = false;
	altctl.flag_control_velocity_enabled = false;
	fe.setControlMode(altctl);

	// What FlightTaskManualAltitude publishes: stick tilt on x/y, altitude locked on z, and
	// no horizontal estimate at all - which is the usual reason to be in this mode. Nothing
	// here is commanded in a quantity the estimator cannot supply, so it must fly.
	trajectory_setpoint_s sp{};
	sp.timestamp = t;
	sp.position[0] = sp.position[1] = NAN;
	sp.position[2] = -5.f;
	sp.velocity[0] = sp.velocity[1] = NAN;
	sp.velocity[2] = 0.f;
	sp.acceleration[0] = 1.5f;
	sp.acceleration[1] = -0.5f;
	sp.acceleration[2] = NAN;
	sp.jerk[0] = sp.jerk[1] = sp.jerk[2] = NAN;
	sp.yaw = NAN;
	sp.yawspeed = 0.f;
	fe.setTrajectorySetpoint(sp);

	auto blind = flyingState(t);
	blind.position(0) = blind.position(1) = NAN;
	blind.velocity(0) = blind.velocity(1) = NAN;
	blind.acceleration(0) = blind.acceleration(1) = NAN;
	blind.position_valid_xy = false;
	blind.velocity_valid_xy = false;

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(blind, t, pubs);

	EXPECT_FLOAT_EQ(cmd.acceleration_sp(0), 1.5f) << "the pilot's stick must still reach the controller";
	EXPECT_FLOAT_EQ(cmd.position_sp(2), -5.f);
}

TEST(CommandFrontEndTest, EkfResetShiftsSetpointExactlyOnce)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = takeOff(fe, 10000000, posctlMode());
	fe.setControlMode(posctlMode());
	fe.setTrajectorySetpoint(posSetpoint(t, 10.f, 20.f, -5.f));

	auto state = flyingState(t);
	state.resets.xy = true;
	state.resets.delta_xy = Vector2f(2.f, -3.f);

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(state, t, pubs);
	EXPECT_FLOAT_EQ(cmd.position_sp(0), 12.f);
	EXPECT_FLOAT_EQ(cmd.position_sp(1), 17.f);

	// Next cycle without a reset must not shift again.
	auto state2 = flyingState(t + 10000);
	const auto &cmd2 = fe.update(state2, t + 10000, pubs);
	EXPECT_FLOAT_EQ(cmd2.position_sp(0), 12.f) << "reset applied twice";
	EXPECT_FLOAT_EQ(cmd2.position_sp(1), 17.f);
}

TEST(CommandFrontEndTest, OnGroundOverrideCommandsNoThrustAndResetsIntegrals)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto state = flyingState(t);
	state.landed = true;
	state.armed = false;

	auto mode = posctlMode();
	mode.flag_armed = false;
	fe.setControlMode(mode);
	fe.setTrajectorySetpoint(posSetpoint(t, 5.f, 5.f, -5.f));

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(state, t, pubs);

	// Strong downward acceleration so no thrust is produced, and no position
	// corrections that could wind up integrators before takeoff.
	EXPECT_FLOAT_EQ(cmd.acceleration_sp(2), 100.f);
	EXPECT_FALSE(PX4_ISFINITE(cmd.position_sp(0)));
	EXPECT_TRUE(cmd.reset_integrals);
}

TEST(CommandFrontEndTest, TakeoffStateSurvivesAFlightSpentOutsideTrajectoryLevel)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto mode = stabMode();
	fe.setControlMode(mode);

	manual_control_setpoint_s m{};
	m.throttle = -0.5f;	// climbing away on the stick, as a Stabilized takeoff does
	fe.setManualControlSetpoint(m);

	CommandFrontEnd::Publications pubs;

	// Take off and fly for 3 s without ever entering Trajectory level. The takeoff state
	// machine only ticks inside buildTrajectoryCommand(), so unless the non-trajectory
	// levels drive it too it is still sitting at ::disarmed when the pilot switches.
	for (int i = 0; i < 750; i++) {
		auto state = flyingState(t);
		state.landed = (i < 5);
		state.maybe_landed = (i < 5);
		fe.update(state, t, pubs);
		t += 4000;
	}

	EXPECT_EQ(fe.takeoffState(), TakeoffState::flight)
			<< "an armed, airborne vehicle must not be held pre-takeoff by a mode that "
			   "never runs the trajectory branch";

	// Now switch to Altitude, hovering: altitude locked, no climb commanded, so nothing
	// here would ever assert want_takeoff and release a stuck state machine.
	auto altctl = posctlMode();
	altctl.flag_control_position_enabled = false;
	altctl.flag_control_velocity_enabled = false;
	fe.setControlMode(altctl);

	trajectory_setpoint_s sp{};
	sp.timestamp = t;
	sp.position[0] = sp.position[1] = NAN;
	sp.position[2] = -5.f;
	sp.velocity[0] = sp.velocity[1] = NAN;
	sp.velocity[2] = 0.f;
	sp.acceleration[0] = 1.f;	// stick tilt, and x/y always come as a pair
	sp.acceleration[1] = 0.f;
	sp.acceleration[2] = NAN;
	sp.jerk[0] = sp.jerk[1] = sp.jerk[2] = NAN;
	sp.yaw = NAN;
	sp.yawspeed = 0.f;
	fe.setTrajectorySetpoint(sp);

	const auto &cmd = fe.update(flyingState(t), t, pubs);

	ASSERT_EQ(cmd.level, ControlLevel::Trajectory);
	// The regression: the on-ground override replacing the setpoint with the 100 m/s^2
	// "make no thrust" sentinel, mid-air, on a Stabilized -> Altitude switch.
	EXPECT_FALSE(PX4_ISFINITE(cmd.acceleration_sp(2)));
	EXPECT_FLOAT_EQ(cmd.position_sp(2), -5.f);
	EXPECT_GT(cmd.thrust_min, 0.f) << "thrust_min is only zeroed to allow a takeoff ramp";
}

TEST(CommandFrontEndTest, TakeoffRampVelocityLimitIsMonotonic)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto mode = posctlMode();
	fe.setControlMode(mode);

	vehicle_constraints_s c{};
	c.want_takeoff = true;
	c.speed_up = NAN;
	c.speed_down = NAN;
	fe.setVehicleConstraints(c);

	CommandFrontEnd::Publications pubs;
	float previous = -INFINITY;
	bool ever_increased = false;

	for (int i = 0; i < 400; i++) {
		auto state = flyingState(t);
		state.landed = (i < 50);
		fe.setTrajectorySetpoint(posSetpoint(t, 0.f, 0.f, -5.f));
		const auto &cmd = fe.update(state, t, pubs);

		ASSERT_TRUE(PX4_ISFINITE(cmd.vel_limit_up)) << "iteration " << i;

		if (i > 60) {
			// Once ramping, the limit must never decrease.
			EXPECT_GE(cmd.vel_limit_up, previous - 1e-4f) << "ramp went backwards at " << i;

			if (cmd.vel_limit_up > previous + 1e-4f) {
				ever_increased = true;
			}
		}

		previous = cmd.vel_limit_up;
		t += 10000;
	}

	EXPECT_TRUE(ever_increased) << "takeoff ramp never increased the climb limit";
}

TEST(CommandFrontEndTest, TiltLimitSlewsFromLandingToAirValue)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	float lnd = 12.f, air = 45.f;
	param_set(param_find("MPC_TILTMAX_LND"), &lnd);
	param_set(param_find("MPC_TILTMAX_AIR"), &air);
	h.updateParams();

	uint64_t t = 10000000;
	fe.setControlMode(posctlMode());

	vehicle_constraints_s c{};
	c.want_takeoff = true;
	fe.setVehicleConstraints(c);

	CommandFrontEnd::Publications pubs;
	float first = NAN, last = NAN;

	for (int i = 0; i < 2000; i++) {
		auto state = flyingState(t);
		state.landed = (i < 20);
		fe.setTrajectorySetpoint(posSetpoint(t, 0.f, 0.f, -5.f));
		const auto &cmd = fe.update(state, t, pubs);

		if (i == 25) { first = cmd.tilt_limit; }

		last = cmd.tilt_limit;
		t += 10000;
	}

	ASSERT_TRUE(PX4_ISFINITE(first));
	EXPECT_LE(first, math::radians(air)) << "tilt must start limited near the ground";
	EXPECT_GT(last, first) << "tilt limit should slew upward once flying";
}

TEST(CommandFrontEndTest, StabilizedGeneratesAndPublishesAttitudeSetpoint)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto state = flyingState(t);
	fe.setControlMode(stabMode());

	manual_control_setpoint_s m{};
	m.roll = 0.5f;
	m.pitch = -0.3f;
	m.throttle = 0.f;
	fe.setManualControlSetpoint(m);

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(state, t, pubs);

	EXPECT_EQ(cmd.level, ControlLevel::Attitude);
	EXPECT_TRUE(pubs.attitude_setpoint) << "sticks generated it, so it must be published";
	EXPECT_FALSE(pubs.rates_setpoint);
	EXPECT_TRUE(cmd.attitude_sp.isAllFinite());
	EXPECT_NEAR(cmd.attitude_sp.norm(), 1.f, 1e-5f) << "must be a unit quaternion";
	EXPECT_TRUE(PX4_ISFINITE(cmd.thrust_body_sp(2)));
}

TEST(CommandFrontEndTest, OffboardAttitudeIsConsumedNotRepublished)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto state = flyingState(t);

	vehicle_control_mode_s mode{};
	mode.flag_armed = true;
	mode.flag_control_offboard_enabled = true;
	mode.flag_control_attitude_enabled = true;
	mode.flag_control_rates_enabled = true;
	fe.setControlMode(mode);

	vehicle_attitude_setpoint_s sp{};
	sp.timestamp = t;
	Quatf(0.9238795f, 0.f, 0.f, 0.3826834f).copyTo(sp.q_d);   // 45 deg yaw
	sp.thrust_body[2] = -0.6f;
	fe.setExternalAttitudeSetpoint(sp);

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(state, t, pubs);

	EXPECT_EQ(cmd.level, ControlLevel::Attitude);
	EXPECT_FALSE(pubs.attitude_setpoint) << "republishing what we consumed is a self-feedback path";
	EXPECT_NEAR(cmd.attitude_sp(0), 0.9238795f, 1e-5f);
	EXPECT_FLOAT_EQ(cmd.thrust_body_sp(2), -0.6f);
}

TEST(CommandFrontEndTest, AcroGeneratesRateSetpointFromSticks)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto state = flyingState(t);
	fe.setControlMode(acroMode());

	manual_control_setpoint_s m{};
	m.roll = 1.f;
	m.throttle = 0.f;
	fe.setManualControlSetpoint(m);

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(state, t, pubs);

	EXPECT_EQ(cmd.level, ControlLevel::BodyRate);
	EXPECT_TRUE(pubs.rates_setpoint);
	EXPECT_GT(cmd.rate_sp(0), 0.f) << "full roll stick should command a positive roll rate";
	EXPECT_TRUE(PX4_ISFINITE(cmd.thrust_body_sp(2)));
}

TEST(CommandFrontEndTest, OffboardBodyRateNaNAxesFallBackToMeasuredRate)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto state = flyingState(t);   // measured rates 0.1 / 0.2 / 0.3

	vehicle_control_mode_s mode{};
	mode.flag_armed = true;
	mode.flag_control_offboard_enabled = true;
	mode.flag_control_rates_enabled = true;
	fe.setControlMode(mode);

	vehicle_rates_setpoint_s sp{};
	sp.timestamp = t;
	sp.roll = 0.5f;
	sp.pitch = NAN;      // uncommanded
	sp.yaw = NAN;
	sp.thrust_body[2] = -0.5f;
	fe.setExternalRatesSetpoint(sp);

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(state, t, pubs);

	EXPECT_FLOAT_EQ(cmd.rate_sp(0), 0.5f);
	EXPECT_FLOAT_EQ(cmd.rate_sp(1), 0.2f) << "NaN pitch must fall back to the measured rate";
	EXPECT_FLOAT_EQ(cmd.rate_sp(2), 0.3f);

	// The framework guarantees rate_sp is fully finite at BodyRate level.
	EXPECT_TRUE(cmd.rate_sp.isAllFinite());
}

TEST(CommandFrontEndTest, ResetIntegralsOnLevelChangeDisarmAndGround)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto state = flyingState(t);
	CommandFrontEnd::Publications pubs;

	fe.setControlMode(stabMode());
	EXPECT_TRUE(fe.update(state, t, pubs).reset_integrals) << "first entry is a level change";

	// Same level, armed, flying -> no reset.
	EXPECT_FALSE(fe.update(state, t + 4000, pubs).reset_integrals);

	// Level change -> reset.
	fe.setControlMode(acroMode());
	EXPECT_TRUE(fe.update(state, t + 8000, pubs).reset_integrals);

	// Disarmed -> reset.
	auto disarmed = acroMode();
	disarmed.flag_armed = false;
	fe.setControlMode(disarmed);
	EXPECT_TRUE(fe.update(state, t + 12000, pubs).reset_integrals);

	// Landed -> reset.
	fe.setControlMode(acroMode());
	auto landed = flyingState(t + 16000);
	landed.landed = true;
	EXPECT_TRUE(fe.update(landed, t + 16000, pubs).reset_integrals);
}

TEST(CommandFrontEndTest, TerminationProducesNoCommand)
{
	param_control_autosave(false);
	ParamHarness h;
	CommandFrontEnd fe{&h};
	h.updateParams();
	fe.reset();

	uint64_t t = 10000000;
	auto state = flyingState(t);

	vehicle_control_mode_s mode{};
	mode.flag_armed = true;
	mode.flag_control_termination_enabled = true;
	fe.setControlMode(mode);

	CommandFrontEnd::Publications pubs;
	const auto &cmd = fe.update(state, t, pubs);

	EXPECT_EQ(cmd.level, ControlLevel::None);
	EXPECT_FALSE(pubs.attitude_setpoint);
	EXPECT_FALSE(pubs.rates_setpoint);
}

/**
 * The manual throttle curve must track the live hover-thrust estimate.
 *
 * Regression test for a real defect: CommandFrontEnd never forwarded the estimate
 * to StickToAttitudeSetpoint, so Stabilized flew on the MPC_THR_HOVER parameter
 * while stock used the estimator's live value (wired at
 * mc_att_control_main.cpp:119-124). The gap was invisible to every unit test and
 * showed up in flight only as a ~1.9% thrust-z difference against stock - the one
 * signal that survived a statistical A/B whose other six channels passed.
 *
 * Mid-stick is the discriminating input: the throttle curve interpolates through
 * hover thrust there, so changing the estimate must change the collective. At the
 * stick extremes the curve is pinned to MPC_THR_MIN/MAX and the estimate cannot
 * be observed.
 */
TEST(CommandFrontEndTest, ManualThrottleCurveFollowsHoverThrustEstimate)
{
	param_control_autosave(false);

	// Mid-stick, so the collective sits on the interpolated part of the curve where
	// hover thrust is the interpolation midpoint. At the stick extremes the curve is
	// pinned to MPC_THR_MIN/MPC_THR_MAX and the estimate cannot be observed at all -
	// which is part of why no pre-existing test caught this defect.
	manual_control_setpoint_s m{};
	m.throttle = 0.f;

	// A FRESH front end per case, not a reset() one. The manual throttle min/max slew
	// rates evolve over the run, so reusing one instance leaks spool-up state from the
	// first case into the second, and the test then "detects" a difference that has
	// nothing to do with hover thrust. An earlier version of this test did exactly
	// that and passed even with the hover-thrust path deleted.
	//
	// Duration matters too: the estimate reaches the curve through a 0.05/s slew
	// (StickToAttitudeSetpoint.cpp), so 8 s is needed to traverse 0.25 of thrust. A
	// short run cannot distinguish the two cases regardless of wiring.
	auto collectiveFor = [&](float hover_thrust) {
		ParamHarness h;
		CommandFrontEnd fe{&h};
		h.updateParams();
		fe.reset();
		fe.setControlMode(stabMode());
		fe.setManualControlSetpoint(m);

		// Delivered on the state, exactly as the module delivers it - there is no
		// setter to call. That is the point: a value pushed in by the module made this
		// defect unreachable from any unit test.
		uint64_t t = 10000000;
		float thrust = NAN;

		for (int i = 0; i < 2000; i++) {            // 2000 * 4 ms = 8 s
			auto state = flyingState(t);
			state.hover_thrust = hover_thrust;
			state.hover_thrust_valid = true;
			CommandFrontEnd::Publications pubs;
			const auto &cmd = fe.update(state, t, pubs);
			thrust = cmd.thrust_body_sp(2);
			t += 4000;
		}

		return thrust;
	};

	const float low = collectiveFor(0.35f);
	const float high = collectiveFor(0.65f);

	ASSERT_TRUE(PX4_ISFINITE(low));
	ASSERT_TRUE(PX4_ISFINITE(high));

	// Thrust is negative-up in body FRD, so a higher hover thrust is more negative.
	EXPECT_LT(high, low)
			<< "a higher hover-thrust estimate must raise the manual collective; "
			<< "identical values mean the estimate is not reaching the throttle curve. "
			<< "low(0.35)=" << low << " high(0.65)=" << high;
	EXPECT_GT(low - high, 0.15f)
			<< "the estimate reaches the curve but barely moves it. "
			<< "low(0.35)=" << low << " high(0.65)=" << high;
}
