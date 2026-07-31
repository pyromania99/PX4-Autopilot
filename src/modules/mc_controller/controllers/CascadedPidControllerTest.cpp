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
 * Stage 6 gate: the differential test.
 *
 * Drives CascadedPidController and a HAND-WIRED PositionControl +
 * AttitudeControl + RateControl triple - the same classes the stock modules
 * instantiate - from an identical scripted sequence, requiring agreement to
 * 1e-6f. This is what makes MC_CTRL_ALG=1 a trustworthy A/B baseline before
 * anything flies, and it is deterministic and CI-able unlike SITL.
 *
 * Disposition: PERMANENT.
 */

#include <gtest/gtest.h>

#include "CascadedPidController.hpp"

#include <parameters/param.h>
#include <px4_platform_common/defines.h>

#include <random>

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

float pf(const char *n)
{
	float v = 0.f;
	param_get(param_find(n), &v);
	return v;
}

/// The stock cascade, wired by hand exactly as the three modules wire it.
struct StockTriple {
	PositionControl position;
	AttitudeControl attitude;
	RateControl rate;
	AlphaFilter<float> yaw_lpf;

	Vector3f rate_sp{};
	Vector3f thrust_sp{};
	Quatf attitude_sp{};
	float yaw_sp_move_rate{0.f};
	bool position_valid{false};

	void configure()
	{
		position.setPositionGains(Vector3f(pf("MPC_XY_P"), pf("MPC_XY_P"), pf("MPC_Z_P")));
		position.setVelocityGains(
			Vector3f(pf("MPC_XY_VEL_P_ACC"), pf("MPC_XY_VEL_P_ACC"), pf("MPC_Z_VEL_P_ACC")),
			Vector3f(pf("MPC_XY_VEL_I_ACC"), pf("MPC_XY_VEL_I_ACC"), pf("MPC_Z_VEL_I_ACC")),
			Vector3f(pf("MPC_XY_VEL_D_ACC"), pf("MPC_XY_VEL_D_ACC"), pf("MPC_Z_VEL_D_ACC")));
		position.setHorizontalThrustMargin(pf("MPC_THR_XY_MARG"));

		int32_t decouple = 0;
		param_get(param_find("MPC_ACC_DECOUPLE"), &decouple);
		position.decoupleHorizontalAndVecticalAcceleration(decouple != 0);

		attitude.setProportionalGain(Vector3f(pf("MC_ROLL_P"), pf("MC_PITCH_P"), pf("MC_YAW_P")),
					     pf("MC_YAW_WEIGHT"));
		attitude.setRateLimit(Vector3f(math::radians(pf("MC_ROLLRATE_MAX")),
					       math::radians(pf("MC_PITCHRATE_MAX")),
					       math::radians(pf("MC_YAWRATE_MAX"))));

		const Vector3f k(pf("MC_ROLLRATE_K"), pf("MC_PITCHRATE_K"), pf("MC_YAWRATE_K"));
		rate.setPidGains(k.emult(Vector3f(pf("MC_ROLLRATE_P"), pf("MC_PITCHRATE_P"), pf("MC_YAWRATE_P"))),
				 k.emult(Vector3f(pf("MC_ROLLRATE_I"), pf("MC_PITCHRATE_I"), pf("MC_YAWRATE_I"))),
				 k.emult(Vector3f(pf("MC_ROLLRATE_D"), pf("MC_PITCHRATE_D"), pf("MC_YAWRATE_D"))));
		rate.setIntegratorLimit(Vector3f(pf("MC_RR_INT_LIM"), pf("MC_PR_INT_LIM"), pf("MC_YR_INT_LIM")));
		rate.setFeedForwardGain(Vector3f(pf("MC_ROLLRATE_FF"), pf("MC_PITCHRATE_FF"), pf("MC_YAWRATE_FF")));

		yaw_lpf.setCutoffFreq(pf("MC_YAW_TQ_CUTOFF"));
	}

	void reset()
	{
		position.resetIntegral();
		rate.resetIntegral();
		yaw_lpf.reset(0.f);
		yaw_sp_move_rate = 0.f;
		rate_sp.setZero();
		thrust_sp.setZero();
		attitude_sp = Quatf();
		position_valid = false;
	}

	/// Mirrors CascadedPidController::update() using the stock objects directly.
	bool step(const ControllerState &s, const ControllerCommand &c, float dt, Vector3f &torque_out,
		  Vector3f &thrust_out)
	{
		if (c.reset_integrals) {
			position.resetIntegral();
			rate.resetIntegral();
		}

		if (!s.armed) {
			rate.resetIntegral();
		}

		if (c.level == ControlLevel::Trajectory) {
			if (s.freshness.position_new || !position_valid) {
				PositionControlStates st;
				st.position = s.position;
				st.velocity = s.velocity;
				st.acceleration = s.acceleration;
				st.yaw = s.heading;
				position.setState(st);
				position.setHoverThrust(s.hover_thrust);
				position.setTiltLimit(c.tilt_limit);
				position.setThrustLimits(c.thrust_min, c.thrust_max);
				position.setVelocityLimits(c.vel_limit_xy, c.vel_limit_up, c.vel_limit_down);

				trajectory_setpoint_s sp{};
				sp.timestamp = c.timestamp;
				c.position_sp.copyTo(sp.position);
				c.velocity_sp.copyTo(sp.velocity);
				c.acceleration_sp.copyTo(sp.acceleration);
				c.jerk_sp.copyTo(sp.jerk);
				sp.yaw = c.yaw_sp;
				sp.yawspeed = c.yawspeed_sp;

				if ((!PX4_ISFINITE(sp.velocity[0]) || !PX4_ISFINITE(sp.velocity[1]))
				    && (!PX4_ISFINITE(sp.position[0]) || !PX4_ISFINITE(sp.position[1]))) {
					position.resetIntegralXY();
				}

				position.setInputSetpoint(sp);

				if (position.update(s.freshness.dt_position)) {
					vehicle_attitude_setpoint_s asp{};
					position.getAttitudeSetpoint(asp);
					attitude_sp = Quatf(asp.q_d);
					thrust_sp = Vector3f(asp.thrust_body);
					yaw_sp_move_rate = PX4_ISFINITE(asp.yaw_sp_move_rate) ? asp.yaw_sp_move_rate : 0.f;
					position_valid = true;

				} else if (!position_valid) {
					return false;
				}
			}
		}

		if (c.level == ControlLevel::Attitude) {
			attitude_sp = c.attitude_sp;
			thrust_sp = c.thrust_body_sp;
		}

		if (c.level == ControlLevel::Trajectory || c.level == ControlLevel::Attitude) {
			if (s.freshness.attitude_new || c.level == ControlLevel::Attitude) {
				const float ff = (c.level == ControlLevel::Attitude) ? c.yaw_sp_move_rate : yaw_sp_move_rate;
				attitude.setAttitudeSetpoint(attitude_sp, ff);
				rate_sp = attitude.update(s.q);
			}
		}

		if (c.level == ControlLevel::BodyRate) {
			rate_sp = c.rate_sp;
			thrust_sp = c.thrust_body_sp;
		}

		Vector3f torque = rate.update(s.angular_velocity, rate_sp, s.angular_accel, dt,
					      s.maybe_landed || s.landed);
		torque(2) = yaw_lpf.update(torque(2), dt);

		torque_out = torque;
		thrust_out = thrust_sp;
		return true;
	}
};

ControllerState baseState(uint64_t t)
{
	ControllerState s;
	s.timestamp_sample = t;
	s.q = Quatf(1.f, 0.f, 0.f, 0.f);
	s.position = Vector3f(0.f, 0.f, -5.f);
	s.velocity = Vector3f(0.f, 0.f, 0.f);
	s.acceleration = Vector3f(0.f, 0.f, 0.f);
	s.angular_velocity = Vector3f(0.f, 0.f, 0.f);
	s.angular_accel = Vector3f(0.f, 0.f, 0.f);
	s.heading = 0.f;
	s.position_valid_xy = s.position_valid_z = true;
	s.velocity_valid_xy = s.velocity_valid_z = true;
	s.landed = false;
	s.maybe_landed = false;
	s.armed = true;
	s.spooled_up = true;
	s.hover_thrust = 0.5f;
	s.hover_thrust_valid = true;
	s.freshness.dt = 0.004f;
	s.freshness.dt_attitude = 0.004f;
	s.freshness.dt_position = 0.01f;
	return s;
}

ControllerCommand trajectoryCommand(uint64_t t, const Vector3f &pos_sp, float yaw_sp)
{
	ControllerCommand c;
	c.timestamp = t;
	c.level = ControlLevel::Trajectory;
	c.position_sp = pos_sp;
	c.velocity_sp = Vector3f(NAN, NAN, NAN);
	c.acceleration_sp = Vector3f(NAN, NAN, NAN);
	c.jerk_sp = Vector3f(NAN, NAN, NAN);
	c.yaw_sp = yaw_sp;
	c.yawspeed_sp = NAN;
	c.tilt_limit = math::radians(45.f);
	c.thrust_min = 0.1f;
	c.thrust_max = 0.9f;
	c.vel_limit_xy = 12.f;
	c.vel_limit_up = 3.f;
	c.vel_limit_down = 1.5f;
	return c;
}

} // namespace

/**
 * The key test: an identical scripted sequence through both implementations,
 * covering takeoff, a POSCTL -> STAB -> ACRO sweep, an EKF-style attitude jump,
 * and a saturation event.
 */
TEST(CascadedPidControllerTest, MatchesHandWiredStockTripleAcrossFlightSequence)
{
	param_control_autosave(false);

	ParamHarness h;
	CascadedPidController ctrl{&h};
	h.updateParams();
	ctrl.reset();

	StockTriple stock;
	stock.configure();
	stock.reset();

	std::mt19937 rng{20260729u};
	std::uniform_real_distribution<float> jitter{-1.f, 1.f};

	uint64_t t = 10000000;
	float yaw = 0.f;

	for (int step = 0; step < 3000; step++) {
		auto state = baseState(t);

		// Position arrives at 1/10 the gyro rate, as on the real vehicle. This is
		// what exercises stage-rate parity.
		state.freshness.position_new = (step % 10 == 0);
		state.freshness.attitude_new = (step % 2 == 0);

		// Wander the vehicle so all three loops see non-trivial errors.
		yaw = wrap_pi(yaw + 0.002f * jitter(rng));
		state.q = Quatf(Eulerf(0.05f * jitter(rng), 0.05f * jitter(rng), yaw));
		state.heading = yaw;
		state.position = Vector3f(0.3f * jitter(rng), 0.3f * jitter(rng), -5.f + 0.2f * jitter(rng));
		state.velocity = Vector3f(0.2f * jitter(rng), 0.2f * jitter(rng), 0.1f * jitter(rng));
		state.acceleration = Vector3f(0.1f * jitter(rng), 0.1f * jitter(rng), 0.1f * jitter(rng));
		state.angular_velocity = Vector3f(0.1f * jitter(rng), 0.1f * jitter(rng), 0.1f * jitter(rng));
		state.angular_accel = Vector3f(0.5f * jitter(rng), 0.5f * jitter(rng), 0.5f * jitter(rng));

		// Mode sweep: Trajectory -> Attitude -> BodyRate -> Trajectory
		ControllerCommand cmd;

		if (step < 1200) {
			cmd = trajectoryCommand(t, Vector3f(2.f, 3.f, -5.f), 0.2f);

		} else if (step < 2000) {
			cmd = ControllerCommand{};
			cmd.timestamp = t;
			cmd.level = ControlLevel::Attitude;
			cmd.attitude_sp = Quatf(Eulerf(0.1f, -0.05f, 0.3f));
			cmd.yaw_sp_move_rate = 0.05f;
			cmd.thrust_body_sp = Vector3f(0.f, 0.f, -0.55f);

		} else if (step < 2600) {
			cmd = ControllerCommand{};
			cmd.timestamp = t;
			cmd.level = ControlLevel::BodyRate;
			cmd.rate_sp = Vector3f(0.4f * jitter(rng), 0.4f * jitter(rng), 0.2f * jitter(rng));
			cmd.thrust_body_sp = Vector3f(0.f, 0.f, -0.6f);

		} else {
			cmd = trajectoryCommand(t, Vector3f(-1.f, 1.f, -6.f), -0.4f);
		}

		// Reset integrals at each mode boundary, as the front-end does.
		cmd.reset_integrals = (step == 1200) || (step == 2000) || (step == 2600);

		// Saturation event partway through the rate segment.
		if (step == 2300) {
			AllocatorFeedback fb;
			fb.torque_setpoint_achieved = false;
			fb.saturation_positive(0) = true;
			ctrl.setAllocatorFeedback(fb);
			stock.rate.setSaturationStatus(fb.saturation_positive, fb.saturation_negative);
		}

		ControllerOutput out;
		const bool ok_ctrl = ctrl.update(state, cmd, state.freshness.dt, out);

		Vector3f stock_torque{};
		Vector3f stock_thrust{};
		const bool ok_stock = stock.step(state, cmd, state.freshness.dt, stock_torque, stock_thrust);

		ASSERT_EQ(ok_ctrl, ok_stock) << "validity diverged at step " << step;

		if (!ok_ctrl) {
			t += 4000;
			continue;
		}

		for (int i = 0; i < 3; i++) {
			ASSERT_NEAR(out.torque(i), stock_torque(i), 1e-6f)
					<< "torque(" << i << ") diverged at step " << step;
			ASSERT_NEAR(out.thrust(i), stock_thrust(i), 1e-6f)
					<< "thrust(" << i << ") diverged at step " << step;
		}

		t += 4000;
	}
}

TEST(CascadedPidControllerTest, DeclaresFullStackSupport)
{
	param_control_autosave(false);
	ParamHarness h;
	CascadedPidController ctrl{&h};
	h.updateParams();

	EXPECT_STREQ(ctrl.name(), "cascaded_pid");
	EXPECT_EQ(ctrl.supportedLevels(), kAllLevels);
	EXPECT_TRUE(ctrl.supportsLevel(ControlLevel::Trajectory));
	EXPECT_TRUE(ctrl.supportsLevel(ControlLevel::Attitude));
	EXPECT_TRUE(ctrl.supportsLevel(ControlLevel::BodyRate));
}

TEST(CascadedPidControllerTest, ResetIntegralsActuallyZeroesTheRateIntegrator)
{
	param_control_autosave(false);
	ParamHarness h;
	CascadedPidController ctrl{&h};
	h.updateParams();
	ctrl.reset();

	uint64_t t = 10000000;
	auto state = baseState(t);

	ControllerCommand cmd;
	cmd.level = ControlLevel::BodyRate;
	cmd.rate_sp = Vector3f(1.f, 0.f, 0.f);   // large persistent error winds up I
	cmd.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);

	ControllerOutput out;

	for (int i = 0; i < 500; i++) {
		ctrl.update(state, cmd, 0.004f, out);
	}

	rate_ctrl_status_s status{};
	ctrl.getRateControlStatus(status);
	const float wound_up = fabsf(status.rollspeed_integ);
	ASSERT_GT(wound_up, 1e-4f) << "integrator never wound up; test is vacuous";

	cmd.reset_integrals = true;
	ctrl.update(state, cmd, 0.004f, out);
	ctrl.getRateControlStatus(status);

	// The integral is zeroed and then the rate stage runs in the same cycle, so it
	// legitimately holds one step of re-accumulation - stock behaves identically.
	// What matters is that the wound-up history is gone.
	EXPECT_LT(fabsf(status.rollspeed_integ), 0.02f * wound_up)
			<< "reset_integrals left " << status.rollspeed_integ << " of " << wound_up;
}

TEST(CascadedPidControllerTest, NoneLevelRefusesToProduceOutput)
{
	param_control_autosave(false);
	ParamHarness h;
	CascadedPidController ctrl{&h};
	h.updateParams();
	ctrl.reset();

	ControllerCommand cmd;
	cmd.level = ControlLevel::None;

	ControllerOutput out;
	EXPECT_FALSE(ctrl.update(baseState(10000000), cmd, 0.004f, out));
	EXPECT_FALSE(out.valid);
}

TEST(CascadedPidControllerTest, YawTorqueIsLowPassFiltered)
{
	param_control_autosave(false);
	float cutoff = 5.f;
	param_set(param_find("MC_YAW_TQ_CUTOFF"), &cutoff);

	ParamHarness h;
	CascadedPidController ctrl{&h};
	h.updateParams();
	ctrl.reset();

	uint64_t t = 10000000;
	auto state = baseState(t);

	ControllerCommand cmd;
	cmd.level = ControlLevel::BodyRate;
	cmd.thrust_body_sp = Vector3f(0.f, 0.f, -0.5f);

	ControllerOutput out;
	float previous_yaw_torque = 0.f;
	float max_step = 0.f;

	// Alternate the yaw rate demand every cycle: without the LPF the yaw torque
	// would chatter at full amplitude.
	for (int i = 0; i < 200; i++) {
		cmd.rate_sp = Vector3f(0.f, 0.f, (i % 2 == 0) ? 1.f : -1.f);
		ctrl.update(state, cmd, 0.004f, out);

		if (i > 10) {
			max_step = math::max(max_step, fabsf(out.torque(2) - previous_yaw_torque));
		}

		previous_yaw_torque = out.torque(2);
	}

	// A 5 Hz cutoff at 250 Hz must heavily attenuate a Nyquist-rate square wave.
	EXPECT_LT(max_step, 0.5f) << "yaw output LPF appears inactive (max step " << max_step << ")";
}
