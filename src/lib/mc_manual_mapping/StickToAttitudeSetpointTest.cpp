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
 * `ReferenceStickToAttitude` below is a VERBATIM copy of the pre-extraction
 * implementation from MulticopterAttitudeControl (mc_att_control_main.cpp
 * generate_attitude_setpoint() + throttle_curve(), and the slew-rate/reset
 * handling from Run()), taken at git 154dc85809. The extraction is proven
 * correct by driving both implementations with an identical randomized input
 * sequence and requiring bit-level agreement.
 *
 * Disposition: PERMANENT. Keep as a regression guard so future edits to
 * StickToAttitudeSetpoint cannot silently change stock flight behaviour.
 */

#include <gtest/gtest.h>

#include "StickToAttitudeSetpoint.hpp"

#include <AttitudeControlMath.hpp>
#include <lib/mathlib/math/filter/AlphaFilter.hpp>
#include <lib/slew_rate/SlewRate.hpp>
#include <lib/stick_yaw/StickYaw.hpp>
#include <mathlib/math/Functions.hpp>
#include <mathlib/math/Limits.hpp>
#include <parameters/param.h>
#include <px4_platform_common/defines.h>

#include <random>

using namespace matrix;

/* ------------------------------------------------------------------------ */
/* Verbatim pre-extraction reference implementation                          */
/* ------------------------------------------------------------------------ */

class ReferenceStickToAttitude : public ModuleParams
{
public:
	explicit ReferenceStickToAttitude(ModuleParams *parent) : ModuleParams(parent)
	{
		_manual_throttle_minimum.setSlewRate(0.05f);
		_manual_throttle_maximum.setSlewRate(0.5f);
		_hover_thrust_slew_rate.setSlewRate(0.05f);
		parameters_updated();
	}

	// mc_att_control_main.cpp:91-109 (stick-related portion only)
	void parameters_updated()
	{
		if (!PX4_ISFINITE(_hover_thrust_estimate)) {
			_hover_thrust_slew_rate.setForcedValue(_param_mpc_thr_hover.get());
		}

		_man_tilt_max = math::radians(_param_mpc_man_tilt_max.get());
	}

	// mc_att_control_main.cpp:229-241
	void setHoverThrustEstimate(float hover_thrust)
	{
		if (PX4_ISFINITE(hover_thrust)) {
			_hover_thrust_estimate = math::constrain(hover_thrust, .05f, .9f);

		} else {
			_hover_thrust_estimate = _param_mpc_thr_hover.get();
		}
	}

	// mc_att_control_main.cpp:376-392
	void updateSlewRates(bool landed, bool spooled_up, float dt)
	{
		if (landed) {
			_manual_throttle_minimum.update(0.f, dt);

		} else {
			_manual_throttle_minimum.update(_param_mpc_manthr_min.get(), dt);
		}

		if (spooled_up) {
			_manual_throttle_maximum.update(1.f, dt);

		} else {
			_manual_throttle_maximum.setForcedValue(0.f);
		}

		if (PX4_ISFINITE(_hover_thrust_estimate)) {
			_hover_thrust_slew_rate.update(_hover_thrust_estimate, dt);
		}
	}

	// mc_att_control_main.cpp:329-334
	void ekfResetHandler(float delta_psi)
	{
		if (PX4_ISFINITE(_yaw_setpoint_stabilized)) {
			_yaw_setpoint_stabilized = wrap_pi(_yaw_setpoint_stabilized + delta_psi);
		}

		_stick_yaw.ekfResetHandler(delta_psi);
	}

	// mc_att_control_main.cpp:305-308 / 370-373
	void reset(const Quatf &q, float unaided_heading)
	{
		_man_roll_input_filter.reset(0.f);
		_man_pitch_input_filter.reset(0.f);
		_yaw_setpoint_stabilized = NAN;
		_stick_yaw.reset(Eulerf(q).psi(), unaided_heading);
	}

	// mc_att_control_main.cpp:111-137
	float throttle_curve(float throttle_stick_input)
	{
		float thrust = 0.f;

		switch (_param_mpc_thr_curve.get()) {
		case 1:
			thrust = math::interpolate(throttle_stick_input, -1.f, 1.f,
						   _manual_throttle_minimum.getState(), _param_mpc_thr_max.get());
			break;

		case 2:
			thrust = math::interpolateNXY(throttle_stick_input,
			{-1.f, 0.f, 1.f},
			{_manual_throttle_minimum.getState(), _param_mpc_thr_hover.get(), _param_mpc_thr_max.get()});
			break;

		default:
			thrust = math::interpolateNXY(throttle_stick_input,
			{-1.f, 0.f, 1.f},
			{_manual_throttle_minimum.getState(), _hover_thrust_slew_rate.getState(), _param_mpc_thr_max.get()});
			break;
		}

		return math::min(thrust, _manual_throttle_maximum.getState());
	}

	// mc_att_control_main.cpp:139-205
	void generate_attitude_setpoint(const manual_control_setpoint_s &_manual_control_setpoint, const Quatf &q,
					float _unaided_heading, float dt, vehicle_attitude_setpoint_s &attitude_setpoint)
	{
		const bool arming_gesture = (_manual_control_setpoint.throttle < -.9f) && (_param_mc_airmode.get() != 2);

		if (arming_gesture) {
			_yaw_setpoint_stabilized = NAN;
		}

		const float yaw = Eulerf(q).psi();
		const float yaw_stick_input = math::expo_deadzone(_manual_control_setpoint.yaw, .6f, _param_man_deadzone.get());
		_stick_yaw.generateYawSetpoint(attitude_setpoint.yaw_sp_move_rate, _yaw_setpoint_stabilized, yaw_stick_input, yaw, dt,
					       _unaided_heading);

		_man_roll_input_filter.setParameters(dt, _param_mc_man_tilt_tau.get());
		_man_pitch_input_filter.setParameters(dt, _param_mc_man_tilt_tau.get());

		Vector2f v = Vector2f(_man_roll_input_filter.update(_manual_control_setpoint.roll * _man_tilt_max),
				      -_man_pitch_input_filter.update(_manual_control_setpoint.pitch * _man_tilt_max));
		float v_norm = v.norm();

		if (v_norm > _man_tilt_max) {
			v *= _man_tilt_max / v_norm;
		}

		Quatf q_sp_rp = AxisAnglef(v(0), v(1), 0.f);
		const float yaw_setpoint = PX4_ISFINITE(_yaw_setpoint_stabilized) ? _yaw_setpoint_stabilized : yaw;
		const Quatf q_sp_yaw(cosf(yaw_setpoint / 2.f), 0.f, 0.f, sinf(yaw_setpoint / 2.f));

		if (_vtol) {
			AttitudeControlMath::correctTiltSetpointForYawError(q_sp_rp, q, q_sp_yaw);
		}

		Quatf q_sp = q_sp_yaw * q_sp_rp;

		q_sp.copyTo(attitude_setpoint.q_d);

		attitude_setpoint.thrust_body[2] = -throttle_curve(_manual_control_setpoint.throttle);
	}

	void setVtol(bool vtol) { _vtol = vtol; }

private:
	StickYaw _stick_yaw{this};

	float _hover_thrust_estimate{NAN};
	SlewRate<float> _hover_thrust_slew_rate{.5f};

	float _yaw_setpoint_stabilized{0.f};
	float _man_tilt_max{0.f};

	SlewRate<float> _manual_throttle_minimum{0.f};
	SlewRate<float> _manual_throttle_maximum{0.f};
	AlphaFilter<float> _man_roll_input_filter;
	AlphaFilter<float> _man_pitch_input_filter;

	bool _vtol{false};

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::MC_AIRMODE>)         _param_mc_airmode,
		(ParamFloat<px4::params::MC_MAN_TILT_TAU>)  _param_mc_man_tilt_tau,
		(ParamFloat<px4::params::MAN_DEADZONE>)     _param_man_deadzone,
		(ParamFloat<px4::params::MPC_MAN_TILT_MAX>) _param_mpc_man_tilt_max,
		(ParamFloat<px4::params::MPC_MANTHR_MIN>)   _param_mpc_manthr_min,
		(ParamFloat<px4::params::MPC_THR_MAX>)      _param_mpc_thr_max,
		(ParamFloat<px4::params::MPC_THR_HOVER>)    _param_mpc_thr_hover,
		(ParamInt<px4::params::MPC_THR_CURVE>)      _param_mpc_thr_curve
	)
};

/* ------------------------------------------------------------------------ */

namespace
{

/**
 * ModuleParams::updateParams() is protected and cascades to children, which is how
 * the owning module drives it. Parent the objects under this harness so a param
 * change can be pushed through the real cascade path.
 */
class ParamHarness : public ModuleParams
{
public:
	ParamHarness() : ModuleParams(nullptr) {}
	using ModuleParams::updateParams;
};

void expectSetpointsEqual(const vehicle_attitude_setpoint_s &a, const vehicle_attitude_setpoint_s &b,
			  int step, const char *phase)
{
	for (int i = 0; i < 4; i++) {
		ASSERT_FLOAT_EQ(a.q_d[i], b.q_d[i]) << "q_d[" << i << "] diverged at step " << step << " (" << phase << ")";
	}

	for (int i = 0; i < 3; i++) {
		ASSERT_FLOAT_EQ(a.thrust_body[i], b.thrust_body[i])
				<< "thrust_body[" << i << "] diverged at step " << step << " (" << phase << ")";
	}

	ASSERT_FLOAT_EQ(a.yaw_sp_move_rate, b.yaw_sp_move_rate)
			<< "yaw_sp_move_rate diverged at step " << step << " (" << phase << ")";
}

/**
 * Drive both implementations through an identical randomized flight-like
 * sequence and require exact agreement at every step.
 */
void runDifferentialSweep(bool vtol, int32_t throttle_curve_param)
{
	param_control_autosave(false);
	param_set(param_find("MPC_THR_CURVE"), &throttle_curve_param);

	ParamHarness harness;
	StickToAttitudeSetpoint extracted{&harness};
	ReferenceStickToAttitude reference{&harness};

	extracted.setVtol(vtol);
	reference.setVtol(vtol);

	// Re-read MPC_THR_CURVE after the param_set above (cascades to both children).
	harness.updateParams();
	reference.parameters_updated();

	std::mt19937 rng{20260729u};
	std::uniform_real_distribution<float> stick{-1.f, 1.f};
	std::uniform_real_distribution<float> angle{-3.14f, 3.14f};
	std::uniform_real_distribution<float> dt_dist{0.002f, 0.02f};
	std::uniform_real_distribution<float> hover{0.f, 1.f};

	float yaw = 0.f;

	for (int step = 0; step < 2000; step++) {
		const float dt = dt_dist(rng);

		manual_control_setpoint_s manual{};
		manual.roll = stick(rng);
		manual.pitch = stick(rng);
		manual.yaw = stick(rng);
		manual.throttle = stick(rng);

		// Walk the attitude around so the yaw lock / tilt correction paths are exercised.
		yaw = wrap_pi(yaw + 0.01f * stick(rng));
		const Quatf q{Eulerf(0.2f * stick(rng), 0.2f * stick(rng), yaw)};

		const float unaided_heading = (step % 7 == 0) ? NAN : yaw + 0.01f;
		const bool landed = (step < 200) || (step % 401 == 0);
		const bool spooled_up = (step >= 100);

		// Periodically feed a hover thrust estimate, sometimes flagged invalid.
		if (step % 13 == 0) {
			const float hte = (step % 91 == 0) ? NAN : hover(rng);
			extracted.setHoverThrustEstimate(hte);
			reference.setHoverThrustEstimate(hte);
		}

		// Periodic EKF heading resets.
		if (step % 257 == 0 && step > 0) {
			const float delta_psi = 0.3f * stick(rng);
			extracted.ekfResetHandler(delta_psi);
			reference.ekfResetHandler(delta_psi);
		}

		// Periodic mode-exit resets.
		if (step % 601 == 0 && step > 0) {
			extracted.reset(q, unaided_heading);
			reference.reset(q, unaided_heading);
		}

		vehicle_attitude_setpoint_s sp_extracted{};
		vehicle_attitude_setpoint_s sp_reference{};

		extracted.update(manual, q, unaided_heading, dt, sp_extracted);
		reference.generate_attitude_setpoint(manual, q, unaided_heading, dt, sp_reference);

		expectSetpointsEqual(sp_extracted, sp_reference, step, vtol ? "vtol" : "mc");

		extracted.updateSlewRates(landed, spooled_up, dt);
		reference.updateSlewRates(landed, spooled_up, dt);
	}
}

} // namespace

TEST(StickToAttitudeSetpointTest, MatchesPreExtractionReferenceMulticopter)
{
	runDifferentialSweep(false, 0);
}

TEST(StickToAttitudeSetpointTest, MatchesPreExtractionReferenceVtol)
{
	runDifferentialSweep(true, 0);
}

TEST(StickToAttitudeSetpointTest, MatchesPreExtractionReferenceThrottleCurve1)
{
	runDifferentialSweep(false, 1);
}

TEST(StickToAttitudeSetpointTest, MatchesPreExtractionReferenceThrottleCurve2)
{
	runDifferentialSweep(false, 2);
}

TEST(StickToAttitudeSetpointTest, TiltIsLimitedToMpcManTiltMax)
{
	param_control_autosave(false);

	float tilt_max_deg = 25.f;
	param_set(param_find("MPC_MAN_TILT_MAX"), &tilt_max_deg);
	// Remove the input filter lag so full stick reaches the output in one step.
	float tilt_tau = 0.f;
	param_set(param_find("MC_MAN_TILT_TAU"), &tilt_tau);

	ParamHarness harness;
	StickToAttitudeSetpoint stick_to_attitude{&harness};
	harness.updateParams();

	manual_control_setpoint_s manual{};
	manual.roll = 1.f;
	manual.pitch = 1.f;	// full diagonal stick -> would exceed the limit without clamping
	manual.throttle = 0.f;

	const Quatf q{Eulerf(0.f, 0.f, 0.f)};
	vehicle_attitude_setpoint_s sp{};

	for (int i = 0; i < 200; i++) {
		stick_to_attitude.update(manual, q, 0.f, 0.01f, sp);
	}

	// Tilt angle of the setpoint = angle between its body z axis and the NED z axis.
	const Vector3f z_sp = Quatf(sp.q_d).dcm_z();
	const float tilt = acosf(math::constrain(z_sp(2), -1.f, 1.f));

	EXPECT_NEAR(math::degrees(tilt), tilt_max_deg, 0.5f);
}

TEST(StickToAttitudeSetpointTest, ArmingGestureUnlocksYaw)
{
	param_control_autosave(false);

	int32_t airmode = 0;	// anything but 2 enables the arming gesture
	param_set(param_find("MC_AIRMODE"), &airmode);

	ParamHarness harness;
	StickToAttitudeSetpoint stick_to_attitude{&harness};
	harness.updateParams();

	const Quatf q{Eulerf(0.f, 0.f, 1.f)};	// yaw = 1 rad
	vehicle_attitude_setpoint_s sp{};

	manual_control_setpoint_s manual{};
	manual.throttle = 0.f;

	// Establish a yaw lock first.
	for (int i = 0; i < 50; i++) {
		stick_to_attitude.update(manual, q, NAN, 0.01f, sp);
	}

	// Throttle below -0.9 is the arming gesture: the held yaw setpoint is dropped,
	// so the setpoint must follow the current yaw exactly instead of a stale lock.
	manual.throttle = -0.95f;
	stick_to_attitude.update(manual, q, NAN, 0.01f, sp);

	const float yaw_sp = Eulerf(Quatf(sp.q_d)).psi();
	EXPECT_NEAR(yaw_sp, 1.f, 1e-4f);
}
