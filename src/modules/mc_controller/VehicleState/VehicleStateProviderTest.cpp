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
 * Stage 4 gate: VehicleStateProvider behaviour.
 *
 * Proves the filter chain, EKF-reset bookkeeping, freshness/dt clamping and
 * NaN semantics. Numerical equality with the stock mc_pos_control filter chain
 * is NOT proven here - that arrives transitively at Stage 6's differential test
 * against the hand-wired stock triple.
 *
 * Disposition: PERMANENT.
 */

#include <gtest/gtest.h>

#include "VehicleStateProvider.hpp"

#include <parameters/param.h>
#include <px4_platform_common/defines.h>

using namespace matrix;

namespace
{

class ParamHarness : public ModuleParams
{
public:
	ParamHarness() : ModuleParams(nullptr) {}
	using ModuleParams::updateParams;
};

/// Local position sample at `t_us` with everything valid.
vehicle_local_position_s makeLocalPos(uint64_t t_us, float vx = 0.f, float vy = 0.f, float vz = 0.f)
{
	vehicle_local_position_s lp{};
	lp.timestamp = t_us;
	lp.timestamp_sample = t_us;
	lp.xy_valid = true;
	lp.z_valid = true;
	lp.v_xy_valid = true;
	lp.v_z_valid = true;
	lp.x = 1.f;
	lp.y = 2.f;
	lp.z = -5.f;
	lp.vx = vx;
	lp.vy = vy;
	lp.vz = vz;
	lp.heading = 0.3f;
	lp.unaided_heading = 0.31f;
	return lp;
}

vehicle_angular_velocity_s makeGyro(uint64_t t_us)
{
	vehicle_angular_velocity_s av{};
	av.timestamp = t_us;
	av.timestamp_sample = t_us;
	av.xyz[0] = 0.01f;
	av.xyz[1] = 0.02f;
	av.xyz[2] = 0.03f;
	return av;
}

} // namespace

TEST(VehicleStateProviderTest, PositionPassesThroughWhenValid)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	p.updateLocalPosition(makeLocalPos(1000000));
	const auto &s = p.getState();

	EXPECT_TRUE(s.position_valid_xy);
	EXPECT_TRUE(s.position_valid_z);
	EXPECT_FLOAT_EQ(s.position(0), 1.f);
	EXPECT_FLOAT_EQ(s.position(1), 2.f);
	EXPECT_FLOAT_EQ(s.position(2), -5.f);
	EXPECT_FLOAT_EQ(s.heading, 0.3f);
	EXPECT_FLOAT_EQ(s.unaided_heading, 0.31f);
}

TEST(VehicleStateProviderTest, InvalidPositionBecomesNaN)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	auto lp = makeLocalPos(1000000);
	lp.xy_valid = false;
	lp.z_valid = false;
	p.updateLocalPosition(lp);
	const auto &s = p.getState();

	EXPECT_FALSE(s.position_valid_xy);
	EXPECT_FALSE(s.position_valid_z);
	EXPECT_FALSE(PX4_ISFINITE(s.position(0)));
	EXPECT_FALSE(PX4_ISFINITE(s.position(1)));
	EXPECT_FALSE(PX4_ISFINITE(s.position(2)));
}

TEST(VehicleStateProviderTest, VelocityLowPassConvergesToDC)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	// A constant velocity must pass through the notch + low-pass chain unattenuated
	// once settled: the filters must not introduce steady-state bias.
	uint64_t t = 1000000;

	for (int i = 0; i < 2000; i++) {
		p.updateLocalPosition(makeLocalPos(t, 3.f, -2.f, 1.f));
		t += 10000;   // 100 Hz
	}

	const auto &s = p.getState();
	EXPECT_NEAR(s.velocity(0), 3.f, 0.02f);
	EXPECT_NEAR(s.velocity(1), -2.f, 0.02f);
	EXPECT_NEAR(s.velocity(2), 1.f, 0.02f);

	// Constant velocity => acceleration must settle to zero.
	EXPECT_NEAR(s.acceleration(0), 0.f, 0.05f);
	EXPECT_NEAR(s.acceleration(2), 0.f, 0.05f);
}

TEST(VehicleStateProviderTest, LosingVelocityValidityResetsFiltersAndNaNs)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	uint64_t t = 1000000;

	for (int i = 0; i < 200; i++) {
		p.updateLocalPosition(makeLocalPos(t, 5.f, 5.f, 5.f));
		t += 10000;
	}

	ASSERT_NEAR(p.getState().velocity(0), 5.f, 0.1f);

	// Lose validity: state must go NaN and the filters must be zeroed so that
	// regaining velocity does not produce a huge derivative spike.
	auto lp = makeLocalPos(t, 5.f, 5.f, 5.f);
	lp.v_xy_valid = false;
	lp.v_z_valid = false;
	p.updateLocalPosition(lp);
	t += 10000;

	EXPECT_FALSE(p.getState().velocity_valid_xy);
	EXPECT_FALSE(PX4_ISFINITE(p.getState().velocity(0)));
	EXPECT_FALSE(PX4_ISFINITE(p.getState().acceleration(0)));

	// Regain validity - the first acceleration sample must stay bounded.
	p.updateLocalPosition(makeLocalPos(t, 5.f, 5.f, 5.f));
	EXPECT_TRUE(p.getState().velocity_valid_xy);
	EXPECT_TRUE(PX4_ISFINITE(p.getState().acceleration(0)));
	EXPECT_LT(fabsf(p.getState().acceleration(0)), 2000.f) << "derivative spike after regaining velocity";
}

TEST(VehicleStateProviderTest, FirstSampleIsNotTreatedAsAnEkfReset)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	// A fresh subscription arriving with non-zero reset counters must not be
	// reported as a reset, or every startup would inject a spurious jump.
	auto lp = makeLocalPos(1000000);
	lp.xy_reset_counter = 7;
	lp.vxy_reset_counter = 3;
	lp.heading_reset_counter = 2;
	lp.delta_xy[0] = 99.f;
	p.updateLocalPosition(lp);

	EXPECT_FALSE(p.getState().resets.any());
}

TEST(VehicleStateProviderTest, EkfResetDeltasAreLatchedThenClearedByEndCycle)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	uint64_t t = 1000000;
	p.updateLocalPosition(makeLocalPos(t));   // establishes counters
	p.endCycle();
	t += 10000;

	auto lp = makeLocalPos(t);
	lp.xy_reset_counter = 1;
	lp.delta_xy[0] = 2.5f;
	lp.delta_xy[1] = -1.5f;
	lp.z_reset_counter = 1;
	lp.delta_z = 0.75f;
	lp.heading_reset_counter = 1;
	lp.delta_heading = 0.2f;
	p.updateLocalPosition(lp);

	const auto &r = p.getState().resets;
	EXPECT_TRUE(r.xy);
	EXPECT_TRUE(r.z);
	EXPECT_TRUE(r.heading);
	EXPECT_FLOAT_EQ(r.delta_xy(0), 2.5f);
	EXPECT_FLOAT_EQ(r.delta_xy(1), -1.5f);
	EXPECT_FLOAT_EQ(r.delta_z, 0.75f);
	EXPECT_FLOAT_EQ(r.delta_heading, 0.2f);

	// One-shot: consumed by the front-end, then cleared.
	p.endCycle();
	EXPECT_FALSE(p.getState().resets.any());

	// A subsequent sample with the SAME counters must not re-report the reset.
	t += 10000;
	auto lp2 = makeLocalPos(t);
	lp2.xy_reset_counter = 1;
	lp2.z_reset_counter = 1;
	lp2.heading_reset_counter = 1;
	p.updateLocalPosition(lp2);
	EXPECT_FALSE(p.getState().resets.any());
}

TEST(VehicleStateProviderTest, VelocityResetCarriesFilterStateAcrossDiscontinuity)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	uint64_t t = 1000000;

	for (int i = 0; i < 300; i++) {
		p.updateLocalPosition(makeLocalPos(t, 4.f, 0.f, 0.f));
		p.endCycle();
		t += 10000;
	}

	ASSERT_NEAR(p.getState().velocity(0), 4.f, 0.1f);

	// EKF jumps velocity by +3: the filter state is shifted by the same delta so
	// the derivative does not see a step (MulticopterPositionControl.cpp:717-720).
	auto lp = makeLocalPos(t, 7.f, 0.f, 0.f);
	lp.vxy_reset_counter = 1;
	lp.delta_vxy[0] = 3.f;
	p.updateLocalPosition(lp);

	EXPECT_TRUE(p.getState().resets.vxy);
	EXPECT_NEAR(p.getState().velocity(0), 7.f, 0.5f) << "filter should follow the reset immediately";
	EXPECT_LT(fabsf(p.getState().acceleration(0)), 200.f) << "reset must not produce a derivative spike";
}

TEST(VehicleStateProviderTest, QuaternionResetIsLatchedFromAttitude)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	vehicle_attitude_s att{};
	att.timestamp_sample = 1000000;
	att.q[0] = 1.f;
	att.quat_reset_counter = 0;
	p.updateAttitude(att);
	p.endCycle();

	att.timestamp_sample = 1004000;
	att.quat_reset_counter = 1;
	att.delta_q_reset[0] = 0.9987503f;
	att.delta_q_reset[3] = 0.0499792f;   // ~5.7 deg yaw
	p.updateAttitude(att);

	EXPECT_TRUE(p.getState().resets.quat);
	EXPECT_NEAR(p.getState().resets.delta_q(0), 0.9987503f, 1e-6f);
}

TEST(VehicleStateProviderTest, FirstAttitudeSampleIsNotTreatedAsAQuaternionReset)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	// Subscribing after the estimator has already reset a few times: the counter is
	// non-zero on the very first sample we see, and delta_q_reset still carries the
	// last reset's rotation. That is history, not a reset we lived through.
	vehicle_attitude_s att{};
	att.timestamp_sample = 1000000;
	att.q[0] = 1.f;
	att.quat_reset_counter = 4;
	att.delta_q_reset[0] = 0.9987503f;
	att.delta_q_reset[3] = 0.0499792f;
	p.updateAttitude(att);

	EXPECT_FALSE(p.getState().resets.quat);

	// A genuine reset after that is still caught.
	p.endCycle();
	att.timestamp_sample = 1004000;
	att.quat_reset_counter = 5;
	p.updateAttitude(att);

	EXPECT_TRUE(p.getState().resets.quat);
}

TEST(VehicleStateProviderTest, ResetReArmsTheQuaternionCounterGuard)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	vehicle_attitude_s att{};
	att.timestamp_sample = 1000000;
	att.q[0] = 1.f;
	att.quat_reset_counter = 1;
	p.updateAttitude(att);
	p.endCycle();

	p.reset();

	// After reset() the next sample is a first sample again, whatever the counter says.
	att.timestamp_sample = 2000000;
	att.quat_reset_counter = 9;
	att.delta_q_reset[0] = 0.9987503f;
	p.updateAttitude(att);

	EXPECT_FALSE(p.getState().resets.quat);
}

TEST(VehicleStateProviderTest, FreshnessFlagsAndDtClamps)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	uint64_t t = 1000000;

	// Position arriving at 1/10 the gyro rate: position_new must only be set on
	// the cycles where a sample actually arrived.
	p.updateLocalPosition(makeLocalPos(t));
	p.updateAngularVelocity(makeGyro(t));
	EXPECT_TRUE(p.getState().freshness.position_new);
	p.endCycle();
	EXPECT_FALSE(p.getState().freshness.position_new);

	for (int i = 1; i < 10; i++) {
		p.updateAngularVelocity(makeGyro(t + i * 1000));
		EXPECT_FALSE(p.getState().freshness.position_new) << "cycle " << i;
		p.endCycle();
	}

	// dt clamps
	p.updateAngularVelocity(makeGyro(t + 10 * 1000));
	EXPECT_GE(p.getState().freshness.dt, 0.000125f);
	EXPECT_LE(p.getState().freshness.dt, 0.02f);

	// A huge gap must clamp rather than produce an enormous dt.
	p.endCycle();
	p.updateAngularVelocity(makeGyro(t + 5000000));
	EXPECT_FLOAT_EQ(p.getState().freshness.dt, 0.02f);

	// Position dt clamps to its own (different) range.
	p.updateLocalPosition(makeLocalPos(t + 10000000));
	EXPECT_FLOAT_EQ(p.getState().freshness.dt_position, 0.04f);
}

TEST(VehicleStateProviderTest, HoverThrustFallsBackToParameter)
{
	param_control_autosave(false);
	float hover = 0.42f;
	param_set(param_find("MPC_THR_HOVER"), &hover);

	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	EXPECT_FALSE(p.getState().hover_thrust_valid);
	EXPECT_NEAR(p.getState().hover_thrust, 0.42f, 1e-6f);

	p.setHoverThrustEstimate(0.6f);
	EXPECT_TRUE(p.getState().hover_thrust_valid);
	EXPECT_NEAR(p.getState().hover_thrust, 0.6f, 1e-6f);

	// Out-of-range estimates are constrained, matching mc_att_control.
	p.setHoverThrustEstimate(0.99f);
	EXPECT_NEAR(p.getState().hover_thrust, 0.9f, 1e-6f);

	// Invalid estimate slews back to the parameter.
	p.setHoverThrustEstimate(NAN);
	EXPECT_FALSE(p.getState().hover_thrust_valid);
	EXPECT_NEAR(p.getState().hover_thrust, 0.42f, 1e-6f);
}

TEST(VehicleStateProviderTest, ResetClearsEverything)
{
	param_control_autosave(false);
	ParamHarness h;
	VehicleStateProvider p{&h};
	h.updateParams();

	uint64_t t = 1000000;

	for (int i = 0; i < 100; i++) {
		p.updateLocalPosition(makeLocalPos(t, 5.f, 5.f, 5.f));
		p.endCycle();
		t += 10000;
	}

	vehicle_land_detected_s ld{};
	ld.landed = false;
	p.updateLandDetected(ld);
	p.setArmed(true, true);
	ASSERT_FALSE(p.getState().landed);

	p.reset();

	const auto &s = p.getState();
	EXPECT_TRUE(s.landed);
	EXPECT_FALSE(s.armed);
	EXPECT_FALSE(s.position_valid_xy);
	EXPECT_FALSE(s.resets.any());
}
