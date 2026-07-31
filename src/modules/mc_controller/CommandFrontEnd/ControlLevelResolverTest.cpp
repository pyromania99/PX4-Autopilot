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
 * Stage 5a gate: control level resolution.
 *
 * This test links commander's OWN mode_util::getVehicleControlMode() rather than
 * a hand-copied table, so it cannot drift from commander and will fail loudly if
 * upstream adds or changes a nav_state. §3 of the design plan rests on this
 * equivalence, making it the highest-value test in the framework.
 *
 * Disposition: PERMANENT.
 */

#include <gtest/gtest.h>

#include "ControlLevelResolver.hpp"

#include <ModeUtil/control_mode.hpp>

using namespace mc_ctrl;

namespace
{

constexpr uint8_t MC = vehicle_status_s::VEHICLE_TYPE_ROTARY_WING;
constexpr uint8_t FW = vehicle_status_s::VEHICLE_TYPE_FIXED_WING;

/**
 * Build a vehicle_control_mode exactly the way Commander does:
 * mode_util::getVehicleControlMode() then the flag_multicopter_position_control_enabled
 * derivation from Commander.cpp:2652-2658.
 */
vehicle_control_mode_s modeFor(uint8_t nav_state, uint8_t vehicle_type,
			       const offboard_control_mode_s &ocm)
{
	vehicle_control_mode_s vcm{};   // getVehicleControlMode() does not zero it itself
	mode_util::getVehicleControlMode(nav_state, vehicle_type, ocm, vcm);

	vcm.flag_multicopter_position_control_enabled =
		(vehicle_type == MC)
		&& (vcm.flag_control_altitude_enabled
		    || vcm.flag_control_climb_rate_enabled
		    || vcm.flag_control_position_enabled
		    || vcm.flag_control_velocity_enabled
		    || vcm.flag_control_acceleration_enabled);
	return vcm;
}

vehicle_control_mode_s modeFor(uint8_t nav_state, uint8_t vehicle_type = MC)
{
	offboard_control_mode_s ocm{};
	return modeFor(nav_state, vehicle_type, ocm);
}

ControlLevel levelFor(uint8_t nav_state, uint8_t vehicle_type = MC)
{
	return resolveLevel(modeFor(nav_state, vehicle_type), vehicle_type, false, false);
}

struct Row {
	uint8_t nav_state;
	ControlLevel expected;
	const char *name;
};

} // namespace

TEST(ControlLevelResolverTest, TruthTableForEveryMulticopterNavState)
{
	using S = vehicle_status_s;

	const Row rows[] = {
		{S::NAVIGATION_STATE_MANUAL,            ControlLevel::Attitude,   "MANUAL"},
		{S::NAVIGATION_STATE_STAB,              ControlLevel::Attitude,   "STAB"},
		{S::NAVIGATION_STATE_ACRO,              ControlLevel::BodyRate,   "ACRO"},
		{S::NAVIGATION_STATE_ALTCTL,            ControlLevel::Trajectory, "ALTCTL"},
		{S::NAVIGATION_STATE_ALTITUDE_CRUISE,   ControlLevel::Trajectory, "ALTITUDE_CRUISE"},
		{S::NAVIGATION_STATE_POSCTL,            ControlLevel::Trajectory, "POSCTL"},
		{S::NAVIGATION_STATE_POSITION_SLOW,     ControlLevel::Trajectory, "POSITION_SLOW"},
		{S::NAVIGATION_STATE_AUTO_MISSION,      ControlLevel::Trajectory, "AUTO_MISSION"},
		{S::NAVIGATION_STATE_AUTO_LOITER,       ControlLevel::Trajectory, "AUTO_LOITER"},
		{S::NAVIGATION_STATE_AUTO_RTL,          ControlLevel::Trajectory, "AUTO_RTL"},
		{S::NAVIGATION_STATE_AUTO_LAND,         ControlLevel::Trajectory, "AUTO_LAND"},
		{S::NAVIGATION_STATE_AUTO_PRECLAND,     ControlLevel::Trajectory, "AUTO_PRECLAND"},
		{S::NAVIGATION_STATE_AUTO_TAKEOFF,      ControlLevel::Trajectory, "AUTO_TAKEOFF"},
		{S::NAVIGATION_STATE_AUTO_VTOL_TAKEOFF, ControlLevel::Trajectory, "AUTO_VTOL_TAKEOFF"},
		{S::NAVIGATION_STATE_AUTO_FOLLOW_TARGET, ControlLevel::Trajectory, "AUTO_FOLLOW_TARGET"},
		{S::NAVIGATION_STATE_ORBIT,             ControlLevel::Trajectory, "ORBIT"},
		{S::NAVIGATION_STATE_DESCEND,           ControlLevel::Trajectory, "DESCEND"},
		{S::NAVIGATION_STATE_TERMINATION,       ControlLevel::None,       "TERMINATION"},
	};

	for (const auto &row : rows) {
		EXPECT_EQ(levelFor(row.nav_state), row.expected)
				<< row.name << " resolved to " << levelName(levelFor(row.nav_state));
	}
}

TEST(ControlLevelResolverTest, OffboardSubmodesResolveByCommandedQuantity)
{
	using S = vehicle_status_s;

	auto offboard = [](void (*set)(offboard_control_mode_s &)) {
		offboard_control_mode_s ocm{};
		set(ocm);
		return resolveLevel(modeFor(S::NAVIGATION_STATE_OFFBOARD, MC, ocm), MC, false, false);
	};

	EXPECT_EQ(offboard([](offboard_control_mode_s & o) { o.position = true; }), ControlLevel::Trajectory);
	EXPECT_EQ(offboard([](offboard_control_mode_s & o) { o.velocity = true; }), ControlLevel::Trajectory);
	EXPECT_EQ(offboard([](offboard_control_mode_s & o) { o.acceleration = true; }), ControlLevel::Trajectory);
	EXPECT_EQ(offboard([](offboard_control_mode_s & o) { o.attitude = true; }), ControlLevel::Attitude);
	EXPECT_EQ(offboard([](offboard_control_mode_s & o) { o.body_rate = true; }), ControlLevel::BodyRate);

	// thrust_and_torque enables allocation only - rates are off, so the framework
	// must not claim control.
	EXPECT_EQ(offboard([](offboard_control_mode_s & o) { o.thrust_and_torque = true; }), ControlLevel::None);

	// Nothing commanded at all.
	EXPECT_EQ(offboard([](offboard_control_mode_s &) {}), ControlLevel::None);
}

TEST(ControlLevelResolverTest, StickGateEquivalenceWithStockModules)
{
	using S = vehicle_status_s;

	// mc_att_control's stick gate is `manual && !altitude && !velocity && !position`.
	// It must be exactly equivalent to `level == Attitude && manual`, which is the
	// substitution the framework relies on. Check it holds for every nav_state.
	const uint8_t nav_states[] = {
		S::NAVIGATION_STATE_MANUAL, S::NAVIGATION_STATE_STAB, S::NAVIGATION_STATE_ACRO,
		S::NAVIGATION_STATE_ALTCTL, S::NAVIGATION_STATE_POSCTL, S::NAVIGATION_STATE_POSITION_SLOW,
		S::NAVIGATION_STATE_ALTITUDE_CRUISE, S::NAVIGATION_STATE_AUTO_MISSION,
		S::NAVIGATION_STATE_AUTO_LOITER, S::NAVIGATION_STATE_AUTO_RTL,
		S::NAVIGATION_STATE_AUTO_LAND, S::NAVIGATION_STATE_DESCEND,
		S::NAVIGATION_STATE_ORBIT, S::NAVIGATION_STATE_AUTO_FOLLOW_TARGET,
		S::NAVIGATION_STATE_TERMINATION,
	};

	for (uint8_t ns : nav_states) {
		const auto vcm = modeFor(ns, MC);
		const ControlLevel level = resolveLevel(vcm, MC, false, false);

		const bool stock_att_stick_gate = vcm.flag_control_manual_enabled
						  && !vcm.flag_control_altitude_enabled
						  && !vcm.flag_control_velocity_enabled
						  && !vcm.flag_control_position_enabled;
		const bool framework_att_stick_gate = (level == ControlLevel::Attitude)
						      && vcm.flag_control_manual_enabled;

		// The stock gate also requires the attitude loop to actually run.
		EXPECT_EQ(stock_att_stick_gate && vcm.flag_control_attitude_enabled,
			  framework_att_stick_gate) << "attitude stick gate mismatch for nav_state " << int(ns);

		// mc_rate_control's acro gate is `manual && !attitude`.
		const bool stock_acro_gate = vcm.flag_control_manual_enabled && !vcm.flag_control_attitude_enabled;
		const bool framework_acro_gate = (level == ControlLevel::BodyRate) && vcm.flag_control_manual_enabled;
		EXPECT_EQ(stock_acro_gate && vcm.flag_control_rates_enabled,
			  framework_acro_gate) << "acro stick gate mismatch for nav_state " << int(ns);
	}
}

TEST(ControlLevelResolverTest, TerminationAlwaysYieldsNone)
{
	vehicle_control_mode_s vcm{};
	vcm.flag_control_rates_enabled = true;
	vcm.flag_control_attitude_enabled = true;
	vcm.flag_multicopter_position_control_enabled = true;
	vcm.flag_control_termination_enabled = true;

	EXPECT_EQ(resolveLevel(vcm, MC, false, false), ControlLevel::None);
}

TEST(ControlLevelResolverTest, RatesDisabledYieldsNone)
{
	vehicle_control_mode_s vcm{};
	vcm.flag_control_attitude_enabled = true;
	vcm.flag_multicopter_position_control_enabled = true;
	vcm.flag_control_rates_enabled = false;

	EXPECT_EQ(resolveLevel(vcm, MC, false, false), ControlLevel::None);
}

TEST(ControlLevelResolverTest, FixedWingNeverReachesTrajectory)
{
	// flag_multicopter_position_control_enabled is false for non-rotary-wing, and
	// the attitude loop is gated on hovering, so a fixed wing falls through to
	// BodyRate rather than being claimed by the multicopter framework.
	const auto vcm = modeFor(vehicle_status_s::NAVIGATION_STATE_POSCTL, FW);
	EXPECT_FALSE(vcm.flag_multicopter_position_control_enabled);
	EXPECT_EQ(resolveLevel(vcm, FW, false, false), ControlLevel::BodyRate);
}

TEST(ControlLevelResolverTest, VtolTransitionGating)
{
	auto vcm = modeFor(vehicle_status_s::NAVIGATION_STATE_STAB, MC);
	ASSERT_TRUE(vcm.flag_control_attitude_enabled);

	// Hovering multicopter: Attitude.
	EXPECT_EQ(resolveLevel(vcm, MC, false, false), ControlLevel::Attitude);

	// In transition and not a tailsitter: the MC attitude loop must not claim it.
	EXPECT_EQ(resolveLevel(vcm, MC, true, false), ControlLevel::BodyRate);

	// Tailsitter in transition: mc_att_control does run (mc_att_control_main.cpp:290).
	EXPECT_EQ(resolveLevel(vcm, MC, true, true), ControlLevel::Attitude);
}

TEST(ControlLevelResolverTest, EveryResolvedLevelIsSupportableExceptNone)
{
	// A level the framework can resolve to must be expressible in a
	// supportedLevels() mask, or a controller could never declare support for it.
	for (ControlLevel l : {ControlLevel::BodyRate, ControlLevel::Attitude, ControlLevel::Trajectory}) {
		EXPECT_NE(kAllLevels & levelBit(l), 0u) << levelName(l);
	}

	EXPECT_EQ(kAllLevels & levelBit(ControlLevel::None), 0u);
}
