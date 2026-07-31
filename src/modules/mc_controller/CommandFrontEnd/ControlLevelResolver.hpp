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
 * @file ControlLevelResolver.hpp
 * @brief Decide which control level the active flight mode commands at.
 *
 * This is the single most correctness-critical function in the framework: it
 * must reproduce the gating of the three stock modules exactly, or a controller
 * will be entered at the wrong level.
 *
 * Equivalence argument (verified against every nav_state in
 * src/modules/commander/ModeUtil/control_mode.cpp by ControlLevelResolverTest):
 *
 *  - `flag_multicopter_position_control_enabled` is derived in Commander.cpp:2652
 *    as rotary_wing && (altitude || climb_rate || position || velocity ||
 *    acceleration). That is exactly mc_pos_control's own run condition, so it
 *    defines Trajectory level.
 *
 *  - mc_att_control gates stick generation on
 *    `manual && !altitude && !velocity && !position`
 *    (mc_att_control_main.cpp:297-300). No manual multicopter mode sets
 *    acceleration or climb_rate without also setting altitude, so that predicate
 *    is identical to `manual && !flag_multicopter_position_control_enabled`,
 *    i.e. Attitude level with command.manual set.
 *
 *  - mc_rate_control gates acro stick generation on `manual && !attitude`
 *    (MulticopterRateControl.cpp:156), i.e. BodyRate level with command.manual.
 */

#pragma once

#include <ControllerIO.hpp>

#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_status.h>

namespace mc_ctrl
{

/**
 * @param vcm the published vehicle_control_mode
 * @param vehicle_type    vehicle_status.vehicle_type
 * @param in_transition   vehicle_status.in_transition_mode
 * @param is_tailsitter   vehicle_status.is_vtol_tailsitter
 */
ControlLevel resolveLevel(const vehicle_control_mode_s &vcm, uint8_t vehicle_type,
			  bool in_transition, bool is_tailsitter);

inline ControlLevel resolveLevel(const vehicle_control_mode_s &vcm, const vehicle_status_s &vs)
{
	return resolveLevel(vcm, vs.vehicle_type, vs.in_transition_mode, vs.is_vtol_tailsitter);
}

} // namespace mc_ctrl
