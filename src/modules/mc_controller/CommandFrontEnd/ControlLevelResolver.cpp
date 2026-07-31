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

#include "ControlLevelResolver.hpp"

namespace mc_ctrl
{

ControlLevel resolveLevel(const vehicle_control_mode_s &vcm, uint8_t vehicle_type,
			  bool in_transition, bool is_tailsitter)
{
	// Nothing to do when control is disabled entirely. Mirrors the inverse of
	// mc_rate_control's run condition (MulticopterRateControl.cpp:179).
	if (vcm.flag_control_termination_enabled || !vcm.flag_control_rates_enabled) {
		return ControlLevel::None;
	}

	// mc_pos_control's own run condition.
	if (vcm.flag_multicopter_position_control_enabled) {
		return ControlLevel::Trajectory;
	}

	// mc_att_control runs the attitude loop only while hovering, or while a
	// tailsitter is transitioning (mc_att_control_main.cpp:289-293).
	const bool hovering = (vehicle_type == vehicle_status_s::VEHICLE_TYPE_ROTARY_WING) && !in_transition;
	const bool tailsitter_transition = is_tailsitter && in_transition;

	if (vcm.flag_control_attitude_enabled && (hovering || tailsitter_transition)) {
		return ControlLevel::Attitude;
	}

	// Rates are enabled but attitude is not: Acro, or offboard body_rate.
	return ControlLevel::BodyRate;
}

} // namespace mc_ctrl
