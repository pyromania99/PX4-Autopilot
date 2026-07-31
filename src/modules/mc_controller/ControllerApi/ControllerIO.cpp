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

#include "ControllerIO.hpp"

#include <uORB/topics/mc_controller_status.h>

namespace mc_ctrl
{

// The enums are published directly as mc_controller_status fields. Keep the two
// definitions from drifting apart.
static_assert(static_cast<uint8_t>(ControlLevel::None) == mc_controller_status_s::CONTROL_LEVEL_NONE, "");
static_assert(static_cast<uint8_t>(ControlLevel::BodyRate) == mc_controller_status_s::CONTROL_LEVEL_BODY_RATE, "");
static_assert(static_cast<uint8_t>(ControlLevel::Attitude) == mc_controller_status_s::CONTROL_LEVEL_ATTITUDE, "");
static_assert(static_cast<uint8_t>(ControlLevel::Trajectory) == mc_controller_status_s::CONTROL_LEVEL_TRAJECTORY, "");

const char *levelName(ControlLevel level)
{
	switch (level) {
	case ControlLevel::None: return "none";

	case ControlLevel::BodyRate: return "body_rate";

	case ControlLevel::Attitude: return "attitude";

	case ControlLevel::Trajectory: return "trajectory";
	}

	return "unknown";
}

bool outputIsFinite(const ControllerOutput &output)
{
	for (int i = 0; i < 3; i++) {
		if (!PX4_ISFINITE(output.torque(i)) || !PX4_ISFINITE(output.thrust(i))) {
			return false;
		}
	}

	return true;
}

} // namespace mc_ctrl
