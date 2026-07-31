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

#include "TemplateController.hpp"

#include <mathlib/math/Limits.hpp>

using namespace matrix;

void TemplateController::reset()
{
	_rate_integral.setZero();
}

bool TemplateController::update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
				float dt, mc_ctrl::ControllerOutput &output)
{
	// HARD CONTRACT: honour this or the integrator winds up while on the ground.
	if (command.reset_integrals) {
		_rate_integral.setZero();
	}

	Vector3f rate_setpoint{};
	Vector3f thrust_setpoint{};

	switch (command.level) {
	case mc_ctrl::ControlLevel::Trajectory: {
			// Minimal example: proportional position -> tilt, and a collective from the
			// hover thrust estimate. Replace with your outer loop.
			Vector3f position_error{};

			for (int i = 0; i < 3; i++) {
				position_error(i) = PX4_ISFINITE(command.position_sp(i))
						    ? (command.position_sp(i) - state.position(i)) : 0.f;
			}

			const float collective = math::constrain(state.hover_thrust + 0.2f * position_error(2),
						 command.thrust_min, command.thrust_max);
			thrust_setpoint = Vector3f(0.f, 0.f, -collective);

			// Tilt toward the horizontal error, respecting the framework's tilt limit.
			const float tilt_limit = PX4_ISFINITE(command.tilt_limit) ? command.tilt_limit : 0.5f;
			Vector2f tilt(math::constrain(0.1f * position_error(0), -tilt_limit, tilt_limit),
				      math::constrain(0.1f * position_error(1), -tilt_limit, tilt_limit));

			const Quatf attitude_setpoint(Eulerf(-tilt(1), tilt(0),
							     PX4_ISFINITE(command.yaw_sp) ? command.yaw_sp : state.heading));
			const Quatf error = attitude_setpoint.inversed() * state.q;
			rate_setpoint = -2.f * _param_att_p.get() * error.imag() * ((error(0) >= 0.f) ? 1.f : -1.f);
			output.attitude_setpoint = attitude_setpoint;
			break;
		}

	case mc_ctrl::ControlLevel::Attitude: {
			const Quatf error = command.attitude_sp.inversed() * state.q;
			rate_setpoint = -2.f * _param_att_p.get() * error.imag() * ((error(0) >= 0.f) ? 1.f : -1.f);
			thrust_setpoint = command.thrust_body_sp;
			output.attitude_setpoint = command.attitude_sp;
			break;
		}

	case mc_ctrl::ControlLevel::BodyRate:
		rate_setpoint = command.rate_sp;
		thrust_setpoint = command.thrust_body_sp;
		break;

	case mc_ctrl::ControlLevel::None:
	default:
		return false;
	}

	// Inner loop: simple PI on body rates.
	const Vector3f rate_error = rate_setpoint - state.angular_velocity;

	if (!state.landed) {
		_rate_integral += rate_error * _param_rate_i.get() * dt;

		for (int i = 0; i < 3; i++) {
			_rate_integral(i) = math::constrain(_rate_integral(i), -0.3f, 0.3f);
		}
	}

	Vector3f torque = rate_error * _param_rate_p.get() + _rate_integral;

	for (int i = 0; i < 3; i++) {
		torque(i) = math::constrain(torque(i), -1.f, 1.f);
	}

	output.torque = torque;
	output.thrust = thrust_setpoint;	// REQUIRED, in both output modes
	output.rate_setpoint = rate_setpoint;
	output.valid = true;
	return true;
}

void TemplateController::fillStatus(mc_controller_status_s &status) const
{
	status.debug[0] = _rate_integral(0);
	status.debug[1] = _rate_integral(1);
	status.debug[2] = _rate_integral(2);
}
