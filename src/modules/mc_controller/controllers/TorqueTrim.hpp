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
 * @file TorqueTrim.hpp
 *
 * Body-frame roll/pitch torque trim: the attitude-level integral of the laws in this
 * module (MC_ATT_TRIM_*).
 *
 * WHAT IT REMOVES. A constant body-fixed torque - a centre of gravity off the thrust
 * centroid, one motor weaker than the rest - that a PD attitude law can only hold against
 * with a standing attitude error. Measured on the 2026-09-24 spin flights: roll torque
 * held at -0.010 to -0.018 normalized for whole flights, the right motor ~8 % above the
 * others, and the attitude settling ~0.8 deg short of its setpoint on both axes.
 *
 * WHAT IT INTEGRATES, AND WHY NOT THE ATTITUDE ERROR. It integrates the roll/pitch torque
 * the law is ALREADY commanding and hands that over to a trim term:
 *
 *   d(trim)/dt = MC_ATT_TRIM_I * tau_law,   tau_out = tau_law + trim
 *
 * At steady state tau_law + trim + tau_bias = 0, and the law stops pushing only once the
 * trim carries the whole bias. With the plant much faster than the trim this is
 * d(trim)/dt = -I * (tau_bias + trim): it converges for ANY loop, because the loop's own
 * gain never enters it.
 *
 * Integrating the attitude error directly does not have that property on a spinning
 * vehicle. The steady attitude error a body-fixed torque produces is ROTATED relative to
 * the torque by the loop itself - for the PD law by roughly atan(ATT_D * r / ATT_P), 37 deg
 * at 3 rad/s and 79 deg at 20 rad/s, plus the gyroscopic coupling, and for the eigen law
 * additionally by the seat angle (up to 80 deg). A per-axis integrator on that error
 * diverges once the rotation passes 90 deg. Without spin the two are the same thing: the
 * held torque is ATT_P times the attitude error, so this is an ordinary attitude integral.
 *
 * WHAT IT DOES NOT TOUCH. Anything that rotates in the body frame - a world-fixed tilt
 * demand seen from a spinning body - averages out of the integral, attenuated by I/r. Yaw
 * is never trimmed: with all rotors turning the same way it cannot be allocated, and a yaw
 * integral would only wind up.
 *
 * WHERE IT GOES. After everything that shapes the law's torque (for the eigen law, after
 * the seat and the actuator lead), in normalized units, before the output clamp. The bias
 * is DC in the body frame, so rotating it by the seat angle would be wrong, and keeping it
 * out of the seat's commanded acceleration means the seat compares like with like: the
 * trim and the bias cancel in the measurement as well.
 *
 * COMPOSITION, like TrajectoryStage, RateLimits and Seat. A ModuleParams child, so its
 * gains refresh through the same cascade the owning law does.
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <px4_platform_common/module_params.h>

class TorqueTrim : public ModuleParams
{
public:
	explicit TorqueTrim(ModuleParams *parent);
	~TorqueTrim() override = default;

	TorqueTrim(const TorqueTrim &) = delete;
	TorqueTrim &operator=(const TorqueTrim &) = delete;

	/**
	 * Advance the trim on the law's torque and return the torque with the trim added.
	 *
	 * @param torque_law normalized body-FRD torque the law commands, before the output clamp
	 * @param dt         [s] time since the previous call
	 * @return torque_law with the roll/pitch trim added; yaw passed through untouched
	 */
	matrix::Vector3f apply(const matrix::Vector3f &torque_law, float dt);

	/// Drop the trim. For command.reset_integrals: on the ground, disarmed, level change.
	void reset() { _trim.setZero(); }

	/**
	 * Roll/pitch allocator saturation. While the mixer cannot deliver what is asked, the
	 * torque the law holds says nothing about the bias, so the trim is frozen.
	 */
	void setSaturated(bool saturated) { _allocator_saturated = saturated; }

	/// Normalized roll/pitch trim currently applied. Diagnostics.
	const matrix::Vector2f &trim() const { return _trim; }

	bool enabled() const { return _gain > 0.f; }

protected:
	void updateParams() override;

private:
	matrix::Vector2f _trim{};
	bool _allocator_saturated{false};

	float _gain{0.f};	///< [1/s] MC_ATT_TRIM_I
	float _limit{0.f};	///< normalized, MC_ATT_TRIM_LIM

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MC_ATT_TRIM_I>)   _param_mc_att_trim_i,
		(ParamFloat<px4::params::MC_ATT_TRIM_LIM>) _param_mc_att_trim_lim
	)
};
