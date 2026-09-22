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
 * @file Seat.hpp
 * @brief Command-frame rotation of the roll/pitch torque ("the seat"), which cancels
 *        the phase rotation actuator lag applies under a continuous spin. MC_SEAT_*.
 *
 * THE PROBLEM. The vehicle spins about yaw at rate r. A roll/pitch torque commanded now
 * is delivered a few milliseconds late - rotor/ESC lag T plus any transport delay tau_a -
 * and by then the body has rotated. At the spin frequency that delay is a pure ROTATION
 * of the commanded roll/pitch vector, so commanded roll arrives partly as pitch.
 *
 * THE FIX. Rotate the commanded wrench by theta_s before it is mixed:
 *
 *     ux = cos(theta_s)*tx - sin(theta_s)*ty
 *     uy = sin(theta_s)*tx + cos(theta_s)*ty
 *
 * and adapt theta_s in flight from the misalignment between the acceleration a LAGLESS
 * actuator would have produced and the one measured. One scalar of state, two
 * transcendentals per tick, no derivative and no dt in the rotation itself.
 *
 * WHY IT MATTERS HERE. Validated at level 1 (Pegasus + Python) at km = 6e-7, i.e.
 * r = 29.5 rad/s, with 20 ms of rotor lag: the eigen law WITHOUT the seat lost control
 * at t = 13.7 s, and WITH it flew 120 s at 0.19 deg roll/pitch RMS. That is the regime
 * the rotor_tau placeholder was recorded as deciding (see the Pegasus commit log), so
 * this is not a refinement - below ~20 rad/s it changes little, and at 30 rad/s it is
 * the difference between flying and not.
 *
 * TWO UPDATE LAWS, and CROSS is the default:
 *
 *   ANGLE  theta_s -= k * alpha * dt, with alpha = atan2(xd x xa, xd . xa). atan2
 *          normalises the magnitudes away, so the step is full-size on every tick no
 *          matter how little signal produced it.
 *   CROSS  theta_s -= k * (xd x xa) * dt = k |xd||xa| sin(alpha). Near convergence
 *          sin(a) ~ a, so it is the same gradient and the same fixed point; away from it
 *          the step is weighted by the signal magnitudes, so a starved channel suppresses
 *          its own update. That removes the epsilon gate and the atan2 NaN path, and at
 *          level 1 it cut theta_s wander 3x in hover. Note k is then DIMENSIONAL.
 *
 * WHAT THE SEAT IS NOT. It corrects the delay's DIRECTION, not its magnitude: the lag
 * also costs a factor 1/sqrt(1+(rT)^2) of torque (7% at r=20, T=20ms), which no rotation
 * recovers. And theta_s converged at level 1 to well below the model angle
 * psi = r*tau_a + atan(r*T) - 0.15 rad against psi = 0.53 at r = 29.5 - so do not read a
 * converged theta_s as an estimate of the lag.
 *
 * The level-1 twin is examples/utils/seat.py in the PegasusSimulator checkout. Keep the
 * two in step: the point of the ladder is that level 2 minus level 1 is PX4's cost, which
 * only means something if both rungs run the same law.
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <px4_platform_common/module_params.h>

class Seat : public ModuleParams
{
public:
	enum class Mode : int32_t {
		Off      = 0,	///< pass the torque through untouched
		Fixed    = 1,	///< theta_s = r*tau_a + atan(r*T), from the live spin rate
		Adaptive = 2,	///< theta_s driven by the measured misalignment
	};

	enum class Law : int32_t {
		Angle = 0,	///< drive on atan2(alpha)
		Cross = 1,	///< drive on the raw cross product. Default.
	};

	/**
	 * Realizability ceiling on |theta_s|. Holding an angle theta costs a pole
	 * magnitude growing as 1/cos(theta), so pi/2 is where the required authority
	 * diverges. Stop short of it rather than at it.
	 */
	static constexpr float kThetaCeiling = 1.5f;

	explicit Seat(ModuleParams *parent);
	~Seat() override = default;

	Seat(const Seat &) = delete;
	Seat &operator=(const Seat &) = delete;

	/// Drop theta_s. Call wherever the owning controller resets.
	void reset();

	/**
	 * Adapt theta_s, then rotate the roll/pitch torque by it.
	 *
	 * Real-time safe: no allocation, no blocking, two transcendentals. Returns
	 * @p torque unchanged when MC_SEAT_MODE is 0.
	 *
	 * @param torque    body torque [N m]. Only roll/pitch are rotated; yaw passes through.
	 * @param xd        commanded roll/pitch angular acceleration [rad/s^2], i.e. what a
	 *                  LAGLESS actuator would have produced from @p torque.
	 * @param xa        achieved roll/pitch angular acceleration [rad/s^2], with the
	 *                  modelled plant drift already removed - so it is the part the
	 *                  delivered wrench caused.
	 * @param spin_rate measured yaw rate r [rad/s]. Gates adaptation via MC_SEAT_RMIN.
	 * @param dt        [s] control timestep. Used by the adaptation only.
	 */
	matrix::Vector3f apply(const matrix::Vector3f &torque,
			       const matrix::Vector2f &xd,
			       const matrix::Vector2f &xa,
			       float spin_rate, float dt);

	/**
	 * Allocator saturation, which freezes the adaptation when MC_SEAT_SATGATE is 1.
	 *
	 * Once the mixer clips, the achieved direction stops following the commanded one
	 * for reasons that have nothing to do with delay geometry, and integrating that is
	 * winding up on error the seat itself caused.
	 *
	 * HOW MUCH THIS GATE DOES, measured 2026-09-19. Across all 51 archived level-2/3
	 * ulogs, roll/pitch allocator saturation occurs on every single one, from 10 % of
	 * ticks to 95 %. On the arms that were adapting, the gate explains the freezing and
	 * nothing else does: P(gated | saturated) is 99-100 % and P(gated | not saturated)
	 * is 0 %. One run had the angle law frozen for 80 % of the flight, and the skipped
	 * ticks were not a random sample - |xd| on them ran 1.3x to 32x the ticks that were
	 * used, i.e. the gate removed exactly the ticks Law::Cross would have weighted most.
	 *
	 * IT IS STILL THE DEFAULT, because the flying baseline was established with it on:
	 * L3 hover, 20 ms pole, r = 29.4, n = 5, theta_s = -0.4731 +/- 0.0009. Turning it
	 * off is an experiment against that baseline, not a fix - hence the parameter.
	 * Level 1 never drove this flag at all before 2026-09-19, so SATGATE = 1 matches PX4
	 * to its own archive and SATGATE = 0 matches level 1 to its own.
	 */
	void setSaturated(bool saturated) { _saturated = saturated; }

	/// Rotation currently applied [rad]. Logged as mc_controller_status.seat_theta.
	float theta() const { return _theta_s; }

	/// Whether the allocator gate is currently freezing the update.
	bool saturated() const { return _saturated; }

	/// Misalignment measured on the last adapting tick [rad], NAN while gated.
	float alpha() const { return _alpha; }

	/**
	 * The misalignment between a commanded and an achieved direction [rad], as a pure
	 * function of the pair. NAN when either vector is too short to carry a direction.
	 *
	 * Stateless and mode-free ON PURPOSE. alpha() is the LAW's alpha: it is written
	 * only on a tick that adapts, so it is NAN for the whole of a seat-OFF or
	 * seat-FIXED flight - and logging that made the misalignment look unmeasurable on
	 * exactly the arms you want to compare the seat against. This is the MEASUREMENT,
	 * and the caller runs it on every arm. It applies nothing and decides nothing.
	 */
	static float measure(const matrix::Vector2f &xd, const matrix::Vector2f &xa);

	/// Cached MC_SEAT_MODE, so a caller can skip the work entirely when disabled.
	Mode mode() const { return _mode; }

	/// The model lag rotation at spin rate @p r [rad]. Fixed mode's target.
	float psiModel(float r) const;

protected:
	void updateParams() override;

private:
	void adapt(const matrix::Vector2f &xd, const matrix::Vector2f &xa, float r, float dt);

	float _theta_s{0.f};	///< [rad] the one piece of adaptation state
	float _alpha{NAN};	///< [rad] last measured misalignment, diagnostics only
	bool _saturated{false};

	// Cached parameters, so nothing here touches the parameter system at gyro rate.
	Mode _mode{Mode::Off};
	Law _law{Law::Cross};
	float _k{0.f};		///< MC_SEAT_K
	float _theta_max{0.f};	///< [rad] MC_SEAT_THMAX, itself ceilinged below pi/2
	float _r_min{0.f};	///< [rad/s] MC_SEAT_RMIN
	float _k_sign{1.f};	///< +/-1 MC_SEAT_KSIGN
	float _tau_a{0.f};	///< [s] MC_SEAT_TAU_A, fixed mode only
	float _T{0.f};		///< [s] MC_SEAT_T, fixed mode only
	bool _sat_gate{true};	///< MC_SEAT_SATGATE: let allocator saturation freeze the angle

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::MC_SEAT_MODE>)    _param_mc_seat_mode,
		(ParamInt<px4::params::MC_SEAT_LAW>)     _param_mc_seat_law,
		(ParamInt<px4::params::MC_SEAT_KSIGN>)   _param_mc_seat_ksign,
		(ParamFloat<px4::params::MC_SEAT_K>)     _param_mc_seat_k,
		(ParamFloat<px4::params::MC_SEAT_THMAX>) _param_mc_seat_thmax,
		(ParamFloat<px4::params::MC_SEAT_RMIN>)  _param_mc_seat_rmin,
		(ParamFloat<px4::params::MC_SEAT_TAU_A>) _param_mc_seat_tau_a,
		(ParamFloat<px4::params::MC_SEAT_T>)     _param_mc_seat_t,
		(ParamInt<px4::params::MC_SEAT_SATGATE>) _param_mc_seat_satgate
	)
};
