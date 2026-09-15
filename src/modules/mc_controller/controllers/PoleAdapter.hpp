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

#pragma once

#include <matrix/matrix/math.hpp>
#include <px4_platform_common/module_params.h>

/**
 * @file PoleAdapter.hpp
 * @brief Adapts theta_b, the ANGLE of the eigen law's designed pole. MC_POLE_*.
 *
 * WHAT IT IS, AND WHAT IT IS NOT. This is not the seat. The seat rotates the commanded
 * wrench on its way to the mixer and changes nothing about the design; this changes WHERE
 * THE DESIGNED CLOSED-LOOP POLE SITS and rotates nothing. The two are independent and can
 * run together or separately.
 *
 * The eigen law's roll/pitch block is, in polar form,
 *
 *     M_dyn = -rho * R(theta_pole),   rho = hypot(MC_EIG_WN, MC_EIG_B),
 *                                     theta_pole = atan2(MC_EIG_B, MC_EIG_WN)
 *
 * placing the pole pair at -rho*e^{+/- j theta_pole}, so the damping ratio is
 * cos(theta_pole). This class adds an OFFSET to that angle:
 *
 *     theta_pole_effective = theta_pole + theta_b
 *
 * with theta_b driven by the misalignment between the acceleration M_dyn asks the rate
 * vector for and the acceleration actually achieved:
 *
 *     alpha_b   = signed angle from (M_dyn . (p,q)) to xa
 *     theta_b  -= k_b * alpha_b * dt
 *
 * theta_b is an OFFSET, so MC_POLE_K = 0 (or MC_POLE_EN = 0) reproduces the unmodified
 * controller exactly - the same property the level-1 twin has, and the reason it is
 * written as an offset rather than as an absolute angle.
 *
 * EXPECT THIS TO FAIL, AND LET IT. Measured in the level-1 harness (seat_harness_v3.py):
 * this alpha_b has NO INTERIOR FIXED POINT. |alpha_b| stayed >= 1.12 rad however slowly
 * theta_b was moved, so theta_b is driven monotonically to its clip and slowing it only
 * reduces the damage - at k_s/k_b = 1000 the arm converges on "do nothing at all". Brief
 * v3 section 4 anticipates exactly this and says the SIGNAL is what to redesign, not the
 * scheme. It is ported here unchanged anyway, because the point of the ladder is that a
 * rung's result only means something if both rungs ran the same law: a level-2 result
 * that differs from the harness is information, and pre-judging it away is not.
 *
 * MODE 2 - EXTREMUM SEEKING, and why it is not the same mistake twice.
 *
 * Mode 1's observable is ill-formed for a structural reason, not a tuning one. Its
 * reference is M_dyn.x, which ROTATES WITH theta_b - so the parameter appears on both
 * sides of the angle and very nearly cancels out of it. Working the algebra through with
 * A_r = i*lam*r and M_dyn = -rho*e^{i theta_b}:
 *
 *     alpha_b = (theta_s - psi) + atan2(c*cos(theta_b), 1 + c*sin(theta_b)),  c = lam*r/rho
 *
 * which is STRICTLY POSITIVE for every |theta_b| < pi/2 and reaches zero only AT
 * +/- pi/2 - the realizability boundary, where the damping ratio cos(theta_b) is zero and
 * the pole magnitude needed diverges as 1/cos(theta_b). The integrator therefore has
 * exactly one thing it can do: ramp one way until it clips. Measured slope at the rail is
 * +0.92 (r=20) and +2.55 (r=30), so it is genuinely attracting - the law converges
 * robustly, to a useless place.
 *
 * Mode 2 fixes the structure rather than the gain, two ways:
 *
 *   EXOGENOUS REFERENCE. The cost is built from the LATERAL TRACKING direction - the angle
 *   between the commanded horizontal acceleration and the achieved one. The commanded
 *   direction comes from the position loop and does NOT move when theta_b moves, so it
 *   cannot self-cancel the way M_dyn.x does.
 *
 *   MEASURED GRADIENT, NOT ASSUMED SIGN. theta_b is dithered at MC_POLE_FDIT and the cost
 *   is demodulated against that dither, which ESTIMATES d(cost)/d(theta_b) including its
 *   sign. Nothing here assumes which way b should move; if the sensitivity reverses, the
 *   estimate reverses with it. That is the property mode 1 lacks and cannot be given by
 *   retuning.
 *
 * The premise is supported by measurement rather than hope: the archived static-b sweep
 * (examples/results/b_vs_tau) has INTERIOR cost minima at every lag where flight is
 * possible at all (best b = 16, 16, 4, 4 at tau = 0, 5, 10, 15 ms). Two honest limits go
 * with it. The bowl is SHALLOW - 0.7 to 3 percent cost spread across the whole b range -
 * so the dither-induced modulation is small and the gradient estimate needs real
 * averaging. And at tau >= 20 ms that sweep has NO stable b at all, so this tunes within
 * the region where flight already works; it does not widen it. The seat is what addresses
 * lag. This is a refinement on top.
 *
 * It also needs the vehicle to actually be translating: with no lateral command the
 * direction error is noise, so the update is gated on MC_POLE_AMIN. In a pure
 * position-hold hover it will correctly do nothing.
 *
 * The level-1 twin is PoleAdapter in examples/utils/seat.py in the PegasusSimulator
 * checkout. Keep the two in step.
 */
class PoleAdapter : public ModuleParams
{
public:
	/// MC_POLE_EN. Mode 1 is kept only so the broken law stays measurable against mode 2.
	enum class Mode : int32_t {
		Off = 0,
		Gradient = 1,		///< brief v3 section 4. Ill-formed; ramps to the rail.
		ExtremumSeek = 2,	///< dithered gradient on the lateral tracking cost.
	};

	explicit PoleAdapter(ModuleParams *parent);
	~PoleAdapter() override = default;

	/**
	 * Largest offset the angle is allowed to reach [rad].
	 *
	 * Shared with the seat's ceiling and for the same reason: holding a pole angle
	 * theta costs a pole magnitude growing as 1/cos(theta), so |theta| < pi/2 is a
	 * realizability limit rather than a safety margin.
	 */
	static constexpr float kThetaCeiling = 1.5f;

	/// Drop theta_b and the diagnostics. Called from EigenController::reset().
	void reset();

	/**
	 * Advance theta_b one tick.
	 *
	 * @param mdyn_x  M_dyn applied to the measured rate vector (p,q) [rad/s^2] - the
	 *                acceleration the DESIGN asks for, in the same units and frame as
	 *                @p xa.
	 * @param xa      achieved roll/pitch angular acceleration [rad/s^2], plant drift
	 *                already removed. This is the SAME vector the seat's observable
	 *                uses, deliberately: two adaptations reading one measurement.
	 * @param r       measured yaw rate [rad/s]. Gates the update via MC_POLE_RMIN.
	 * @param dt      [s] control timestep.
	 */
	void adapt(const matrix::Vector2f &mdyn_x, const matrix::Vector2f &xa, float r, float dt);

	/**
	 * Mode 2. Advance the dithered gradient descent one tick.
	 *
	 * @param a_cmd  commanded horizontal acceleration (NED xy) [m/s^2] - the EXOGENOUS
	 *               reference, from the position loop, independent of theta_b.
	 * @param a_meas achieved horizontal acceleration (NED xy) [m/s^2].
	 * @param r      measured yaw rate [rad/s]. Gates via MC_POLE_RMIN.
	 * @param dt     [s] control timestep.
	 */
	void adaptExtremum(const matrix::Vector2f &a_cmd, const matrix::Vector2f &a_meas,
			   float r, float dt);

	/// Cached MC_POLE_EN, so a caller can skip the work entirely when disabled.
	Mode mode() const { return _mode; }

	/// Converged estimate without the dither [rad] - what the tuning has actually found.
	float thetaBHat() const { return _theta_b_hat; }

	/// Last cost sample (|lateral direction error|, rad), NAN while gated. Mode 2 only.
	float cost() const { return _cost; }

	/// Offset actually APPLIED to the pole angle [rad] - includes mode 2's dither.
	/// Logged as mc_controller_status.pole_theta_b.
	float thetaB() const { return _theta_b; }

	/// Misalignment measured on the last adapting tick [rad], NAN while gated.
	float alphaB() const { return _alpha_b; }

	/// Whether adaptation is enabled at all, so a caller can skip the work.
	bool enabled() const { return _mode != Mode::Off; }

	/**
	 * Allocator saturation, which freezes the update - same argument as the seat's:
	 * once the mixer clips, the achieved acceleration stops following the designed one
	 * for reasons the pole angle did not cause and cannot fix.
	 */
	void setSaturated(bool saturated) { _saturated = saturated; }

protected:
	void updateParams() override;

private:
	float _theta_b{0.f};	///< [rad] offset APPLIED, = _theta_b_hat + dither in mode 2
	float _theta_b_hat{0.f};///< [rad] the converged estimate, dither excluded
	float _alpha_b{NAN};	///< [rad] mode 1's misalignment, diagnostics only
	float _cost{NAN};	///< [rad] mode 2's cost sample, diagnostics only
	bool _saturated{false};

	// Mode 2 internals.
	float _dither_phase{0.f};	///< [rad] wraps at 2pi, so it never grows without bound
	float _cost_lp{NAN};		///< slow average of the cost, the high-pass reference
	float _grad_lp{0.f};		///< demodulated gradient estimate
	matrix::Vector2f _a_cmd_lp{NAN, NAN};	///< [m/s^2] commanded accel, spin-averaged
	matrix::Vector2f _a_meas_lp{NAN, NAN};	///< [m/s^2] achieved accel, spin-averaged

	// Cached parameters, so nothing here touches the parameter system at gyro rate.
	Mode _mode{Mode::Off};
	float _k_b{0.f};		///< MC_POLE_K
	float _theta_b_max{0.f};	///< [rad] MC_POLE_MAX, itself ceilinged below pi/2
	float _r_min{0.f};		///< [rad/s] MC_POLE_RMIN
	float _dither_amp{0.f};		///< [rad] MC_POLE_DITH
	float _dither_w{0.f};		///< [rad/s] 2*pi*MC_POLE_FDIT
	float _k_es{0.f};		///< MC_POLE_KES
	float _a_min{0.f};		///< [m/s^2] MC_POLE_AMIN

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::MC_POLE_EN>)     _param_mc_pole_en,
		(ParamFloat<px4::params::MC_POLE_K>)    _param_mc_pole_k,
		(ParamFloat<px4::params::MC_POLE_MAX>)  _param_mc_pole_max,
		(ParamFloat<px4::params::MC_POLE_RMIN>) _param_mc_pole_rmin,
		(ParamFloat<px4::params::MC_POLE_DITH>) _param_mc_pole_dith,
		(ParamFloat<px4::params::MC_POLE_FDIT>) _param_mc_pole_fdit,
		(ParamFloat<px4::params::MC_POLE_KES>)  _param_mc_pole_kes,
		(ParamFloat<px4::params::MC_POLE_AMIN>) _param_mc_pole_amin
	)
};
