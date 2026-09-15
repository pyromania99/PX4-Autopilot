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

#include "PoleAdapter.hpp"

#include <float.h>
#include <mathlib/mathlib.h>

using namespace matrix;

namespace
{
/**
 * Smallest vector magnitude with a usable direction.
 *
 * Unlike the seat's CROSS law, this law drives on the ANGLE, which atan2 normalises the
 * magnitudes out of - so a near-zero pair returns a confident wrong answer rather than a
 * small one, and it has to be gated explicitly. Matches the level-1 twin's default eps.
 */
constexpr float kEpsAngle = 1e-9f;
} // namespace

PoleAdapter::PoleAdapter(ModuleParams *parent) :
	ModuleParams(parent)
{
	PoleAdapter::updateParams();
	PoleAdapter::reset();
}

void PoleAdapter::updateParams()
{
	ModuleParams::updateParams();

	_mode = static_cast<Mode>(_param_mc_pole_en.get());
	_k_b = math::max(_param_mc_pole_k.get(), 0.f);
	_dither_amp = math::max(_param_mc_pole_dith.get(), 0.f);
	_dither_w = 2.f * M_PI_F * math::max(_param_mc_pole_fdit.get(), 0.f);
	_k_es = math::max(_param_mc_pole_kes.get(), 0.f);
	_a_min = math::max(_param_mc_pole_amin.get(), 0.f);
	_theta_b_max = math::constrain(_param_mc_pole_max.get(), 0.f, kThetaCeiling);
	_r_min = math::max(_param_mc_pole_rmin.get(), 0.f);

	// A parameter change that shrinks MAX has to pull the live offset in with it, or
	// theta_b sits outside its own limit until the next adapting tick.
	_theta_b = math::constrain(_theta_b, -_theta_b_max, _theta_b_max);
	_theta_b_hat = math::constrain(_theta_b_hat, -_theta_b_max, _theta_b_max);
}

void PoleAdapter::reset()
{
	_theta_b = 0.f;
	_theta_b_hat = 0.f;
	_alpha_b = NAN;
	_cost = NAN;
	_dither_phase = 0.f;
	_cost_lp = NAN;
	_grad_lp = 0.f;
	_a_cmd_lp = matrix::Vector2f(NAN, NAN);
	_a_meas_lp = matrix::Vector2f(NAN, NAN);
}

void PoleAdapter::adapt(const Vector2f &mdyn_x, const Vector2f &xa, float r, float dt)
{
	_alpha_b = NAN;

	// Same gate as the seat, for the same reason: with no spin there is no frequency for
	// any of this geometry to be defined at, and a clipped mixer means the achieved
	// acceleration stopped following the designed one for reasons the pole cannot fix.
	if ((_mode != Mode::Gradient) || _saturated || (fabsf(r) <= _r_min) || !(dt > 0.f)
	    || !mdyn_x.isAllFinite() || !xa.isAllFinite()) {
		return;
	}

	const float cross = mdyn_x(0) * xa(1) - mdyn_x(1) * xa(0);
	const float dot = mdyn_x(0) * xa(0) + mdyn_x(1) * xa(1);

	// The ANGLE law has no self-suppression in a starved channel - atan2 divides the
	// magnitudes out - so the only protection is this explicit magnitude gate.
	if ((mdyn_x.norm() <= kEpsAngle) || (xa.norm() <= kEpsAngle)) {
		return;
	}

	const float alpha_b = atan2f(cross, dot);

	if (!PX4_ISFINITE(alpha_b)) {
		return;
	}

	_theta_b = math::constrain(_theta_b - _k_b * alpha_b * dt, -_theta_b_max, _theta_b_max);
	_alpha_b = alpha_b;
}

void PoleAdapter::adaptExtremum(const Vector2f &a_cmd, const Vector2f &a_meas, float r, float dt)
{
	_cost = NAN;

	if ((_mode != Mode::ExtremumSeek) || !(dt > 0.f)
	    || !a_cmd.isAllFinite() || !a_meas.isAllFinite()) {
		return;
	}

	/*
	 * The dither runs even while the cost is ungated, so the phase stays continuous and a
	 * gap in the excitation does not put a step into the applied angle. Wrapped rather than
	 * accumulated: a float phase growing for a long flight loses resolution exactly where
	 * the demodulation needs it.
	 */
	_dither_phase += _dither_w * dt;

	if (_dither_phase > 2.f * M_PI_F) {
		_dither_phase -= 2.f * M_PI_F;
	}

	const float dither = _dither_amp * sinf(_dither_phase);

	// Gates. Saturation and spin, as for mode 1 - plus the one this law needs and mode 1
	// did not: without a lateral command there is no direction to track, and the angle
	// between two noise vectors is noise. In position hold this correctly does nothing.
	/*
	 * SPIN-SYNCHRONOUS REJECTION, and this is not optional.
	 *
	 * Measured at level 2, r = 29.5 rad/s: the commanded horizontal acceleration averages
	 * 0.064 m/s^2 while the ACHIEVED one averages 4.40 m/s^2 - seventy times larger. That
	 * is not tracking; it is the spin. A small residual tilt on a body turning at 4.7 Hz
	 * sweeps the thrust vector around, putting ~g*tilt of rotating acceleration into the
	 * world frame at the spin frequency. An angle taken between the raw pair measures that
	 * rotation, not whether the vehicle is going where it was sent.
	 *
	 * Averaging both vectors over several spin periods separates them: the spin-synchronous
	 * part integrates to near zero over a revolution, the tracking response does not. The
	 * cost is therefore built from the DC parts, which is also the only part the position
	 * loop was ever commanding.
	 */
	const float spin_period = 2.f * M_PI_F / math::max(fabsf(r), 1e-3f);
	const float avg_alpha = math::constrain(dt / math::max(4.f * spin_period, 0.2f), 0.f, 1.f);

	if (!_a_cmd_lp.isAllFinite()) {
		_a_cmd_lp = a_cmd;
		_a_meas_lp = a_meas;
	}

	_a_cmd_lp += (a_cmd - _a_cmd_lp) * avg_alpha;
	_a_meas_lp += (a_meas - _a_meas_lp) * avg_alpha;

	// Gate on the AVERAGED magnitudes: the instantaneous achieved vector is dominated by
	// spin-synchronous content and would pass this gate on that alone.
	const bool excited = (_a_cmd_lp.norm() > _a_min) && (_a_meas_lp.norm() > _a_min);

	if (!_saturated && (fabsf(r) > _r_min) && excited) {
		// THE COST. Unsigned: extremum seeking descends a cost, it does not null a signed
		// error, so what matters is that this is >= 0 with a minimum where the achieved
		// direction matches the commanded one. Using |angle| rather than the signed angle
		// is what lets the demodulator find the minimum from either side.
		const float cross = _a_cmd_lp(0) * _a_meas_lp(1) - _a_cmd_lp(1) * _a_meas_lp(0);
		const float dot = _a_cmd_lp(0) * _a_meas_lp(0) + _a_cmd_lp(1) * _a_meas_lp(1);
		const float J = fabsf(atan2f(cross, dot));
		_cost = J;

		// Slow average = the high-pass reference. Seeded on first use so the estimator does
		// not spend its first seconds chasing a zero it was initialised to.
		const float lp_alpha = math::constrain(dt / 2.f, 0.f, 1.f);	// ~2 s average

		if (!PX4_ISFINITE(_cost_lp)) {
			_cost_lp = J;
		}

		_cost_lp += lp_alpha * (J - _cost_lp);

		/*
		 * DEMODULATION. Correlating the cost's AC part against the dither that caused it
		 * estimates d(cost)/d(theta_b) up to a positive scale: if the cost rises when the
		 * dither pushes theta_b up, the product is positive and theta_b must come down.
		 * The sign is MEASURED here, not assumed - which is the whole difference from
		 * mode 1, where a sign was derived and turned out to point at the rail.
		 */
		const float grad = (J - _cost_lp) * sinf(_dither_phase);
		const float g_alpha = math::constrain(dt / 1.f, 0.f, 1.f);	// ~1 s average
		_grad_lp += g_alpha * (grad - _grad_lp);

		_theta_b_hat = math::constrain(_theta_b_hat - _k_es * _grad_lp * dt,
					       -_theta_b_max, _theta_b_max);
	}

	// Applied angle carries the dither; the estimate does not. Clipped again because the
	// dither can push an already-limited estimate past the rail.
	_theta_b = math::constrain(_theta_b_hat + dither, -_theta_b_max, _theta_b_max);
}
