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

#include "EigenController.hpp"

#include <float.h>
#include <geo/geo.h>
#include <mathlib/mathlib.h>

using namespace matrix;

/// Tilt fallback when CommandFrontEnd hands over a NaN limit. Matches MPC_TILTMAX_AIR's
/// default of 45 deg.
static constexpr float kDefaultTiltLimit = M_PI_F / 4.f;

EigenController::EigenController(ModuleParams *parent) :
	MulticopterControllerBase(parent),
	_seat(this),
	_pole(this),
	_lead(this),
	_rate_limits(this),
	_trajectory_stage(this)
{
	EigenController::updateParams();
	EigenController::reset();
}

void EigenController::updateParams()
{
	ModuleParams::updateParams();

	// Cached rather than read per cycle: update() runs at gyro rate and must not touch
	// the parameter system.
	_pos_p = Vector3f(_param_mc_eig_xy_p.get(), _param_mc_eig_xy_p.get(), _param_mc_eig_z_p.get());
	_pos_d = Vector3f(_param_mc_eig_xy_d.get(), _param_mc_eig_xy_d.get(), _param_mc_eig_z_d.get());

	_att_p = _param_mc_eig_att_p.get();
	_att_d = _param_mc_eig_att_d.get();

	_wn = _param_mc_eig_wn.get();
	_b = _param_mc_eig_b.get();
	_alpha = _param_mc_eig_alpha.get();
	_beta = _param_mc_eig_beta.get();

	// Floored well above zero: a zero inertia is not a "disable this axis" request, it is
	// a nonsensical rigid body, and it would silently zero the torque on that axis while
	// also zeroing the gyroscopic compensation of the other two.
	_inertia = Vector3f(math::max(_param_mc_eig_ixx.get(), 1e-4f),
			    math::max(_param_mc_eig_iyy.get(), 1e-4f),
			    math::max(_param_mc_eig_izz.get(), 1e-4f));

	_inv_torque_max = 1.f / math::max(_param_mc_eig_trq_max.get(), 1e-3f);

	// Effective pair, seeded at the nominal. update() rotates it by the adaptive offset
	// every tick when MC_POLE_EN is set; with the adapter off these stay equal to the
	// configured (wn, b), so the law is bit-identical to one built without this feature.
	_wn_eff = _wn;
	_b_eff = _b;
}

void EigenController::reset()
{
	// No integrators and no filters. What has to be dropped is the cached stage output
	// and, crucially, the angle-error history: a stale error from the previous mode
	// differentiates into exactly the spike _first_attitude_update exists to prevent.
	_roll_setpoint = 0.f;
	_pitch_setpoint = 0.f;
	_attitude_setpoint = Quatf();
	_acceleration_setpoint.setZero();
	_thrust_setpoint.setZero();
	_rate_setpoint.setZero();
	_torque.setZero();

	_roll_error_prev = 0.f;
	_pitch_error_prev = 0.f;
	_roll_error_rate = 0.f;
	_pitch_error_rate = 0.f;
	_first_attitude_update = true;
	_pole.reset();

	_attitude_error.setZero();

	// Drops the shared stage's integrator along with its cached setpoints - reset() is the
	// full clear, unlike the reset_integrals path in update().
	_trajectory_stage.reset();
	// A converged angle must not survive an arm transition or a level change: it
	// would rotate the very first torque of the next flight by an angle nothing has
	// measured.
	_seat.reset();
	_lead.reset();
}

void EigenController::stepTrajectoryToAcceleration(const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command)
{
	// Stage 1 and the lateral clamp live in the shared trajectory stage: they were
	// identical across every law in this module, and that is now where the position
	// integrator lives too. The gains stay here - MC_EIG_* is this law's own tuning.
	//
	// This function deliberately stops short of the tilt setpoint. Everything it computes is
	// a function of the position estimate and is therefore only worth recomputing at
	// position rate; resolving that acceleration against the heading is not, and update()
	// re-runs resolveAccelerationToTilt() every cycle.
	_acceleration_setpoint = _trajectory_stage.computeAccelerationSetpoint(state, command, _pos_p, _pos_d);

	// Stage 2, magnitude: shared, and a deliberate behaviour change. This law used to
	// divide only the hover term by cos(tilt) - a faithful port of the prototype's
	// base_thrust = m*g/(4*tilt_den), which leaves the altitude correction undivided and so
	// under-commands thrust in a tilted climb. The shared stage divides the whole demand,
	// which is the correct derivation. The two agree exactly at level and whenever
	// acceleration_sp(2) is zero; they diverge by ~2% at 30 deg of bank while climbing.
	//
	// It also drops the fabsf()-and-floor on cos(tilt), so there is no longer a capped 10x
	// collective boost past ~84 deg of tilt or while inverted. See TrajectoryStage.hpp.
	_thrust_setpoint = _trajectory_stage.computeThrustSetpoint(state, command, _acceleration_setpoint);
}

void EigenController::resolveAccelerationToTilt(const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command)
{
	const Vector3f acceleration_sp = _acceleration_setpoint;

	// Needed again below for the second, angle-domain clamp the small-angle inversion
	// makes necessary. The stage has already applied the acceleration-domain one.
	const float tilt_limit = PX4_ISFINITE(command.tilt_limit) ? command.tilt_limit : kDefaultTiltLimit;

	// Stage 2, direction. The prototype inverts the small-angle hover relation in ENU/FLU;
	// re-derived here for NED/FRD, where the third column of R(phi, theta, psi) is
	// approximately [theta*cos(psi) + phi*sin(psi), theta*sin(psi) - phi*cos(psi), 1] and
	// horizontal acceleration is -g times its first two entries:
	//
	//   theta_des = -(a_n*cos(psi) + a_e*sin(psi)) / g
	//   phi_des   =  (a_e*cos(psi) - a_n*sin(psi)) / g
	//
	// Sanity: a pure north demand at psi = 0 gives nose-down pitch and zero roll; a pure
	// east demand gives right-wing-down roll and zero pitch. Both are asserted by test,
	// because a sign error here is a controller that flies away from its setpoint.
	//
	// THE HEADING DECISION, same as MC_CTRL_ALG=1: the vehicle's own heading, not
	// command.yaw_sp. The tilt is rebuilt around wherever the vehicle points right now, so a
	// heading error can never appear and the yaw axis never sees a proportional term.
	//
	// Taken from state.q via the shared helper, not from state.heading. This resolution is
	// the whole reason the function runs at gyro rate: the sin/cos pair below IS the
	// rotation from NED into the heading-aligned frame, so a heading that lags by
	// r*dt_position rotates the commanded roll/pitch pair by the same angle, and
	// angleToRateSetpoint() then reads that rotation as a tilt error to correct. In a spin
	// it is a standing, rate-proportional cross-coupling between the roll and pitch
	// channels. See TrajectoryStage::currentHeading().
	const float heading = TrajectoryStage::currentHeading(state);
	const float sin_heading = sinf(heading);
	const float cos_heading = cosf(heading);

	float pitch_sp = -(acceleration_sp(0) * cos_heading + acceleration_sp(1) * sin_heading) / CONSTANTS_ONE_G;
	float roll_sp = (acceleration_sp(1) * cos_heading - acceleration_sp(0) * sin_heading) / CONSTANTS_ONE_G;

	// The linear inversion overshoots the true angle: clamping lateral acceleration to
	// g*tan(tilt) above leaves it free to ask for tan(tilt) RADIANS, which at a 45 deg
	// limit is 57 deg. Clamp the tilt magnitude too, so the framework's limit is what
	// actually reaches the vehicle.
	const float tilt_magnitude = sqrtf(roll_sp * roll_sp + pitch_sp * pitch_sp);

	if (tilt_magnitude > tilt_limit && tilt_magnitude > FLT_EPSILON) {
		const float scale = tilt_limit / tilt_magnitude;
		roll_sp *= scale;
		pitch_sp *= scale;
	}

	_roll_setpoint = roll_sp;
	_pitch_setpoint = pitch_sp;
	_attitude_setpoint = Quatf(Eulerf(roll_sp, pitch_sp, heading));
}

Vector3f EigenController::angleToRateSetpoint(const mc_ctrl::ControllerState &state, const float yaw_rate_setpoint)
{
	const Eulerf euler(state.q);
	const float roll_error = wrap_pi(_roll_setpoint - euler.phi());
	const float pitch_error = wrap_pi(_pitch_setpoint - euler.theta());

	// The derivative is refreshed only on a new attitude sample and held in between. At
	// gyro rate the attitude has usually not changed since the last call, so
	// differentiating every cycle would divide estimator quantization by a very small dt
	// and inject that straight into the rate setpoint.
	if (_first_attitude_update) {
		// Seed from the current error so the first step produces a zero derivative
		// rather than (error - 0) / dt. The prototype's _first_update guard.
		_roll_error_prev = roll_error;
		_pitch_error_prev = pitch_error;
		_roll_error_rate = 0.f;
		_pitch_error_rate = 0.f;
		_first_attitude_update = false;

	} else if (state.freshness.attitude_new && state.freshness.dt_attitude > FLT_EPSILON) {
		const float inv_dt = 1.f / state.freshness.dt_attitude;
		_roll_error_rate = (roll_error - _roll_error_prev) * inv_dt;
		_pitch_error_rate = (pitch_error - _pitch_error_prev) * inv_dt;
		_roll_error_prev = roll_error;
		_pitch_error_prev = pitch_error;
	}

	_attitude_error = Vector2f(roll_error, pitch_error);

	Vector3f rate_setpoint(_att_p * roll_error + _att_d * _roll_error_rate,
			       _att_p * pitch_error + _att_d * _pitch_error_rate,
			       0.f);

	/*
	 * THE YAW FEED-FORWARD IS A WORLD-Z ROTATION, not a body-z one. The pilot's yaw stick
	 * asks the vehicle to turn about the vertical, and this rate setpoint is expressed in
	 * body FRD, so the two coincide only while level. Dropping the command into rate(2) -
	 * as this did - means that at bank the vehicle turns about its own yaw axis instead,
	 * which is a different rotation: the missing part shows up as the roll and pitch rate
	 * the turn actually needs, and the angle PD is left to discover it as an error after
	 * the fact. The world z-axis expressed in the body frame is the last column of
	 * R.transposed(), i.e. q.inversed().dcm_z(); at MPC_MAN_TILT_MAX's 35 deg default the
	 * correction is ~57% of the commanded rate, so this is not a small-angle nicety.
	 * Same construction as stock (AttitudeControl.cpp:99-107).
	 */
	if (PX4_ISFINITE(yaw_rate_setpoint) && (fabsf(yaw_rate_setpoint) > FLT_EPSILON)) {
		rate_setpoint += state.q.inversed().dcm_z() * yaw_rate_setpoint;
	}

	// Last, as stock does it: the ceiling applies to the whole demand including the
	// feed-forward, not to the angle term alone.
	return _rate_limits.apply(rate_setpoint);
}

Vector3f EigenController::eigenTorque(const Vector3f &rate_setpoint, const Vector3f &angular_velocity) const
{
	// Stage 3, the eigen-dynamics inner loop. The prototype writes this as
	//
	//   corrections = B @ (M - A_r) @ [p, q, r, 1]'
	//
	// with B = diag(I), M = [M_dyn | M_des], M_dyn placing the eigenvalues and A_r the
	// assumed plant. That is feedback linearization: torque = I * (desired omega_dot -
	// assumed omega_dot). Expanded, the 3x4 product is the handful of terms below, which
	// is what runs here - a matrix multiply at gyro rate to reach six useful numbers is
	// not a trade worth making.
	const float p = angular_velocity(0);
	const float q = angular_velocity(1);
	const float r = angular_velocity(2);

	const float roll_rate_error = rate_setpoint(0) - p;
	const float pitch_rate_error = rate_setpoint(1) - q;

	// Desired angular acceleration. The cross terms (+b, -b) are the imaginary part of the
	// eigenvalue pair -wn +/- j*b: this is what couples the two axes into one rotating
	// mode. The +alpha terms cancel the rate damping the airframe is assumed to have.
	//
	// _wn_eff / _b_eff, not _wn / _b: the adaptive pole angle (MC_POLE_*) acts by
	// rotating this pair in polar form, and with the adapter disabled they are the
	// configured values exactly.
	const float roll_accel_des = _wn_eff * roll_rate_error + _b_eff * pitch_rate_error + _alpha * p;
	const float pitch_accel_des = _wn_eff * pitch_rate_error - _b_eff * roll_rate_error + _alpha * q;

	// Yaw carries no feedback term. In the prototype's algebra M_dyn(2,2) = -beta and
	// A_r(2,2) = -beta cancel exactly, leaving pure feedforward - cancelling the assumed
	// yaw drag and imposing the same eigenvalue is a no-op on the measured rate. So a zero
	// yaw rate setpoint commands zero yaw torque, which is the free-running heading the
	// design wants, and it is a property of the law rather than a special case bolted on.
	const float yaw_accel_des = _beta * rate_setpoint(2);

	// torque = I * omega_dot_des + omega x (I * omega). The gyroscopic term is derived
	// from the three inertias rather than hardcoded, and this is the one place the port
	// deliberately corrects the prototype instead of reproducing it.
	//
	// The prototype's A_r fixes the coupling at -Ixx*q*r on roll and +Iyy*p*r on pitch,
	// with nothing at all on yaw. Matching omega x (I*omega) would need Ixx == Iyy AND
	// Izz == 0 simultaneously, which no rigid body satisfies. For an ordinary quadrotor,
	// where the rotors lie in a plane and Izz is about 2*Ixx, the true roll term is
	// +Ixx*q*r - the prototype's has the OPPOSITE SIGN, so it adds to the coupling it was
	// meant to cancel. The error is second order in body rate, which is why it stayed
	// invisible in gentle hover sweeps and would not have stayed invisible under the spin
	// that follows a rotor failure.
	//
	// EigenControllerTest pins this against a direct omega x (I*omega) evaluation.
	Vector3f torque;
	torque(0) = _inertia(0) * roll_accel_des + (_inertia(2) - _inertia(1)) * q * r;
	torque(1) = _inertia(1) * pitch_accel_des + (_inertia(0) - _inertia(2)) * p * r;
	torque(2) = _inertia(2) * yaw_accel_des + (_inertia(1) - _inertia(0)) * p * q;

	return torque;
}

bool EigenController::update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
			     float dt, mc_ctrl::ControllerOutput &output)
{
	// HARD CONTRACT from the interface. The shared stage's position integrator is the one
	// thing here that genuinely winds up, and zeroing it on the ground is what keeps the
	// vehicle from leaping at takeoff. The cached stage outputs and the angle-error history
	// go too, so a mode change cannot carry either across.
	if (command.reset_integrals || !state.armed) {
		_rate_setpoint.setZero();
		_trajectory_stage.resetIntegral();
		_trajectory_stage.invalidateStage();
		_first_attitude_update = true;
	}

	Vector3f rate_setpoint{};

	switch (command.level) {
	case mc_ctrl::ControlLevel::Trajectory: {
			// Gated on a fresh local position sample so the position stage keeps
			// mc_pos_control's cadence instead of being pulled up to gyro rate.
			if (state.freshness.position_new || !_trajectory_stage.stageValid()) {
				stepTrajectoryToAcceleration(state, command);
			}

			// Resolved against the heading EVERY cycle, deliberately outside the gate
			// above. The acceleration demand is a position-stage quantity and is
			// correctly held between position samples; the heading it is resolved
			// against is not. Holding both would freeze the roll/pitch pair in a frame
			// the vehicle has since rotated out of, and the angle PD would then chase
			// that frozen frame. Two trig calls at gyro rate.
			resolveAccelerationToTilt(state, command);

			// Yaw rate setpoint is zero: heading is not controlled at this level. See
			// the file comment.
			rate_setpoint = angleToRateSetpoint(state, 0.f);
			break;
		}

	case mc_ctrl::ControlLevel::Attitude: {
			// Stabilized/Manual. Take the pilot's tilt through the same Euler angle PD
			// the trajectory stage feeds, and let the yaw stick reach the feedforward
			// term - unlike MC_CTRL_ALG=1, yaw rate commands do work here, because the
			// law has somewhere to put them.
			const Eulerf euler_sp(command.attitude_sp);
			_roll_setpoint = euler_sp.phi();
			_pitch_setpoint = euler_sp.theta();
			_attitude_setpoint = command.attitude_sp;
			_thrust_setpoint = command.thrust_body_sp;

			rate_setpoint = angleToRateSetpoint(state, command.yaw_sp_move_rate);
			_trajectory_stage.invalidateStage();
			break;
		}

	case mc_ctrl::ControlLevel::BodyRate:
		// Acro. No angle stage to run, so the commanded rates go straight into the
		// eigen inner loop and MC_EIG_WN / MC_EIG_B do double duty as rate gains.
		rate_setpoint = command.rate_sp;
		_thrust_setpoint = command.thrust_body_sp;
		_attitude_setpoint = Quatf(NAN, NAN, NAN, NAN);
		_attitude_error.setZero();
		_trajectory_stage.invalidateStage();
		// The angle PD is not running, so its error history is stale the moment we
		// return to a level that uses it.
		_first_attitude_update = true;
		break;

	case mc_ctrl::ControlLevel::None:
	default:
		return false;
	}

	_rate_setpoint = rate_setpoint;

	/*
	 * ADAPTIVE POLE ANGLE (MC_POLE_*), applied BEFORE the block is built from the pair.
	 *
	 * In polar form (wn, b) IS the pole: M_dyn = -rho*R(theta) with rho = hypot(wn, b)
	 * and theta = atan2(b, wn), placing the pair at -rho*e^{+-j theta} and making the
	 * damping ratio cos(theta). Rotating the pair therefore slides the pole around its
	 * own circle of constant magnitude - it retunes the damping without changing how hard
	 * the block pulls, which is the whole reason the level-1 twin was moved to this
	 * parameterisation. theta_b is an OFFSET from the configured angle, so with the
	 * adapter disabled these stay exactly the configured pair (set in updateParams).
	 */
	if (_pole.enabled()) {
		const float rho = sqrtf(_wn * _wn + _b * _b);
		const float theta = atan2f(_b, _wn) + _pole.thetaB();
		_wn_eff = rho * cosf(theta);
		_b_eff = rho * sinf(theta);
	}

	_torque = eigenTorque(rate_setpoint, state.angular_velocity);

	/*
	 * THE SEAT (MC_SEAT_*). Applied to the PHYSICAL torque, before MC_EIG_TRQ_MAX
	 * normalises it, so it operates in the same units as the level-1 twin.
	 *
	 * xd = tau/I, UNMODIFIED - deliberately not "corrected" to strip the gyroscopic
	 * feedforward back out. That correction (tried 2026-09-11) breaks a lag-invariance
	 * identity this pairing has and the stripped one does not:
	 *
	 *   Let g = (Izz-Iyy)/Ixx * q*r (the feedforward eigenTorque() folds into tau) and
	 *   drift = -alpha*p - g, both evaluated at the CURRENT tick, no lag anywhere in
	 *   either. With actuator delay tau_lag, Euler's equation (real q,r, real physics)
	 *   gives measured pdot(t) = accel_des(t-tau_lag) + g(t-tau_lag) - g(t). Substitute:
	 *
	 *     xa(t) = pdot_meas(t) - drift(t) = accel_des(t-tau_lag) + g(t-tau_lag) + alpha*p(t)
	 *     xd(t-tau_lag) = tau(t-tau_lag)/I = accel_des(t-tau_lag) + g(t-tau_lag)
	 *     xa(t) - xd(t-tau_lag) = alpha*p(t)
	 *
	 *   which is EXACTLY ZERO at alpha=0 (this law's every tested configuration),
	 *   for ANY lag, ANY q,r trajectory - g cancels completely regardless of how much
	 *   it moved during the delay. That property is what makes "the gap between xd and
	 *   xa is attributable to lag alone" true. Stripping g from xd and dropping the
	 *   drift subtraction from xa (the 2026-09-11 attempt) does NOT have this property:
	 *   redo the same substitution and a residual g(t-tau_lag)-g(t) survives, which does
	 *   not vanish (measured ~0.37-0.95 rad/s^2 in real telemetry, the same order as the
	 *   roll/pitch command itself) - a self-inflicted confound with no counterpart at
	 *   level 1. Reverted back to the original, level-1-identical pairing.
	 *
	 * xa is the measured acceleration with the SAME assumed coupling/damping (drift)
	 * removed, evaluated fresh from CURRENT p,q,r every tick - not the feedforward's own
	 * (possibly stale) internal value. PX4 supplies angular_accel directly
	 * (vehicle_angular_velocity.xyz_derivative), so unlike level 1 there is no rate
	 * difference and no sample alignment to get wrong - but note it is a backward
	 * difference through a 1-pole AlphaFilter (time constant 1/(2*pi*f_c), NOT the 2-pole
	 * LowPassFilter2p) at IMU_DGYRO_CUTOFF, flown here at 50 Hz, so it carries a phase of
	 * its own that the seat will read as part of the lag. The order matters for the phase
	 * arithmetic: 3.18 ms at 50 Hz 1-pole against 4.50 ms if it were 2-pole.
	 *
	 * And note what pairs with it: xd above is built from the COMMANDED torque, which is
	 * internal and unfiltered. So xa/xd = H_filter * Ga * e^{i theta_s} - the measurement
	 * filter's phase is INSIDE alpha, and the seat already cancels it. There is no separate
	 * sensor feed-forward to build.
	 */
	/*
	 * UNCONDITIONAL since 2026-09-17. This block used to be skipped entirely when the
	 * seat was off and the pole disabled, which meant the misalignment alpha - the
	 * quantity the seat exists to drive to zero - was never computed on the seat-OFF
	 * arm. Every seat figure in experiments/ therefore had an empty OFF trace, and no
	 * way to tell whether the seat reduced a misalignment that was there to begin with.
	 *
	 * Nothing in here applies anything: xd, xa and the alpha below are measurements,
	 * and the two things that DO act - the pole adapter and _seat.apply() - keep their
	 * own guards below. The cost is a handful of flops per tick on an arm that used to
	 * skip them.
	 */
	{
		const float p = state.angular_velocity(0);
		const float q = state.angular_velocity(1);
		const float r = state.angular_velocity(2);

		const Vector2f xd(_torque(0) / _inertia(0), _torque(1) / _inertia(1));

		const Vector2f drift(-_alpha * p + (_inertia(1) - _inertia(2)) / _inertia(0) * r * q,
				     -_alpha * q + (_inertia(2) - _inertia(0)) / _inertia(1) * r * p);
		const Vector2f xa(state.angular_accel(0) - drift(0),
				  state.angular_accel(1) - drift(1));

		_seat_saturated = _seat.saturated() ? 1.f : 0.f;
		_seat_xd_norm = xd.norm();
		_seat_xa_norm = xa.norm();
		// The MEASUREMENT, on every arm. Not _seat.alpha(), which is written only on a
		// tick that adapts and so is NAN for a whole seat-OFF or seat-FIXED flight.
		_seat_alpha_meas = Seat::measure(xd, xa);

		/*
		 * The pole adapter reads the SAME achieved acceleration the seat does, but a
		 * different reference. The seat's xd is "what the wrench I just sent should
		 * produce"; this one is M_dyn applied to the measured rate vector - "what the
		 * DESIGN asks of where the vehicle actually is". Expanded from the block
		 * [[-wn, -b], [b, -wn]] acting on (p, q), using the effective pair so the
		 * observable is read against the design currently in force rather than the
		 * nominal one.
		 *
		 * Ordered before the seat so both see the same pre-rotation torque, and it
		 * applies no rotation of its own - theta_b takes effect through the block on
		 * the NEXT tick, exactly as at level 1.
		 */
		if (_pole.mode() == PoleAdapter::Mode::Gradient) {
			const Vector2f mdyn_x(-(_wn_eff * p + _b_eff * q),
					      -(_wn_eff * q - _b_eff * p));
			_pole.adapt(mdyn_x, xa, r, dt);

		} else if (_pole.mode() == PoleAdapter::Mode::ExtremumSeek) {
			/*
			 * Mode 2's pair is TRANSLATIONAL, not rotational, and that is the point: the
			 * commanded horizontal acceleration comes from the position loop, so unlike
			 * mode 1's M_dyn.x reference it does not move when theta_b moves and cannot
			 * cancel itself out of the cost. state.acceleration is the filtered derivative
			 * of velocity (MPC_VELD_LP), which is the only achieved-acceleration signal
			 * the framework offers - its filter lag is common to both vectors' directions
			 * only approximately, which is a further reason this law descends an averaged
			 * cost rather than trusting any single sample.
			 */
			const Vector2f a_cmd(_acceleration_setpoint(0), _acceleration_setpoint(1));
			const Vector2f a_meas(state.acceleration(0), state.acceleration(1));
			_pole.adaptExtremum(a_cmd, a_meas, r, dt);
		}

		if (_seat.mode() != Seat::Mode::Off) {
			_torque = _seat.apply(_torque, xd, xa, r, dt);
			// 1 = the law stepped this tick, 0 = it was gated (saturated, below
			// MC_SEAT_RMIN, or an unmeasurable pair). NAN = no law running.
			_seat_stepped = PX4_ISFINITE(_seat.alpha()) ? 1.f : 0.f;
		}
	}

	// LAST, after the seat: the actuator receives lead(seat(tau)) and therefore delivers
	// e^{-sL} * seat(tau), i.e. a pure delay. The seat's observable is unaffected - it
	// forms xd from the PRE-seat torque and xa from the measurement, so it simply sees a
	// faster actuator and adapts to the smaller residual. See ActuatorLead.hpp.
	_torque = _lead.apply(_torque, dt);

	// N m -> normalized. control_allocator normalizes its own mix columns, so what it
	// wants here is dimensionless and physical torque would be silently misinterpreted.
	Vector3f torque = _torque * _inv_torque_max;

	for (int i = 0; i < 3; i++) {
		torque(i) = PX4_ISFINITE(torque(i)) ? math::constrain(torque(i), -1.f, 1.f) : 0.f;
	}

	for (int i = 0; i < 3; i++) {
		if (!PX4_ISFINITE(_thrust_setpoint(i))) {
			_thrust_setpoint(i) = 0.f;
		}
	}

	output.torque = torque;
	output.thrust = _thrust_setpoint;
	output.rate_setpoint = _rate_setpoint;
	output.attitude_setpoint = _attitude_setpoint;
	output.valid = true;

	(void)dt;	// the only rate-dependent term uses dt_attitude, not the gyro dt
	return true;
}

void EigenController::fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const
{
	// Telemetry only. Stock fills this from PositionControl's internal setpoints; here it
	// comes from what the PD stage actually used, so a log still shows what was asked for
	// versus what was flown. The attitude and thrust setpoints are this law's own, so the
	// stage cannot source them itself.
	_trajectory_stage.fillLocalPositionSetpoint(sp, _attitude_setpoint, _thrust_setpoint);
}

void EigenController::setAllocatorFeedback(const mc_ctrl::AllocatorFeedback &feedback)
{
	// Roll and pitch only: the seat does not rotate yaw, so a yaw-only clip is none of
	// its business. Freezing matters because once the mixer saturates the achieved
	// direction stops following the commanded one for reasons unrelated to lag.
	const bool rp_saturated = feedback.saturation_positive(0) || feedback.saturation_negative(0)
				  || feedback.saturation_positive(1) || feedback.saturation_negative(1);
	_seat.setSaturated(rp_saturated);
	_pole.setSaturated(rp_saturated);
}

void EigenController::fillStatus(mc_controller_status_s &status) const
{
	status.debug[0] = _attitude_error(0);
	status.debug[1] = _attitude_error(1);
	status.debug[2] = _rate_setpoint(0);
	status.debug[3] = _rate_setpoint(1);
	status.debug[4] = _rate_setpoint(2);
	// Seat diagnostics. The physical torque that used to sit in [5] and [6] is exactly
	// recoverable as torque_sp * MC_EIG_TRQ_MAX, so nothing is lost by reusing them, and
	// what the seat does is not recoverable from the log at all without these: whether
	// the allocator gate froze the update, and the two magnitudes that decide both the
	// CROSS drive and whether the pair is measurable in the first place.
	status.debug[5] = _seat_saturated;
	status.debug[6] = _seat_xd_norm;
	status.debug[7] = _seat_xa_norm;
	status.seat_theta = _seat.theta();
	status.seat_alpha = _seat_alpha_meas;
	status.seat_stepped = _seat_stepped;
	status.pole_theta_b = _pole.thetaB();
	status.pole_alpha_b = _pole.alphaB();
	status.pole_theta_hat = _pole.thetaBHat();
	status.pole_cost = _pole.cost();
}
