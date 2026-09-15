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
 * @file tilt_eigen_params.c
 * Gains and rigid-body properties for the tilt-axis eigen controller (MC_CTRL_ALG=4).
 *
 * This is the same control law as MC_CTRL_ALG=3 with the Euler attitude chart replaced by
 * a reduced-attitude (tilt-axis) error, so the parameter set is deliberately a mirror of
 * MC_EIG_* with the SAME DEFAULTS and the same meanings. A tune developed on MC_CTRL_ALG=3
 * transfers by copying the numbers across.
 *
 * There are two differences from MC_EIG_*, both consequences of the change:
 *
 *   - There is no MC_TEIG_ATT_D. The tilt error is a body-frame vector and the body frame
 *     rotates, so numerically differencing it injects a spurious omega_z x e term that is
 *     harmless at hover and dominant under a rotor-failure spin. MC_TEIG_WN carries the
 *     damping instead. Nothing in this controller depends on dt as a result.
 *
 *   - MC_TEIG_ATT_P is exact at large angles where MC_EIG_ATT_P's chart is not, but the two
 *     agree to first order, so the tuned value carries over unchanged.
 *
 * Like MC_CTRL_ALG=3 this controller works in PHYSICAL units: its inner loop multiplies a
 * desired angular acceleration by the inertia tensor, so it needs real kg m^2 and converts
 * the resulting N m to the normalized torque control_allocator expects via MC_TEIG_TRQ_MAX.
 *
 * Tilt limit, thrust limits and velocity limits still come from the framework's
 * MPC_TILTMAX_*, MPC_THR_* and MPC_*_VEL_MAX; mass and gravity are replaced by the hover
 * thrust estimate. There are no integrator gains because the law has no integrators.
 */

/**
 * Tilt-eigen controller horizontal position gain
 *
 * Desired horizontal acceleration per metre of horizontal position error.
 *
 * Same stage and same units as MC_EIG_XY_P; the default matches it.
 *
 * @min 0.0
 * @max 5.0
 * @decimal 2
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_XY_P, 0.50f);

/**
 * Tilt-eigen controller horizontal velocity gain
 *
 * Desired horizontal acceleration per m/s of horizontal velocity error. This is the
 * damping term: it acts on the filtered estimator velocity, not on a numerical derivative
 * of the position error.
 *
 * Same stage and same units as MC_EIG_XY_D; the default matches it.
 *
 * @min 0.0
 * @max 10.0
 * @decimal 2
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_XY_D, 1.00f);

/**
 * Tilt-eigen controller vertical position gain
 *
 * Desired vertical acceleration per metre of altitude error.
 *
 * Same stage and same units as MC_EIG_Z_P; the default matches it, including its
 * inheritance of the Isaac Sim prototype's stiffness. That stiffness saturates the
 * collective on any sizeable altitude step - expect to walk this down in SITL, as
 * MC_PD_Z_P next door already has.
 *
 * @min 0.0
 * @max 30.0
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_Z_P, 10.0f);

/**
 * Tilt-eigen controller vertical velocity gain
 *
 * Desired vertical acceleration per m/s of climb rate error.
 *
 * Same stage and same units as MC_EIG_Z_D; the default matches it.
 *
 * @min 0.0
 * @max 20.0
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_Z_D, 5.3f);

/**
 * Tilt-eigen controller attitude gain
 *
 * Desired body rate per radian of TILT error - the angle between the vehicle's thrust axis
 * and the demanded one, about the axis that closes it. Roll and pitch share one gain
 * because the error is a single rotation, not two angles.
 *
 * This is the direct counterpart of MC_EIG_ATT_P and carries its tuned value over
 * unchanged: to first order the tilt error vector IS (roll error, pitch error), same sign
 * and same magnitude. The difference is at large angles, where this one stays exact and
 * MC_EIG_ATT_P's Euler decomposition does not.
 *
 * It feeds the desired rates the eigen-dynamics inner loop tracks; it is not a torque gain.
 *
 * @min 0.0
 * @max 30.0
 * @unit 1/s
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_ATT_P, 7.0f);

/**
 * Tilt-eigen controller rate bandwidth (eigenvalue real part)
 *
 * Own-axis body rate error gain: the -wn in the complex-conjugate eigenvalue pair
 * -wn +/- j*MC_TEIG_B that this law places on the roll/pitch subsystem. Larger is a
 * stiffer, faster rate loop.
 *
 * This carries MORE of the damping load than MC_EIG_WN does, because there is no
 * MC_TEIG_ATT_D stacked on top of it. If a tune ported from MC_CTRL_ALG=3 feels soft in
 * attitude, raise this rather than looking for the missing derivative gain.
 *
 * @min 0.0
 * @max 60.0
 * @unit rad/s
 * @decimal 2
 * @increment 0.5
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_WN, 18.0f);

/**
 * Tilt-eigen controller rate cross-coupling (eigenvalue imaginary part)
 *
 * Cross-axis body rate error gain: roll rate error produces pitch torque and vice versa,
 * with opposite signs. This is what makes the law an "eigen" controller rather than two
 * independent axes - roll and pitch are stabilized as one rotating mode, which is the
 * coning mode a spinning or asymmetric vehicle actually exhibits.
 *
 * Identical in meaning and value to MC_EIG_B. The cross-coupling is a rotation in the body
 * x-y plane and the tilt error is a vector in that same plane, which is why changing the
 * attitude chart left this stage untouched.
 *
 * Set to 0 to decouple the axes and recover a conventional rate law.
 *
 * @min 0.0
 * @max 60.0
 * @unit rad/s
 * @decimal 2
 * @increment 0.5
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_B, 10.2f);

/**
 * Tilt-eigen controller assumed roll/pitch rate damping
 *
 * The aerodynamic rate damping the vehicle is assumed to have, which the feedback
 * linearization cancels. It therefore enters the torque as +alpha*rate: it ADDS energy,
 * on the theory that the airframe is removing the same amount.
 *
 * Overestimating it is destabilizing. Keep it well below MC_TEIG_WN.
 *
 * @min 0.0
 * @max 5.0
 * @unit 1/s
 * @decimal 3
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_ALPHA, 0.5f);

/**
 * Tilt-eigen controller yaw rate bandwidth
 *
 * Assumed yaw aerodynamic drag, and simultaneously the gain on the commanded yaw rate.
 * In this law the two are the same number: cancelling the assumed drag and imposing the
 * same eigenvalue leaves the yaw feedback term exactly zero, so what survives is a pure
 * feedforward Izz*beta*yaw_rate_setpoint.
 *
 * Heading is never held. Here that is structural rather than arranged: the reduced-attitude
 * error is an axis in the body x-y plane and has no yaw component to begin with, so unlike
 * MC_CTRL_ALG=3 there is no heading injected into the setpoint and differenced back out.
 * At Trajectory level the yaw rate setpoint is always zero, so yaw torque there is nothing
 * but the gyroscopic compensation term, which vanishes for a symmetric airframe.
 *
 * Set to 0 to remove yaw actuation entirely, including in Acro and Stabilized.
 *
 * @min 0.0
 * @max 10.0
 * @unit 1/s
 * @decimal 3
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_BETA, 0.6f);

/**
 * Tilt-eigen controller roll inertia
 *
 * Vehicle moment of inertia about the body x axis, including the rotors' parallel-axis
 * contribution - not just the centre body. Scales roll torque directly and enters the
 * gyroscopic coupling of the other two axes.
 *
 * The default is the Isaac Sim prototype's quadrotor, matching MC_EIG_IXX. Measure or
 * estimate this for your airframe; a bifilar pendulum test is the usual method.
 *
 * @min 0.0001
 * @max 5.0
 * @unit kg m^2
 * @decimal 5
 * @increment 0.001
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_IXX, 0.01f);

/**
 * Tilt-eigen controller pitch inertia
 *
 * Vehicle moment of inertia about the body y axis. See MC_TEIG_IXX.
 *
 * @min 0.0001
 * @max 5.0
 * @unit kg m^2
 * @decimal 5
 * @increment 0.001
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_IYY, 0.01f);

/**
 * Tilt-eigen controller yaw inertia
 *
 * Vehicle moment of inertia about the body z axis.
 *
 * This does more than scale yaw torque, which is why it matters even though heading is
 * not controlled: it appears in the gyroscopic term omega x (I*omega) of the ROLL and
 * PITCH axes, as (Izz - Iyy)*q*r and (Ixx - Izz)*p*r. Those are the coupling a vehicle
 * feels while yawing, and for a planar quadrotor with Izz close to 2*Ixx they do not
 * vanish at all.
 *
 * The Isaac Sim prototype hardcoded this coupling instead, at a value that is
 * sign-inverted for an ordinary quadrotor; this implementation derives it from the three
 * parameters, so set them honestly.
 *
 * @min 0.0001
 * @max 5.0
 * @unit kg m^2
 * @decimal 5
 * @increment 0.001
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_IZZ, 0.02f);

/**
 * Tilt-eigen controller full-scale torque
 *
 * The body torque produced at a normalized torque setpoint of 1.0, used to convert this
 * controller's physical N m into the dimensionless value control_allocator expects.
 *
 * Estimate it as roughly 2 * (maximum thrust of one rotor) * (arm length) for a
 * quadrotor. Getting it wrong is not dangerous in itself - it rescales the whole inner
 * loop uniformly, so the vehicle behaves as though MC_TEIG_WN, MC_TEIG_B and the inertias
 * were all scaled together - but the shipped gains only mean what they say once it is
 * right.
 *
 * @min 0.001
 * @max 100.0
 * @unit Nm
 * @decimal 3
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_TEIG_TRQ_MAX, 1.0f);
