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
 * @file eigen_params.c
 * Gains and rigid-body properties for the eigen-dynamics controller (MC_CTRL_ALG=3).
 *
 * Unlike the cascaded PD law next door, this controller works in PHYSICAL units: its
 * inner loop is a feedback linearization that multiplies a desired angular acceleration
 * by the inertia tensor, so it needs real kg m^2 and converts the resulting N m to the
 * normalized torque control_allocator expects via MC_EIG_TRQ_MAX.
 *
 * Tilt limit, thrust limits and velocity limits still come from the framework's
 * MPC_TILTMAX_*, MPC_THR_* and MPC_*_VEL_MAX; mass and gravity are replaced by the hover
 * thrust estimate. There are no integrator gains because the law has no integrators.
 */

/**
 * Eigen controller horizontal position gain
 *
 * Desired horizontal acceleration per metre of horizontal position error.
 *
 * The default is the Isaac Sim prototype's swept kp_pos, which is already in acceleration
 * units and therefore transfers directly.
 *
 * @min 0.0
 * @max 5.0
 * @decimal 2
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_XY_P, 0.50f);

/**
 * Eigen controller horizontal velocity gain
 *
 * Desired horizontal acceleration per m/s of horizontal velocity error. This is the
 * damping term: it acts on the filtered estimator velocity, not on a numerical derivative
 * of the position error as the prototype did.
 *
 * @min 0.0
 * @max 10.0
 * @decimal 2
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_XY_D, 1.00f);

/**
 * Eigen controller vertical position gain
 *
 * Desired vertical acceleration per metre of altitude error.
 *
 * The prototype's kp_z was 15 N per metre; divided by its 1.5 kg vehicle that is 10
 * m/s^2 per metre, which is what this default is. Note that MC_PD_Z_P next door
 * deliberately runs at 4.0 because the prototype's stiffness saturates the collective on
 * any sizeable altitude step - expect to walk this down in SITL.
 *
 * @min 0.0
 * @max 30.0
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_Z_P, 10.0f);

/**
 * Eigen controller vertical velocity gain
 *
 * Desired vertical acceleration per m/s of climb rate error.
 *
 * The prototype's kd_z of 8 N per m/s divided by its 1.5 kg vehicle.
 *
 * @min 0.0
 * @max 20.0
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_Z_D, 5.3f);

/**
 * Eigen controller attitude gain
 *
 * Desired body rate per radian of roll/pitch angle error. Roll and pitch share one gain.
 * This feeds the desired rates the eigen-dynamics inner loop tracks; it is not a torque
 * gain.
 *
 * @min 0.0
 * @max 30.0
 * @unit 1/s
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_ATT_P, 7.0f);

/**
 * Eigen controller attitude error derivative gain
 *
 * Desired body rate per rad/s of roll/pitch angle ERROR RATE, computed as a finite
 * difference of the angle error across attitude samples. This is a second damping path
 * stacked on MC_EIG_WN, and it is the term most exposed to estimator noise - lower it
 * first if the inner loop buzzes.
 *
 * @min 0.0
 * @max 10.0
 * @decimal 2
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_ATT_D, 1.0f);

/**
 * Eigen controller rate bandwidth (eigenvalue real part)
 *
 * Own-axis body rate error gain: the -wn in the complex-conjugate eigenvalue pair
 * -wn +/- j*MC_EIG_B that this law places on the roll/pitch subsystem. Larger is a
 * stiffer, faster rate loop.
 *
 * @min 0.0
 * @max 60.0
 * @unit rad/s
 * @decimal 2
 * @increment 0.5
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_WN, 18.0f);

/**
 * Eigen controller rate cross-coupling (eigenvalue imaginary part)
 *
 * Cross-axis body rate error gain: roll rate error produces pitch torque and vice versa,
 * with opposite signs. This is what makes the law an "eigen" controller rather than two
 * independent axes - roll and pitch are stabilized as one rotating mode, which is the
 * coning mode a spinning or asymmetric vehicle actually exhibits.
 *
 * Set to 0 to decouple the axes and recover a conventional rate law.
 *
 * SIGN. With perfect gyroscopic cancellation the rate error obeys
 *   Om_err(t) = Om_err(0) * exp(-wn*t) * exp(+i*b*t)
 * so b is the imaginary part of a complex FIRST-ORDER eigenvalue: it sets the direction
 * the error spirals in, not how fast it decays. Decay is wn, whatever b is. b therefore
 * cannot by itself stabilise anything - and it does cost margin, because the loop delay
 * acts at |wn - i*b| = sqrt(wn^2 + b^2), which b only ever increases.
 *
 * NEGATIVE values are now permitted, and are what the analysis points at. A positive b
 * rotates the rate error PROGRADE, the same direction as both the residual gyroscopic
 * coupling ((Izz-It)/It * r, prograde) and the plant's own rotation; a negative b rotates
 * retrograde and can oppose them. Candidate values, none of which is settled:
 *   -c*r          (c = (Izz-It)/It = 0.896) if the explicit cancellation were absent
 *   +wn*tau*c*r   if the cancellation is correct but computed from a delayed omega
 *   -6 to -2      the optimum of the full delayed cascade at r = 6-10
 * These disagree with each other; sweep, do not trust any of them.
 *
 * @min -60.0
 * @max 60.0
 * @unit rad/s
 * @decimal 2
 * @increment 0.5
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_B, 10.2f);

/**
 * Eigen controller assumed roll/pitch rate damping
 *
 * The aerodynamic rate damping the vehicle is assumed to have, which the feedback
 * linearization cancels. It therefore enters the torque as +alpha*rate: it ADDS energy,
 * on the theory that the airframe is removing the same amount.
 *
 * Overestimating it is destabilizing. Keep it well below MC_EIG_WN.
 *
 * @min -10.0
 * @max 10.0
 * @unit 1/s
 * @decimal 3
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_ALPHA, 0.5f);

/**
 * Eigen controller yaw rate bandwidth
 *
 * Assumed yaw aerodynamic drag, and simultaneously the gain on the commanded yaw rate.
 * In this law the two are the same number: cancelling the assumed drag and imposing the
 * same eigenvalue leaves the yaw feedback term exactly zero, so what survives is a pure
 * feedforward Izz*beta*yaw_rate_setpoint.
 *
 * The practical consequence is that heading is never held. At Trajectory level the yaw
 * rate setpoint is always zero, so yaw torque there is nothing but the gyroscopic
 * compensation term, which vanishes for a symmetric airframe. That is deliberate and
 * matches MC_CTRL_ALG=1: a multirotor that has lost a rotor cannot hold heading, and
 * surrendering yaw is what leaves enough authority to hold position.
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
PARAM_DEFINE_FLOAT(MC_EIG_BETA, 0.6f);

/**
 * Eigen controller roll inertia
 *
 * Vehicle moment of inertia about the body x axis, including the rotors' parallel-axis
 * contribution - not just the centre body. Scales roll torque directly and enters the
 * gyroscopic coupling of the other two axes.
 *
 * The default is the Isaac Sim prototype's quadrotor. Measure or estimate this for your
 * airframe; a bifilar pendulum test is the usual method.
 *
 * @min 0.0001
 * @max 5.0
 * @unit kg m^2
 * @decimal 5
 * @increment 0.001
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_IXX, 0.01f);

/**
 * Eigen controller pitch inertia
 *
 * Vehicle moment of inertia about the body y axis. See MC_EIG_IXX.
 *
 * @min 0.0001
 * @max 5.0
 * @unit kg m^2
 * @decimal 5
 * @increment 0.001
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_EIG_IYY, 0.01f);

/**
 * Eigen controller yaw inertia
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
PARAM_DEFINE_FLOAT(MC_EIG_IZZ, 0.02f);

/**
 * Eigen controller full-scale torque
 *
 * The body torque produced at a normalized torque setpoint of 1.0, used to convert this
 * controller's physical N m into the dimensionless value control_allocator expects.
 *
 * Estimate it as roughly 2 * (maximum thrust of one rotor) * (arm length) for a
 * quadrotor. Getting it wrong is not dangerous in itself - it rescales the whole inner
 * loop uniformly, so the vehicle behaves as though MC_EIG_WN, MC_EIG_B and the inertias
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
PARAM_DEFINE_FLOAT(MC_EIG_TRQ_MAX, 1.0f);
