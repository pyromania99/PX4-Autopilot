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
 * @file cascaded_pd_params.c
 * Gains for the cascaded PD controller (MC_CTRL_ALG=1).
 *
 * Only gains live here. Tilt limit, thrust limits and velocity limits come from the
 * framework's MPC_TILTMAX_*, MPC_THR_* and MPC_*_VEL_MAX; mass and gravity are replaced
 * by the hover thrust estimate. There are no integrator gains because the law has no
 * integrators.
 */

/**
 * Cascaded PD horizontal position gain
 *
 * Desired horizontal acceleration per metre of horizontal position error.
 *
 * @min 0.0
 * @max 5.0
 * @decimal 2
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_PD_XY_P, 0.50f);

/**
 * Cascaded PD horizontal velocity gain
 *
 * Desired horizontal acceleration per m/s of horizontal velocity error. This is the
 * damping term: it acts on the filtered estimator velocity, not on a numerical
 * derivative of the position error.
 *
 * @min 0.0
 * @max 10.0
 * @decimal 2
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_PD_XY_D, 1.00f);

/**
 * Cascaded PD vertical position gain
 *
 * Desired vertical acceleration per metre of altitude error.
 *
 * The default is stock's altitude cascade collapsed into one gain:
 * MPC_Z_P * MPC_Z_VEL_P_ACC = 1.0 * 4.0.
 *
 * The Isaac Sim prototype used 15.0, which is ~4x stiffer and saturates the
 * collective on any sizeable altitude step.
 *
 * @min 0.0
 * @max 30.0
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_PD_Z_P, 4.0f);

/**
 * Cascaded PD vertical velocity gain
 *
 * Desired vertical acceleration per m/s of climb rate error.
 *
 * The default is stock's MPC_Z_VEL_P_ACC. The Isaac Sim prototype used 8.0.
 *
 * @min 0.0
 * @max 20.0
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_PD_Z_D, 4.0f);

/**
 * Cascaded PD attitude gain
 *
 * Normalized roll/pitch torque per radian of geometric attitude error. Roll and pitch
 * share one gain; yaw has no proportional term at all.
 *
 * The default is stock's two-stage cascade collapsed into one:
 * MC_ROLLRATE_P * MC_ROLL_P = 0.15 * 4.0. Note the effective attitude gain is
 * MC_PD_ATT_P / MC_PD_ATT_D, so this must be re-derived whenever MC_PD_ATT_D
 * changes if you want to keep parity with a given MC_ROLL_P.
 *
 * @min 0.0
 * @max 10.0
 * @decimal 3
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_PD_ATT_P, 0.6f);

/**
 * Cascaded PD attitude rate damping gain
 *
 * Normalized roll/pitch torque per rad/s of body rate. Also acts as the rate gain in
 * Acro, where there is no attitude stage above it.
 *
 * The default is stock's MC_ROLLRATE_P. Expect to retune.
 *
 * Values below 0.01 are clamped to 0.01: the inner loop is factored as a rate law, so
 * zero damping would disable roll and pitch entirely rather than leaving them undamped.
 *
 * @min 0.01
 * @max 2.0
 * @decimal 3
 * @increment 0.005
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_PD_ATT_D, 0.15f);

/**
 * Cascaded PD yaw rate damping gain
 *
 * Normalized yaw torque per rad/s of yaw rate. This is the ONLY yaw term in the
 * controller - there is no yaw proportional term, so heading is never held and RC yaw
 * stick has no effect.
 *
 * The default of 0 disables yaw actuation entirely, which is what lets the vehicle spin
 * freely after a rotor failure instead of saturating yaw torque fighting it. Raise it to
 * damp the spin rate at the cost of some roll/pitch authority.
 *
 * @min 0.0
 * @max 2.0
 * @decimal 3
 * @increment 0.005
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_PD_YAWR_D, 0.0f);
