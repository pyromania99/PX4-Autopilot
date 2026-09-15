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
 * @file trajectory_stage_params.c
 * Integrator gains for the shared trajectory stage.
 *
 * These belong to the stage itself rather than to any one control law, so they apply to
 * every law that opts in - MC_CTRL_ALG=1 and 3 today. The proportional and derivative
 * gains stay with the laws (MC_PD_*, MC_EIG_*), because those were tuned per law and
 * their vertical defaults genuinely differ.
 *
 * The integrator acts on POSITION error, not velocity error as stock mc_pos_control does.
 * These laws are not cascades - the D term acts on velocity error directly, with no
 * internally generated velocity setpoint - so at a steady position offset the velocity
 * error is zero and a velocity integrator would never see it. See TrajectoryStage.hpp.
 *
 * BOTH GAINS SHIP AT ZERO. A PD law with no integrator holds position perfectly in
 * simulation, where the airframe is symmetric and there is no wind, and drifts to a
 * standing offset on real hardware, where CG offset, thrust asymmetry and estimator tilt
 * bias are always present. Enabling this is therefore a hardware-tuning step, and turning
 * it on by default would change flight behaviour for a disturbance no simulation shows.
 */

/**
 * Trajectory stage horizontal position integral gain
 *
 * Rate of change of desired horizontal acceleration per metre of horizontal position
 * error. Removes the standing offset a PD law settles into against a constant
 * disturbance - wind, a laterally offset centre of gravity, thrust asymmetry, or an
 * estimator tilt bias, all of which look identical to the controller.
 *
 * Only active on axes where the flight mode actually commands a position. Altitude modes
 * command no horizontal position, so this has no effect there.
 *
 * Zero by default: the horizontal offset is invisible in simulation, so a nonzero default
 * would be untested. Raise it on hardware once a hover log shows the drift.
 *
 * @min 0.0
 * @max 2.0
 * @decimal 3
 * @increment 0.005
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_OL_XY_I, 0.0f);

/**
 * Trajectory stage vertical position integral gain
 *
 * Rate of change of desired vertical acceleration per metre of altitude error.
 *
 * Altitude is normally trimmed by the hover thrust estimator rather than by an
 * integrator: hover thrust is by definition the collective that produces 1 g, so once the
 * estimate converges the steady-state altitude error is already zero. This gain covers
 * what the estimator does not - the window before it converges, the periods where it is
 * invalid or heavily de-weighted, and sustained aggressive flight where its measurement
 * noise scaling keeps it from ever validating.
 *
 * Zero by default, for the same reason as the horizontal gain.
 *
 * @min 0.0
 * @max 5.0
 * @decimal 3
 * @increment 0.005
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_OL_Z_I, 0.0f);

/**
 * Trajectory stage integral limit
 *
 * Maximum acceleration any one axis of the integrator may contribute. Applied per axis,
 * both to the integration itself and to the correction absorbed when the hover thrust
 * estimate changes.
 *
 * One g by default, matching the bound stock mc_pos_control puts on its own vertical
 * velocity integral: a term larger than gravity can invert the sign of the collective on
 * its own, which is never a recovery from anything.
 *
 * @min 0.0
 * @max 20.0
 * @decimal 2
 * @increment 0.1
 * @unit m/s^2
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_OL_I_LIM, 9.81f);
