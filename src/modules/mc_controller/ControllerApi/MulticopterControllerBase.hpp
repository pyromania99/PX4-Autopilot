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
 * @file MulticopterControllerBase.hpp
 * @brief Abstract base class for pluggable multicopter control laws.
 *
 * A controller receives the estimator state and the commanded setpoint (at
 * whatever level the active flight mode commands) and produces motor moments.
 * It is selected at runtime by MC_CTRL_ALG.
 *
 * To add a controller see src/modules/mc_controller/README.md.
 */

#pragma once

#include "ControllerIO.hpp"

#include <px4_platform_common/module_params.h>
#include <uORB/topics/mc_controller_status.h>
#include <uORB/topics/vehicle_attitude_setpoint.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>

/**
 * Deriving from ModuleParams gives every controller DEFINE_PARAMETERS(...) and
 * automatic parameter refresh: the framework calls updateParams() on itself and
 * ModuleParams cascades to children, so a controller constructed with the module
 * as its parent needs no parameter plumbing of its own.
 */
class MulticopterControllerBase : public ModuleParams
{
public:
	explicit MulticopterControllerBase(ModuleParams *parent) : ModuleParams(parent) {}
	~MulticopterControllerBase() override = default;

	MulticopterControllerBase(const MulticopterControllerBase &) = delete;
	MulticopterControllerBase &operator=(const MulticopterControllerBase &) = delete;

	/// Short name for `mc_controller status` and the boot log. Must be a string literal.
	virtual const char *name() const = 0;

	/**
	 * Bitmask of mc_ctrl::levelBit(...) for every ControlLevel this controller can
	 * be entered at. A full-stack controller returns mc_ctrl::kAllLevels.
	 *
	 * Entering a level outside this mask latches the reference-cascade fallback and
	 * logs loudly; it is checked at boot and on every MC_CTRL_ALG change so a bad
	 * configuration surfaces on the ground rather than at the first mode switch.
	 */
	virtual uint8_t supportedLevels() const = 0;

	bool supportsLevel(mc_ctrl::ControlLevel level) const
	{
		return (supportedLevels() & mc_ctrl::levelBit(level)) != 0;
	}

	/**
	 * Zero every integrator, filter and piece of internal state.
	 * Called on the `rate_ctrl` work queue when this controller is selected, on every
	 * arm transition, and on every ControlLevel change.
	 */
	virtual void reset() = 0;

	/**
	 * Run the control law.
	 *
	 * Called at gyro rate on the rate_ctrl work queue. MUST be real-time safe:
	 * no heap allocation, no blocking, no printf, bounded execution time.
	 *
	 * The controller MUST honour command.reset_integrals.
	 *
	 * @return false to request the framework failsafe (equivalent to leaving
	 *         output.valid false).
	 */
	virtual bool update(const mc_ctrl::ControllerState &state,
			    const mc_ctrl::ControllerCommand &command,
			    float dt,
			    mc_ctrl::ControllerOutput &output) = 0;

	/**
	 * Optional: expose the trajectory stage's internal position setpoint for publication.
	 *
	 * vehicle_local_position_setpoint is consumed by the flight tasks for smooth
	 * setpoint resets and by the POSITION_TARGET_LOCAL_NED mavlink stream. A
	 * controller with no such internal concept simply leaves this unimplemented.
	 */
	virtual void fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &) const {}

	/// Optional anti-windup feedback from control_allocator. Default: ignore.
	virtual void setAllocatorFeedback(const mc_ctrl::AllocatorFeedback &) {}

	/// Optional: fill mc_controller_status.debug[] and friends. Called at ~50 Hz.
	virtual void fillStatus(mc_controller_status_s &) const {}

	/// Optional: extra detail for `mc_controller status`.
	virtual int printStatus() const { return 0; }
};
