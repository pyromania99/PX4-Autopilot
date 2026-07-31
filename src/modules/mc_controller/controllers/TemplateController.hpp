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
 * @file TemplateController.hpp
 * @brief Skeleton controller - copy this to start your own.
 *
 * Implements a simple full-stack law so it is flyable in SITL as-is, and shows
 * every hook you are expected to use. See src/modules/mc_controller/README.md.
 *
 * THINGS THAT WILL BITE YOU IF YOU IGNORE THEM:
 *  - honour command.reset_integrals, or you wind up on the ground
 *  - always fill output.thrust; land_detector and the hover
 *    thrust estimator consume the published vehicle_thrust_setpoint
 *  - update() runs at gyro rate on a real-time work queue: no allocation, no
 *    blocking, no printf
 *  - declare every level you handle in supportedLevels(), or the framework
 *    latches the reference-cascade fallback
 */

#pragma once

#include <MulticopterControllerBase.hpp>

class TemplateController : public MulticopterControllerBase
{
public:
	explicit TemplateController(ModuleParams *parent) : MulticopterControllerBase(parent) {}
	~TemplateController() override = default;

	const char *name() const override { return "template"; }


	uint8_t supportedLevels() const override { return mc_ctrl::kAllLevels; }

	void reset() override;

	bool update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
		    float dt, mc_ctrl::ControllerOutput &output) override;

	void fillStatus(mc_controller_status_s &status) const override;

private:
	matrix::Vector3f _rate_integral{};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MC_TPL_RATE_P>) _param_rate_p,
		(ParamFloat<px4::params::MC_TPL_RATE_I>) _param_rate_i,
		(ParamFloat<px4::params::MC_TPL_ATT_P>)  _param_att_p
	)
};
