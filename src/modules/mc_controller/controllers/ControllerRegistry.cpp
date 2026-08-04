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

#include "ControllerRegistry.hpp"
#include "TemplateController.hpp"

#include <px4_platform_common/log.h>

namespace mc_ctrl
{

MulticopterControllerBase *createController(int32_t alg, ModuleParams *parent, CascadedPdController *reference)
{
	switch (static_cast<Algorithm>(alg)) {
	case Algorithm::CascadedPd:
		// The reference is always allocated so fallback is allocation-free; reuse it
		// rather than constructing a second identical controller.
		return reference;

	case Algorithm::Template:
		return new TemplateController(parent);

	case Algorithm::Stock:

	// mc_controller is not started when MC_CTRL_ALG=0, so reaching here means the
	// parameter changed after boot. Fall back rather than run nothing. Note this does
	// NOT get you the stock cascade: reaching stock needs MC_CTRL_ALG=0 and a reboot,
	// because that is a decision about which modules the startup script launches.
	default:
		PX4_ERR("MC_CTRL_ALG=%d not available, using %s", (int)alg, reference->name());
		return reference;
	}
}

const char *algorithmName(int32_t alg)
{
	switch (static_cast<Algorithm>(alg)) {
	case Algorithm::Stock: return "stock";

	case Algorithm::CascadedPd: return "cascaded_pd";

	case Algorithm::Template: return "template";

	}

	return "unknown";
}

} // namespace mc_ctrl
