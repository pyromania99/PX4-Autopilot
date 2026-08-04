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
 * @file ControllerRegistry.hpp
 * @brief Map MC_CTRL_ALG to a controller instance.
 *
 * A plain switch rather than a generated registry: it is greppable, gives a
 * -Wswitch warning when an enumerator is added without a case, and needs no
 * build-system change to add a controller. Same precedent as
 * ControlAllocator::update_allocation_method() selecting on CA_METHOD.
 *
 * To add a controller, see src/modules/mc_controller/README.md.
 */

#pragma once

#include "CascadedPdController.hpp"

#include <MulticopterControllerBase.hpp>

namespace mc_ctrl
{

enum class Algorithm : int32_t {
	Stock       = 0,	///< mc_controller is not started at all; stock modules run
	CascadedPd  = 1,	///< cascaded PD, geometric attitude law, no yaw control
	Template    = 2,	///< skeleton for new controllers
};

/**
 * @param alg        MC_CTRL_ALG value
 * @param parent     ModuleParams parent for the new controller
 * @param reference  the always-allocated fallback controller, returned for
 *                   Algorithm::CascadedPd so no second allocation occurs
 * @return a controller; never nullptr (falls back to @p reference)
 */
MulticopterControllerBase *createController(int32_t alg, ModuleParams *parent, CascadedPdController *reference);

const char *algorithmName(int32_t alg);

} // namespace mc_ctrl
