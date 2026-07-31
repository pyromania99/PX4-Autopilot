############################################################################
#
#   Copyright (c) 2026 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

# Pluggable multicopter controller framework.

add_subdirectory(ControllerApi)
add_subdirectory(VehicleState)
add_subdirectory(CommandFrontEnd)
add_subdirectory(controllers)

px4_add_module(
	MODULE modules__mc_controller
	MAIN mc_controller
	COMPILE_FLAGS
		${MAX_CUSTOM_OPT_LEVEL}
	SRCS
		MulticopterController.cpp
		MulticopterController.hpp
		OuterLoop.cpp
		OuterLoop.hpp
	MODULE_CONFIG
		module.yaml
	# PositionControl / AttitudeControl / RateControl / Takeoff / SlewRate are NOT
	# listed here: px4_add_module() calls get_target_property() at configure time,
	# and this directory is processed before mc_pos_control defines those targets.
	# They arrive transitively through the PUBLIC linkage of the libraries below.
	DEPENDS
		MulticopterControllerApi
		MulticopterControllers
		McControllerCommandFrontEnd
		McControllerVehicleState
		McManualMapping
		mathlib
		px4_work_queue
	)
