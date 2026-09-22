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


px4_add_library(MulticopterControllers
	CascadedPdController.cpp
	CascadedPdController.hpp
	ControllerRegistry.cpp
	ControllerRegistry.hpp
	EigenController.cpp
	EigenController.hpp
	PoleAdapter.cpp
	PoleAdapter.hpp
	RateLimits.cpp
	RateLimits.hpp
	ActuatorLead.cpp
	ActuatorLead.hpp
	Seat.cpp
	Seat.hpp
	TemplateController.cpp
	TemplateController.hpp
)
target_compile_options(MulticopterControllers PRIVATE ${MAX_CUSTOM_OPT_LEVEL})
target_include_directories(MulticopterControllers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Deliberately NOT linking PositionControl / AttitudeControl / RateControl. The
# controllers here implement their own control laws end to end; the stock cascade
# remains reachable only as MC_CTRL_ALG=0, which runs the stock modules themselves.
# Keeping the link out is what stops the two from quietly growing back together.
target_link_libraries(MulticopterControllers PUBLIC
	MulticopterControllerApi
	McControllerTrajectoryStage
)

# Functional gtests: both controllers use DEFINE_PARAMETERS, so the live parameter system
# is required.
px4_add_functional_gtest(SRC CascadedPdControllerTest.cpp
	LINKLIBS MulticopterControllers MulticopterControllerApi McControllerTrajectoryStage)

px4_add_functional_gtest(SRC EigenControllerTest.cpp
	LINKLIBS MulticopterControllers MulticopterControllerApi McControllerTrajectoryStage)

# The seat is a standalone scalar law - it takes bare vectors and knows nothing about
# ControllerState - so it is tested without a vehicle. Functional rather than unit: it
# uses DEFINE_PARAMETERS, so the live parameter system is required.
px4_add_functional_gtest(SRC SeatTest.cpp
	LINKLIBS MulticopterControllers MulticopterControllerApi McControllerTrajectoryStage)
