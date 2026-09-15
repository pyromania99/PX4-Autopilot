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

# trajectory_stage_params.c is deliberately NOT listed below. src/lib/parameters
# generates the definitions from a directory scan, so compiling it here as well would be a
# redefinition error. What makes the scan reach this directory at all is the
# px4_add_library() call itself: it appends CMAKE_CURRENT_SOURCE_DIR to PX4_MODULE_PATHS,
# and the scanner is non-recursive. A params file in a directory with no px4_add_library()
# or px4_add_module() is silently ignored.
px4_add_library(McControllerTrajectoryStage
	TrajectoryStage.cpp
	TrajectoryStage.hpp
)
target_compile_options(McControllerTrajectoryStage PRIVATE ${MAX_CUSTOM_OPT_LEVEL})
target_include_directories(McControllerTrajectoryStage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Same fence as controllers/: no PositionControl / AttitudeControl / RateControl. This
# stage is shared between the laws in this module, not with the stock cascade.
target_link_libraries(McControllerTrajectoryStage PUBLIC MulticopterControllerApi)

# Functional gtest: TrajectoryStage uses DEFINE_PARAMETERS, so the live parameter system
# is required (contrast ControllerApi, which is parameter-free and uses the unit variant).
px4_add_functional_gtest(SRC TrajectoryStageTest.cpp
	LINKLIBS McControllerTrajectoryStage MulticopterControllerApi)
