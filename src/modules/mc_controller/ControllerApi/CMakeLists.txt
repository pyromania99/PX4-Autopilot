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

px4_add_library(MulticopterControllerApi
	ControllerIO.cpp
	ControllerIO.hpp
	MulticopterControllerBase.hpp
)
target_compile_options(MulticopterControllerApi PRIVATE ${MAX_CUSTOM_OPT_LEVEL})
target_include_directories(MulticopterControllerApi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Unit rather than functional gtest: the interface types are header-only plus a
# small .cpp, ModuleParams is header-only, and nothing here touches the parameter
# system or uORB at runtime. Matches the other module-local tests (PositionControl,
# AttitudeControl).
#
# The functional variant does not link for this target (undefined orb_publish /
# uorb_start / init_app_map). Note that is NOT a module-subdirectory limitation -
# VehicleState/ uses px4_add_functional_gtest from a sibling directory and links
# fine because it genuinely pulls in the parameter system. Prefer the unit variant
# wherever the code under test needs nothing from the runtime.
px4_add_unit_gtest(SRC ControllerApiTest.cpp LINKLIBS MulticopterControllerApi)
