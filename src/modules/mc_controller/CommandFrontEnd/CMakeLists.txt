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


px4_add_library(McControllerCommandFrontEnd
	ControlLevelResolver.cpp
	ControlLevelResolver.hpp
	CommandFrontEnd.cpp
	CommandFrontEnd.hpp
)
target_compile_options(McControllerCommandFrontEnd PRIVATE ${MAX_CUSTOM_OPT_LEVEL})
target_include_directories(McControllerCommandFrontEnd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(McControllerCommandFrontEnd PUBLIC MulticopterControllerApi McManualMapping Takeoff SlewRate)

# Links commander's own mode_util so the level truth table cannot drift from
# commander. Unit gtest: nothing here needs the parameter system or uORB runtime.
px4_add_unit_gtest(SRC ControlLevelResolverTest.cpp
	LINKLIBS McControllerCommandFrontEnd MulticopterControllerApi mode_util
	INCLUDES ${PX4_SOURCE_DIR}/src/modules/commander)

# Functional gtest: CommandFrontEnd uses DEFINE_PARAMETERS.
px4_add_functional_gtest(SRC CommandFrontEndTest.cpp
	LINKLIBS McControllerCommandFrontEnd MulticopterControllerApi McManualMapping Takeoff)
