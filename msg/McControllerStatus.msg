# Status of the pluggable multicopter controller framework (mc_controller).
# Published at a reduced rate for logging and diagnostics; the control path itself
# publishes vehicle_torque_setpoint / vehicle_thrust_setpoint, which control_allocator
# mixes into actuator setpoints.

uint64 timestamp		# time since system start (microseconds)

uint8 algorithm			# MC_CTRL_ALG value in effect
uint8 control_level		# level the active flight mode commands at

uint8 CONTROL_LEVEL_NONE = 0
uint8 CONTROL_LEVEL_BODY_RATE = 1
uint8 CONTROL_LEVEL_ATTITUDE = 2
uint8 CONTROL_LEVEL_TRAJECTORY = 3

bool fallback_active		# true while the reference cascade has taken over
uint8 fallback_reason

uint8 FALLBACK_NONE = 0
uint8 FALLBACK_UNSUPPORTED_LEVEL = 1	# controller does not support the active control level
uint8 FALLBACK_INVALID_OUTPUT = 2	# controller produced a non-finite or invalid output
uint8 FALLBACK_STALE = 3		# controller produced no valid output within MC_CTRL_WD_MS
uint8 FALLBACK_UNINITIALISED = 4	# no controller instance available

float32[3] torque_sp		# normalized torque setpoint published this cycle
float32[3] thrust_sp		# normalized thrust setpoint published this cycle

float32 dt			# [s] time step of the controller update
float32 dt_attitude		# [s] time since the previous vehicle_attitude sample
float32 dt_position		# [s] time since the previous vehicle_local_position sample

uint32 update_count		# controller updates since start
uint32 invalid_output_count	# updates rejected by the output stage

float32[8] debug		# free-form, meaning defined by the active controller
