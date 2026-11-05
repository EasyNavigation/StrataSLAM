# Floor state published on /strataslam/floor_state.
std_msgs/Header header
int32 floor                # active floor; -1 while a transition is in progress or the floor is pending
string state               # on_floor | in_transition | lost | pending
string transition_type     # type of the last transition: stairs | ramp | elevator | aerial | unknown | ""
uint32 session
uint32 num_floors
float64 altitude           # m above floor 0 (barometer / IMU / odometry)
bool has_baro              # a barometer is enabled and has reported
float64 baro_height        # m, filtered barometric height since start-up (if has_baro)
