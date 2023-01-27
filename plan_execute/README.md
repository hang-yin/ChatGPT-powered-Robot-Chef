**Instructions:**

To plan or execute you need to call our service `/go_here` (there are sample commands below), which
takes in start pose, goal pose, and a boolean to indicate whether to plan or execute. If there is no 
start position indicated then the plan/execute happens with the robot's current position. If there 
is one given then the robot will plan from that postion. If there is more than one start position 
given then the program throws you an error.

To place a obstacle call the `/place` service (sample command below) and give it a pose. A block
will spawn at the indicated location realtive to the end effector. If the service is called again 
then a new block is not made, the current block just moves. 

Our node defaults to using our `plan_to_pose` function which takes the user input and moves the 
robot to the indicated pose. To see the robot move and execute you can run one of the service calls 
with no start position indicated and execute set to true. To test the other functions (`plan_to_orientation` or `plan_to_position`) you can
comment out the current call in the node and uncomment one of the other ones.

Sometime our IK service doesn't load up and the node crashes, to fix it relaunch the program.

# Sample Commands

## Run the Launch File
`ros2 launch plan_execute simple_move.launch.py`

## 0 initial start positions

**plan_to_orientation:**

execute:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [], goal_pose: {position: {x: 0.5, y: 0.5, z: 0.5}, orientation: {x: .707, y: 0.707, z: 0.0, w: 1}}, execute: 'true'}'`

plan:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [], goal_pose: {position: {x: 0.5, y: 0.5, z: 0.5}, orientation: {x: 0.707, y: 0.707, z: 0.0, w: 1}}, execute: 'false'}'`

**plan_to_pos:**

execute:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [], goal_pose: {position: {x: 0.5, y: 0.5, z: 0.5}, orientation: {x: .707, y: 0.707, z: 0.0, w: 1}}, execute: 'true'}'`

plan:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [], goal_pose: {position: {x: 0.5, y: 0.5, z: 0.5}, orientation: {x: .707, y: 0.707, z: 0.0, w: 1}}, execute: 'false'}'`

**plan_to_position:**

execute:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [], goal_pose: {position: {x: 0.3, y: 0.3, z: 0.3}, orientation: {x: .707, y: 0.707, z: 0.0, w: 1}}, execute: 'true'}'`

plan:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [], goal_pose: {position: {x: 0.3, y: 0.3, z: 0.3}, orientation: {x: .707, y: 0.707, z: 0.0, w: 1}}, execute: 'false'}'`

## 1 initial start positions

Will only ever plan even if the execute value is true since the robot can't teleport. 
plan_to_orientation:

plan:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [{position: {x: 0.5, y: 0.5, z: 0.5}, orientation: {x: 0, y: 0, z: 0, w: 1}}], goal_pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.707, y: 0.707, z: 0, w: 1}},execute: 'true'}'`

**plan_to_pos:**

plan:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [{position: {x: 0.5, y: 0.5, z: 0.5}, orientation: {x: 0, y: 0, z: 0, w: 1}}], goal_pose: {position: {x: 0.2, y: 0.2, z: 0.2}, orientation: {x: 0.707, y: 0.707, z: 0, w: 1}},execute: 'true'}'`

**plan_to_position:**

plan:

`ros2 service call /go_here plan_execute_interface/srv/GoHere '{start_pose: [{position: {x: 0.5, y: 0.5, z: 0.5}, orientation: {x: 0, y: 0, z: 0, w: 1}}], goal_pose: {position: {x: 0.2, y: 0.2, z: 0.2}, orientation: {x: 0.707, y: 0.707, z: 0, w: 1}},execute: 'true'}'`

## Place the block

`ros2 service call /place plan_execute_interface/srv/Place '{place: {position: {x: 0.5, y: 0.5, z: 0.5}, orientation: {x: 0.5, y: 0.5, z: 0.5, w: 1}}}'`


####################################################################