import numpy as np
import rclpy
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from rclpy.action import ActionClient
from moveit_msgs.srv import GetPositionIK, GetPlanningScene, GetCartesianPath
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint, \
                            PlanningScene, PlanningSceneComponents, CollisionObject, \
                            RobotState
from franka_msgs.action import Grasp
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
import math
import copy


class PlanAndExecute:
    """
    Execute functions based off of service calls to return either return or execute a move plan.

    Uses information about the current state of the robot and the goal position to generate
    a move plan and either return or execute it. Functions occur asynchronously. This will timeout
    if the necassary services are not availible.

    CLIENTS:
    -------
    _action_client (type: MoveGroup): Creates a MoveGroup object we fill out
    _execute_client (type: ExecuteTrajectory): Executes a plan
    IK (type: PositionIKRequest): Create an IK message based off of goal states


    SUBSCRIPTIONS:
    -------------
    js_sub (type: sensor_msgs/JoinState.msg) - subs to the joint states

    PUBLISHERS:
    ----------
    block_pub (type: PlanningScene): publishes an added block to the scene
    """

    def __init__(self, node):
        """Initilize varibles and states, fill messages, and create callback groups."""
        self.node = node
        self.node.cbgroup = MutuallyExclusiveCallbackGroup()
        self.node._action_client = ActionClient(self.node,
                                                MoveGroup,
                                                '/move_action')
        self.node._execute_client = ActionClient(self.node,
                                                 ExecuteTrajectory,
                                                 '/execute_trajectory')
        self.node._gripper_client = ActionClient(self.node,
                                                 Grasp,
                                                 '/panda_gripper/grasp')
        self.node.IK = self.node.create_client(GetPositionIK,
                                               "/compute_ik",
                                               callback_group=self.node.cbgroup)
        if not self.node.IK.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Timeout waiting for "IK" service to become available')
        self.node.planscene = self.node.create_client(GetPlanningScene,
                                                      "/get_planning_scene",
                                                      callback_group=self.node.cbgroup)
        if not self.node.planscene.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Timeout waiting for "planscene" service to become available')
        self.node.cartisian = self.node.create_client(GetCartesianPath,
                                                      "/compute_cartesian_path",
                                                      callback_group=self.node.cbgroup)
        if not self.node.cartisian.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Timeout waiting for "cartisian" service to become available')
        self.move_group = self.node.movegroup
        self.node.js_sub = self.node.create_subscription(JointState,
                                                         "/joint_states",
                                                         self.js_callback,
                                                         10)
        self.js = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.node.block_pub = self.node.create_publisher(PlanningScene, "/planning_scene", 10)
        self.master_goal = MoveGroup.Goal()
        self.master_goal.request.workspace_parameters.header.frame_id = 'panda_link0'
        self.master_goal.request.workspace_parameters.min_corner.x = -1.0
        self.master_goal.request.workspace_parameters.min_corner.y = -1.0
        self.master_goal.request.workspace_parameters.min_corner.z = -1.0
        self.master_goal.request.workspace_parameters.max_corner.x = 1.0
        self.master_goal.request.workspace_parameters.max_corner.y = 1.0
        self.master_goal.request.workspace_parameters.max_corner.z = 1.0
        self.master_goal.request.group_name = 'panda_manipulator'
        self.master_goal.request.num_planning_attempts = 10
        self.master_goal.request.allowed_planning_time = 5.0
        self.master_goal.request.planner_id = ''
        self.master_goal.request.start_state.is_diff = False
        self.master_goal.request.start_state.joint_state.velocity = []
        self.master_goal.request.start_state.joint_state.effort = []
        self.master_goal.request.start_state.multi_dof_joint_state.header.frame_id = 'panda_link0'
        self.master_goal.request.start_state.attached_collision_objects = []
        self.master_goal.request.start_state.is_diff = False
        self.master_goal.request.pipeline_id = 'move_group'
        self.master_goal.request.max_velocity_scaling_factor = 0.1
        self.master_goal.request.max_acceleration_scaling_factor = 0.1
        self.collision_list = []

    async def removeTower(self):
        """Remove a collision object from the planning scene."""
        scene_request = PlanningSceneComponents()
        scene_request.components = 0
        temp_scene_request = GetPlanningScene.Request(components=scene_request)
        scene = await self.node.planscene.call_async(temp_scene_request)
        scene = scene.scene
        scene.robot_state.joint_state = self.js
        self.collision_list = self.collision_list[:2]
        scene.world.collision_objects = self.collision_list
        self.node.block_pub.publish(scene)

    def printBlock(self, req):
        """Make a string of a request or message then log it to the terminal."""
        new_str = (str(req)).replace(',', ',\n')
        self.node.get_logger().info(new_str)

    def js_callback(self, data):
        """Save js with the joint state data (sensor_msgs/JointStates type)."""
        self.js = data

    def fill_constraints(self, joint_names, joint_positions, tol):
        """
        Fill the joint constraints field with information from the joint states message.

        Argument Description:
            joint_names[] (str): List of robot joint names.
            joint_positions[] (float): List of joint position corresponding to joint_names.
            tol (float): Allowable tolerance for joints.
        """
        self.node.get_logger().info("Filling Constraints")
        constraints = []
        for n, i in enumerate(joint_names):
            name = i
            pos = joint_positions[n]
            constraint_i = JointConstraint(joint_name=name,
                                           position=float(pos),
                                           tolerance_above=tol,
                                           tolerance_below=tol,
                                           weight=1.0)
            constraints.append(constraint_i)
        self.master_goal.request.goal_constraints = [Constraints(name='',
                                                                 joint_constraints=constraints)]

    def createIKreq(self, end_pos, end_orientation):
        """
        Create an IK message filled with info from the goal pose and orientation.

        Argument Description:
            end_pose (Point): The end/goal position of the robot.
            end_pose (Quaternion): The end/goal orientation of the robot.

        Output Description:
            request: The end position IK message from end goal pose.
        """
        request = PositionIKRequest()
        request.group_name = self.master_goal.request.group_name
        request.robot_state.joint_state.name = self.js.name
        request.robot_state.joint_state.position = self.js.position
        request.robot_state.joint_state.header.stamp = self.node.get_clock().now().to_msg()
        request.constraints.name = ''
        request.pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        request.pose_stamped.header.frame_id = 'panda_link0'
        request.pose_stamped.pose.position = end_pos
        request.pose_stamped.pose.orientation = end_orientation
        request.timeout.sec = 5
        return request

    def createWaypoints(self, start_pose, end_pose, max_step):
        """
        Create a list of poses  that follow a straight line.

        Argument Description:
            start_pose (Pose): The start pose of the robot.
            end_pose (Pose): The end/goal pose of the robot.
            max_step (float): The biggest step size that the robot can make.

        Output Description:
            points[] (Pose): List of pose points from start to end that are in a straight line.
        """
        points = [start_pose]
        xi = start_pose.position.x
        yi = start_pose.position.y
        zi = start_pose.position.z
        xf = end_pose.position.x
        yf = end_pose.position.y
        zf = end_pose.position.z
        last_point = copy.copy(end_pose)
        last_point.orientation = start_pose.orientation
        self.node.get_logger().info("initial and final angles")
        d = math.sqrt((xf-xi)**2 + (yf-yi)**2 + (zf-zi)**2)
        sp = math.ceil(d / max_step)+1
        sx = (xf-xi)/sp
        sy = (yf-yi)/sp
        sz = (zf-zi)/sp
        self.node.get_logger().info("delta angles")
        for i in range(sp):
            npose = points[i]
            npose.position.x = points[i].position.x + sx
            npose.position.y = points[i].position.y + sy
            npose.position.z = points[i].position.z + sz
            points.append(npose)
        points.append(last_point)
        self.node.get_logger().info("POINTS")
        return points

    def createCartreq(self, start_pose, end_pose):
        """
        Create an Cartisian message filled with info from the goal pose and orientation.

        Argument Description:
            start_pose (Pose): The start pose of the robot.
            end_pose (Pose): The end/goal pose of the robot.

        Output Description:
            Cartreq: Cartesian request information.
        """
        max_step = 0.005
        points = self.createWaypoints(start_pose, end_pose, max_step)
        self.node.get_logger().info("creating cartisian message")
        constraint = Constraints()
        group_name = self.master_goal.request.group_name
        header = Header()
        link_name = 'panda_hand_tcp'
        start_state = RobotState()
        start_state.joint_state.name = self.js.name
        start_state.joint_state.position = self.js.position
        start_state.joint_state.header.stamp = self.node.get_clock().now().to_msg()
        start_state.multi_dof_joint_state.header.frame_id = 'panda_link0'
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = 'panda_link0'
        waypoints = points
        jump_threshold = 10.0
        prismatic_jump_threshold = 10.0
        revolute_jump_threshold = 10.0
        avoid_collisions = True
        path_constraints = constraint
        return header, start_state, group_name, link_name, waypoints, max_step, jump_threshold,\
            prismatic_jump_threshold, revolute_jump_threshold, avoid_collisions,\
            path_constraints

    def getStartPose(self):
        """
        Calculate the postion and orientation of the robot based on the tf frames.

        Argument Description:
            None

        Output Description:
            startpose (Pose): The current pose of the robot.
        """
        startpose = Pose()
        temp_frame_id = self.master_goal.request.workspace_parameters.header.frame_id
        t = self.tf_buffer.lookup_transform(temp_frame_id,
                                            'panda_hand_tcp',
                                            rclpy.time.Time())
        startpose.position.x = t.transform.translation.x
        startpose.position.y = t.transform.translation.y
        startpose.position.z = t.transform.translation.z
        startpose.orientation.x = t.transform.rotation.x
        startpose.orientation.y = t.transform.rotation.y
        startpose.orientation.z = t.transform.rotation.z
        startpose.orientation.w = t.transform.rotation.w
        return startpose

    async def plan_to_position(self, start_pose, end_pos, tol, execute):
        """
        Return MoveGroup action from a start pose to an end position.

        Argument Description:
            start_pose (Pose): The start pose of the robot.
            end_pos (Pose): The end/goal positin of the robot.
            tol (float): The allowable tolerance for the robot joints.
            execute (boolean): True to execute the IK plan, False to plan.

        Output Description:
            executed_result: The result after the execute if execute is true.
            plan_result: The results of the plan if execute is false.
        """
        self.node.get_logger().info("Plan to position")
        if not start_pose:
            start_pose = self.getStartPose()
            self.master_goal.request.start_state.joint_state = self.js
        else:
            request_start = self.createIKreq(start_pose.position, start_pose.orientation)
            request_temp = GetPositionIK.Request(ik_request=request_start)
            response_start = await self.node.IK.call_async(request_temp)
            self.master_goal.request.start_state.joint_state = response_start.solution.joint_state
        self.master_goal.planning_options.plan_only = not execute
        request = self.createIKreq(end_pos.position, start_pose.orientation)
        IK_response = await self.callIK(request)
        if IK_response:
            plan_result = await self.plan(IK_response, None, tol)
            if execute:
                execute_result = await self.execute(plan_result)
                return execute_result
            else:
                return plan_result
        else:
            return None

    async def plan_to_orientation(self, start_pose, end_orientation, tol, execute):
        """
        Return MoveGroup action from a start pose to an end orientation.

        Argument Description:
            start_pose (Pose): The start pose of the robot.
            end_orientaion (Pose): The end/goal pose of the robot.
            tol (float): The allowable tolerance for the robot joints.
            execute (boolean): True to execute the IK plan, False to plan.

        Output Description:
            executed_result: The result after the execute if execute is true.
            plan_result: The results of the plan if execute is flase.
        """
        if not start_pose:
            start_pose = self.getStartPose()
            self.master_goal.request.start_state.joint_state = self.js
        else:
            request_start = self.createIKreq(start_pose.position, start_pose.orientation)
            request_temp = GetPositionIK.Request(ik_request=request_start)
            response_start = await self.node.IK.call_async(request_temp)
            self.master_goal.request.start_state.joint_state = response_start.solution.joint_state
        self.master_goal.planning_options.plan_only = not execute
        request = self.createIKreq(start_pose.position, end_orientation.orientation)
        IK_response = await self.callIK(request)
        if IK_response:
            plan_result = await self.plan(IK_response, None, tol)
            if execute:
                self.node.get_logger().info("Executing plan")
                execute_result = await self.execute(plan_result)
                self.node.get_logger().info("Plan executed")
                return execute_result
            else:
                return plan_result
        else:
            return None

    async def plan_to_pose(self, start_pose, end_pose, joint_pos, tol, execute):
        """
        Return MoveGroup action from a start to end pose (position + orientation).

        Argument Description:
            start_pose (Pose): The start pose of the robot.
            end_pose (Pose): The end/goal pose of the robot.
            tol (float): The allowable tolerance for the robot joints.
            execute (boolean): True to execute the IK plan, False to plan.

        Output Description:
            executed_result: The result after the execute if execute is true.
            plan_result: The results of the plan if execute is false.
        """
        self.node.get_logger().info("Plan to Pose")
        if not start_pose:
            start_pose = self.getStartPose()
            self.master_goal.request.start_state.joint_state = self.js
        else:
            request_start = self.createIKreq(start_pose.position, start_pose.orientation)
            request_temp = GetPositionIK.Request(ik_request=request_start)
            response_start = await self.node.IK.call_async(request_temp)
            self.master_goal.request.start_state.joint_state = response_start.solution.joint_state
            self.node.get_logger().info(response_start)
            self.node.get_logger().info(self.master_goal.request.start_state.joint_state)

        self.master_goal.planning_options.plan_only = not execute
        request = self.createIKreq(end_pose.position, end_pose.orientation)
        IK_response = await self.callIK(request)
        if IK_response:
            plan_result = await self.plan(IK_response, joint_pos, tol)
            if execute:
                execute_result = await self.execute(plan_result)
                return execute_result
            else:
                return plan_result
        else:
            return None

    async def plan_to_cartisian_pose(self, start_pose, end_pose, v, execute):
        """
        Return MoveGroup action from a start to end pose (position + orientation).

        Argument Description:
            start_pose (Pose): The start pose of the robot.
            end_pose (Pose): The end/goal pose of the robot.
            v (float): Velocity multiplier to change cartesian speed.
            execute (boolean): True to execute the IK plan, False to plan.

        Output Description:
            executed_result: The result after the execute if execute is true.
        """
        self.node.get_logger().info("Plan to Pose")
        start_pose = self.getStartPose()
        self.master_goal.request.start_state.joint_state = self.js
        self.fill_constraints(self.js.name, self.js.position, 0.005)

        self.master_goal.planning_options.plan_only = not execute
        self.node.get_logger().info("requesting cartisian message")
        header, start_state, group_name, link_name, waypoints, max_step, jump_threshold,\
            prismatic_jump_threshold, revolute_jump_threshold, avoid_collisions,\
            path_constraints = self.createCartreq(start_pose, end_pose)
        self.node.get_logger().info("recieved cartesian message")
        self.node.get_logger().info("calling cartisian service")
        Cart_response = await self.callCart(header, start_state, group_name, link_name, waypoints,
                                            max_step, jump_threshold, prismatic_jump_threshold,
                                            revolute_jump_threshold, avoid_collisions,
                                            path_constraints)
        self.node.get_logger().info("finished cartstian service")

        if execute:
            for point in Cart_response.joint_trajectory.points:
                total_time = point.time_from_start.nanosec + point.time_from_start.sec * 1000000000
                total_time *= 1.0/v
                point.time_from_start.sec = math.floor(total_time / 1000000000)
                point.time_from_start.nanosec = math.floor(total_time % 1000000000)

                for i in range(len(point.velocities)):
                    point.velocities[i] *= v

                for i in range(len(point.accelerations)):
                    point.accelerations[i] *= v
            self.node.get_logger().info(f"Cart Len: {len(Cart_response.joint_trajectory.points)}")
            traj_goal = ExecuteTrajectory.Goal(trajectory=Cart_response)
            execute_future = await self.node._execute_client.send_goal_async(traj_goal)
            execute_result = await execute_future.get_result_async()
            self.node.get_logger().info("finished executing cartstian service")
            return execute_result

    async def callCart(self, Cartheader, Cartstart_state, Cartgroup_name, Cartlink_name,
                       Cartwaypoints, Cartmax_step, Cartjump_threshold,
                       pris_jump_thres, rev_jump_thres,
                       Cartavoid_collisions, Cartpath_constraints):
        """Compute cartesian path of from the Cartesian message."""
        response = await self.node.cartisian.call_async(GetCartesianPath.Request(header=Cartheader,
                                                        start_state=Cartstart_state,
                                                        group_name=Cartgroup_name,
                                                        link_name=Cartlink_name,
                                                        waypoints=Cartwaypoints,
                                                        max_step=Cartmax_step,
                                                        jump_threshold=Cartjump_threshold,
                                                        prismatic_jump_threshold=pris_jump_thres,
                                                        revolute_jump_threshold=rev_jump_thres,
                                                        avoid_collisions=Cartavoid_collisions,
                                                        path_constraints=Cartpath_constraints))
        error_code = response.error_code
        if error_code.val == -31:
            self.node.get_logger().info("Cartisian service Failed :(")
            return None
        else:
            self.printBlock(type(response.solution))
            return response.solution

    async def callIK(self, IKrequest):
        """Compute joint states using inverse kinematics from a IKrequest message."""
        self.node.get_logger().info("Computing IK!")
        response = await self.node.IK.call_async(GetPositionIK.Request(ik_request=IKrequest))
        error_code = response.error_code
        if error_code.val == -31:
            self.node.get_logger().info("IK service Failed :(")
            return None
        else:
            self.node.get_logger().info("IK Succeeded :)")
            return response.solution.joint_state

    async def plan(self, joint_state, joint_positions, tol):
        """Plan to a joint state."""
        joint_names = joint_state.name
        if joint_positions is None:
            joint_positions = np.array(joint_state.position)
        else:
            joint_positions.append(joint_state.position[7])
            joint_positions.append(joint_state.position[8])
            joint_positions = np.array(joint_positions)
        self.fill_constraints(joint_names, joint_positions, tol)
        self.node._action_client.wait_for_server()
        plan = await self.node._action_client.send_goal_async(self.master_goal)
        plan_result = await plan.get_result_async()
        self.node.get_logger().info("***************************")
        self.printBlock(type(plan_result))
        return plan_result

    async def execute(self, plan_result):
        """Execute a planned trajectory on RVIZ."""
        self.node.get_logger().info("Wait for execute client")
        self.node._execute_client.wait_for_server()
        self.node.get_logger().info("check1")
        traj_goal = ExecuteTrajectory.Goal(trajectory=plan_result.result.planned_trajectory)
        execute_future = await self.node._execute_client.send_goal_async(traj_goal)
        self.node.get_logger().info("check2")
        execute_result = await execute_future.get_result_async()
        self.node.get_logger().info("check3")
        return execute_result

    async def place_block(self, pos, dimensions, name):
        """Place block in RVIZ from pose when service Place is called."""
        self.node.get_logger().info("Place Block")
        scene_request = PlanningSceneComponents()
        scene_request.components = 0
        temp_scene_request = GetPlanningScene.Request(components=scene_request)
        scene = await self.node.planscene.call_async(temp_scene_request)
        scene = scene.scene
        scene.robot_state.joint_state = self.js
        primepose = Pose()
        prime = SolidPrimitive()
        prime.type = 1
        prime.dimensions = dimensions
        collision = CollisionObject()
        collision.header = self.js.header
        collision.header.frame_id = 'panda_link0'
        collision.pose = pos
        collision.id = name
        collision.primitives = [prime]
        collision.primitive_poses = [primepose]
        self.collision_list.append(collision)
        scene.world.collision_objects = self.collision_list
        self.node.block_pub.publish(scene)

    async def grab(self, width):
        """
        Send Service to Robot to grasp gripper.

        Argument Description:
            width: Width of expected object to be grabbed.
        """
        self.node.get_logger().info("grabbing")
        self.node._gripper_client.wait_for_server()
        self.node.get_logger().info("gripper client connected")
        grasp_goal = Grasp.Goal()
        grasp_goal.width = width
        grasp_goal.speed = 0.03
        grasp_goal.force = 50.0
        grasp_goal.epsilon.inner = 0.01
        grasp_goal.epsilon.outer = 0.01
        await self.node._gripper_client.send_goal_async(grasp_goal)
        self.node.get_logger().info("grabbed")

    async def release(self):
        """Releases grippers after grabbing object."""
        self.node.get_logger().info("releasing")
        self.node._gripper_client.wait_for_server()
        self.node.get_logger().info("gripper client connected")
        grasp_goal = Grasp.Goal()
        grasp_goal.width = 0.1
        grasp_goal.speed = 0.03
        grasp_goal.force = 50.0
        await self.node._gripper_client.send_goal_async(grasp_goal)
        self.node.get_logger().info("released")
