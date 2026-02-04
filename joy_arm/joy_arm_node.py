#!/usr/bin/env python3
"""
Joystick-based Cartesian control of robot arm using MoveGroup action.

Control mapping:
- axes[5] == -1: Enable control mode (trigger)
- axes[0]: X movement (end-effector frame)
- axes[1]: Y movement (end-effector frame)
- axes[2]: Z movement (end-effector frame)
- axes[3]: Roll
- axes[4]: Pitch
- button[0]: Close gripper
- button[1]: Open gripper
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
    BoundingVolume,
    MotionPlanRequest,
    PlanningOptions,
    RobotState,
)
from shape_msgs.msg import SolidPrimitive
from tf2_msgs.msg import TFMessage

import tf2_ros
from scipy.spatial.transform import Rotation as R
import numpy as np


class JoyArmNode(Node):
    """Node for joystick-based Cartesian control of robot arm."""

    # Configuration
    NAMESPACE = "j100_0819"
    JOY_TOPIC = f"/{NAMESPACE}/joy_teleop/joy"
    MOVE_ACTION = f"/{NAMESPACE}/move_action"

    # Frame names
    BASE_FRAME = "arm_0_base_link"
    EE_FRAME = "arm_0_end_effector_link"
    PLANNING_FRAME = "arm_0_base_link"

    # MoveIt group names
    ARM_GROUP = "arm_0"
    GRIPPER_GROUP = "arm_0_gripper"
    GRIPPER_JOINT_NAME = "arm_0_gripper_right_finger_bottom_joint"

    # Gripper positions
    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSED = 1.0

    # Joystick mapping
    ENABLE_AXIS = 5  # axes[5] == -1 to enable
    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2
    ROLL_AXIS = 3
    PITCH_AXIS = 4

    # MoveIt error code mapping
    ERROR_CODES = {
        1: "SUCCESS",
        99999: "FAILURE",
        -1: "PLANNING_FAILED",
        -2: "INVALID_MOTION_PLAN",
        -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
        -4: "CONTROL_FAILED",
        -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
        -6: "TIMED_OUT",
        -7: "PREEMPTED",
        -10: "START_STATE_IN_COLLISION",
        -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
        -12: "START_STATE_INVALID",
        -13: "GOAL_IN_COLLISION",
        -14: "GOAL_VIOLATES_PATH_CONSTRAINTS",
        -15: "GOAL_CONSTRAINTS_VIOLATED",
        -16: "GOAL_STATE_INVALID",
        -17: "UNRECOGNIZED_GOAL_TYPE",
        -18: "FRAME_TRANSFORM_FAILURE",
        -19: "COLLISION_CHECKING_UNAVAILABLE",
        -20: "ROBOT_STATE_STALE",
        -21: "SENSOR_INFO_STALE",
        -22: "COMMUNICATION_FAILURE",
        -23: "CRASH",
        -24: "ABORT",
        -31: "INVALID_GOAL_CONSTRAINTS",
        -32: "INVALID_GROUP_NAME",
        -33: "INVALID_OBJECT_NAME",
        -34: "INVALID_ROBOT_STATE",
        -35: "INVALID_LINK_NAME",
        -36: "NO_IK_SOLUTION",
    }

    def __init__(self):
        super().__init__('joy_arm_node')

        # Declare parameters
        self.declare_parameter('linear_scale', 0.05)  # m/s per joystick unit
        self.declare_parameter('angular_scale', 0.2)  # rad/s per joystick unit
        self.declare_parameter('deadzone', 0.1)
        self.declare_parameter('control_rate', 10.0)  # Hz
        self.declare_parameter('gripper_close_button', 0)
        self.declare_parameter('gripper_open_button', 1)
        self.declare_parameter('max_velocity_scaling', 0.5)
        self.declare_parameter('max_acceleration_scaling', 0.3)

        # Get parameters
        self.linear_scale = self.get_parameter('linear_scale').value
        self.angular_scale = self.get_parameter('angular_scale').value
        self.deadzone = self.get_parameter('deadzone').value
        self.control_rate = self.get_parameter('control_rate').value
        self.gripper_close_button = self.get_parameter('gripper_close_button').value
        self.gripper_open_button = self.get_parameter('gripper_open_button').value
        self.max_velocity_scaling = self.get_parameter('max_velocity_scaling').value
        self.max_acceleration_scaling = self.get_parameter('max_acceleration_scaling').value

        # State variables
        self.is_enabled = False
        self.joy_linear = np.zeros(3)  # x, y, z velocities
        self.joy_angular = np.zeros(2)  # roll, pitch velocities
        self.prev_buttons = []
        self.goal_in_progress = False
        self.current_goal_handle = None

        # Callback group for concurrent execution
        self.cb_group = ReentrantCallbackGroup()

        # Setup TF2 with namespaced topics
        self._setup_tf()

        # Subscribe to joystick
        self.joy_sub = self.create_subscription(
            Joy,
            self.JOY_TOPIC,
            self.joy_callback,
            10,
            callback_group=self.cb_group
        )

        # Action client for MoveGroup (used for both arm and gripper)
        self.move_action_client = ActionClient(
            self,
            MoveGroup,
            self.MOVE_ACTION,
            callback_group=self.cb_group
        )

        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_timer_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info(f"JoyArmNode initialized")
        self.get_logger().info(f"  Joy topic: {self.JOY_TOPIC}")
        self.get_logger().info(f"  Move action: {self.MOVE_ACTION}")
        self.get_logger().info(f"  Control rate: {self.control_rate} Hz")
        self.get_logger().info(f"  Enable: axes[{self.ENABLE_AXIS}] == -1")

    def _setup_tf(self):
        """Setup TF2 with namespaced topics."""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS profiles matching the default TF listener
        tf_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE
        )
        tf_static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Destroy the default subscriptions
        self.destroy_subscription(self.tf_listener.tf_sub)
        self.destroy_subscription(self.tf_listener.tf_static_sub)

        # Create new subscriptions with namespaced topics
        # - manipulators/tf_static: robot base static transforms + arm mounting
        # - manipulators/tf: wheel odometry
        # - tf: arm joint transforms (dynamic)
        self.tf_static_sub_manip = self.create_subscription(
            TFMessage,
            f"/{self.NAMESPACE}/manipulators/tf_static",
            self.tf_listener.static_callback,
            tf_static_qos,
            callback_group=self.tf_listener.group
        )
        self.tf_sub_manip = self.create_subscription(
            TFMessage,
            f"/{self.NAMESPACE}/manipulators/tf",
            self.tf_listener.callback,
            tf_qos,
            callback_group=self.tf_listener.group
        )
        # Arm joint transforms are on /tf (not manipulators)
        self.tf_sub_robot = self.create_subscription(
            TFMessage,
            f"/{self.NAMESPACE}/tf",
            self.tf_listener.callback,
            tf_qos,
            callback_group=self.tf_listener.group
        )

        self.get_logger().info("TF listener configured for namespaced topics")

    def joy_callback(self, msg: Joy):
        """Process joystick input."""
        # Check enable condition
        if len(msg.axes) > self.ENABLE_AXIS:
            self.is_enabled = (msg.axes[self.ENABLE_AXIS] == -1.0)
        else:
            self.is_enabled = False

        if self.is_enabled:
            # Read position axes with deadzone
            if len(msg.axes) > max(self.X_AXIS, self.Y_AXIS, self.Z_AXIS):
                self.joy_linear[0] = self._apply_deadzone(msg.axes[self.X_AXIS])
                self.joy_linear[1] = self._apply_deadzone(msg.axes[self.Y_AXIS])
                self.joy_linear[2] = self._apply_deadzone(msg.axes[self.Z_AXIS])

            # Read orientation axes with deadzone
            if len(msg.axes) > max(self.ROLL_AXIS, self.PITCH_AXIS):
                self.joy_angular[0] = self._apply_deadzone(msg.axes[self.ROLL_AXIS])
                self.joy_angular[1] = self._apply_deadzone(msg.axes[self.PITCH_AXIS])

            # Handle gripper buttons (rising edge detection) - only when enabled
            if len(msg.buttons) > max(self.gripper_close_button, self.gripper_open_button):
                if len(self.prev_buttons) == len(msg.buttons):
                    # Close gripper on button press
                    if (msg.buttons[self.gripper_close_button] == 1 and
                            self.prev_buttons[self.gripper_close_button] == 0):
                        self.get_logger().info("Gripper close requested")
                        self.send_gripper_goal(self.GRIPPER_CLOSED)

                    # Open gripper on button press
                    if (msg.buttons[self.gripper_open_button] == 1 and
                            self.prev_buttons[self.gripper_open_button] == 0):
                        self.get_logger().info("Gripper open requested")
                        self.send_gripper_goal(self.GRIPPER_OPEN)
        else:
            # Clear velocities when disabled
            self.joy_linear = np.zeros(3)
            self.joy_angular = np.zeros(2)

        # Always update prev_buttons for edge detection (outside is_enabled check)
        if len(msg.buttons) > 0:
            self.prev_buttons = list(msg.buttons)

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def control_timer_callback(self):
        """Main control loop - runs at control_rate Hz."""
        if not self.is_enabled:
            return

        # Skip if no motion commanded
        if np.allclose(self.joy_linear, 0) and np.allclose(self.joy_angular, 0):
            return

        # Skip if a goal is already in progress
        if self.goal_in_progress:
            return

        # Get current pose
        current_pose = self.get_current_pose()
        if current_pose is None:
            return

        # Compute delta from joystick (in EE frame)
        dt = 1.0 / self.control_rate
        delta_pos_ee = self.joy_linear * self.linear_scale * dt
        delta_rot = self.joy_angular * self.angular_scale * dt

        # Transform position delta from EE frame to base frame
        delta_pos_base = self.transform_velocity_to_base(delta_pos_ee)

        # Compute target pose
        target_pose = self.apply_delta(current_pose, delta_pos_base, delta_rot)

        # Send goal
        self.send_ik_goal(target_pose)

    def get_current_pose(self) -> Pose:
        """Get current end-effector pose from TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.BASE_FRAME,
                self.EE_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            pose = Pose()
            pose.position = Point(
                x=transform.transform.translation.x,
                y=transform.transform.translation.y,
                z=transform.transform.translation.z
            )
            pose.orientation = Quaternion(
                x=transform.transform.rotation.x,
                y=transform.transform.rotation.y,
                z=transform.transform.rotation.z,
                w=transform.transform.rotation.w
            )
            return pose

        except tf2_ros.TransformException as e:
            self.get_logger().warning(f"Could not get current pose: {e}")
            return None

    def transform_velocity_to_base(self, vel_ee_frame: np.ndarray) -> np.ndarray:
        """Transform velocity vector from end-effector frame to base frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.BASE_FRAME,
                self.EE_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            # Get rotation quaternion
            q = transform.transform.rotation
            rotation = R.from_quat([q.x, q.y, q.z, q.w])

            # Rotate velocity vector
            vel_base = rotation.apply(vel_ee_frame)
            return vel_base

        except tf2_ros.TransformException as e:
            self.get_logger().warning(f"Could not transform velocity: {e}")
            return vel_ee_frame

    def apply_delta(self, current_pose: Pose, delta_pos: np.ndarray,
                    delta_rot: np.ndarray) -> Pose:
        """Apply position and rotation delta to current pose."""
        target_pose = Pose()

        # Apply position delta
        target_pose.position = Point(
            x=current_pose.position.x + delta_pos[0],
            y=current_pose.position.y + delta_pos[1],
            z=current_pose.position.z + delta_pos[2]
        )

        # Apply rotation delta (roll, pitch only)
        q = current_pose.orientation
        current_rot = R.from_quat([q.x, q.y, q.z, q.w])

        # Create delta rotation from roll and pitch
        delta_rotation = R.from_euler('xy', [delta_rot[0], delta_rot[1]])

        # Combine rotations (delta applied in base frame)
        new_rot = delta_rotation * current_rot
        new_quat = new_rot.as_quat()

        target_pose.orientation = Quaternion(
            x=new_quat[0],
            y=new_quat[1],
            z=new_quat[2],
            w=new_quat[3]
        )

        return target_pose

    def send_ik_goal(self, target_pose: Pose):
        """Send IK goal to MoveGroup action."""
        if not self.move_action_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warning("MoveGroup action server not available")
            return

        # Cancel previous goal if still running
        if self.current_goal_handle is not None:
            try:
                self.current_goal_handle.cancel_goal_async()
            except Exception:
                pass

        self.goal_in_progress = True

        # Build the goal
        goal_msg = MoveGroup.Goal()

        # Motion plan request
        request = MotionPlanRequest()
        request.group_name = self.ARM_GROUP
        request.num_planning_attempts = 1
        request.allowed_planning_time = 1.0
        request.max_velocity_scaling_factor = self.max_velocity_scaling
        request.max_acceleration_scaling_factor = self.max_acceleration_scaling

        # Use OMPL planner (default, more compatible with constraint-based goals)
        # Note: Pilz planner ("pilz_industrial_motion_planner") requires different goal format
        request.pipeline_id = "ompl"
        request.planner_id = "RRTConnect"

        # Create goal constraints
        constraints = Constraints()

        # Position constraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.PLANNING_FRAME
        position_constraint.link_name = self.EE_FRAME
        position_constraint.weight = 1.0

        # Define target position inside constraint region
        target_pose_for_region = Pose()
        target_pose_for_region.position = target_pose.position
        target_pose_for_region.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        position_constraint.constraint_region.primitive_poses.append(target_pose_for_region)
        position_constraint.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.02])  # 2cm tolerance
        )

        constraints.position_constraints.append(position_constraint)

        # Orientation constraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.PLANNING_FRAME
        orientation_constraint.link_name = self.EE_FRAME
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.5
        orientation_constraint.absolute_y_axis_tolerance = 0.5
        orientation_constraint.absolute_z_axis_tolerance = 0.5
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)

        request.goal_constraints.append(constraints)

        # Start state: use is_diff=True to indicate "use current state"
        request.start_state = RobotState()
        request.start_state.is_diff = True

        goal_msg.request = request

        # Planning options
        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = False

        # Send goal asynchronously
        self.get_logger().info(
            f"Sending IK goal: pos=({target_pose.position.x:.3f}, "
            f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f})")
        future = self.move_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._move_feedback_callback
        )
        future.add_done_callback(self._move_goal_response_callback)

    def _move_goal_response_callback(self, future):
        """Handle MoveGroup goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Move goal rejected")
            self.goal_in_progress = False
            self.current_goal_handle = None
            return

        self.get_logger().info("Move goal accepted")
        self.current_goal_handle = goal_handle

        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._move_result_callback)

    def _move_result_callback(self, future):
        """Handle MoveGroup result."""
        self.goal_in_progress = False
        self.current_goal_handle = None

        result = future.result()
        error_code = result.result.error_code.val
        error_name = self.ERROR_CODES.get(error_code, f"UNKNOWN({error_code})")

        if error_code == 1:  # SUCCESS
            self.get_logger().info("Move completed successfully")
        else:
            self.get_logger().info(f"Move failed: {error_name}")

    def _move_feedback_callback(self, feedback_msg):
        """Handle MoveGroup feedback."""
        pass  # Could log progress here if needed

    def send_gripper_goal(self, position: float):
        """Send gripper goal via MoveGroup action with joint constraint."""
        if not self.move_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning("MoveGroup action server not available for gripper")
            return

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.GRIPPER_GROUP

        # Create joint constraint for gripper
        goal_constraints = Constraints()
        jc = JointConstraint()
        jc.joint_name = self.GRIPPER_JOINT_NAME
        jc.position = position
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0

        goal_constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(goal_constraints)

        goal_msg.request.max_acceleration_scaling_factor = 1.0
        goal_msg.request.max_velocity_scaling_factor = 0.8
        goal_msg.request.num_planning_attempts = 1
        goal_msg.request.allowed_planning_time = 5.0

        future = self.move_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._gripper_goal_response_callback)

    def _gripper_goal_response_callback(self, future):
        """Handle gripper goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Gripper goal rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._gripper_result_callback)

    def _gripper_result_callback(self, future):
        """Handle gripper result."""
        result = future.result()
        if result.result.error_code.val == 1:
            self.get_logger().info("Gripper action completed successfully")
        else:
            self.get_logger().warning(
                f"Gripper action failed with error code: {result.result.error_code.val}")


def main(args=None):
    rclpy.init(args=args)

    node = JoyArmNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
