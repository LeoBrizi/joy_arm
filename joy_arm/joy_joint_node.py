#!/usr/bin/env python3
"""
Joystick-based joint-space control of robot arm.

Control mapping:
- axes[5] == -1: Enable control mode (trigger hold)
- axes[0]: Joint 1 (base rotation)
- axes[1]: Joint 2 (shoulder)
- axes[3]: Joint 3 (elbow)
- axes[4]: Joint 4 (wrist 1)
- button[-2] / button[-1]: Joint 5 positive / negative
- button[-4] / button[-3]: Joint 6 positive / negative
- button[0]: Close gripper
- button[1]: Open gripper
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Joy, JointState
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import numpy as np


class JoyJointNode(Node):
    """Node for joystick-based joint-space control of robot arm."""

    # Configuration
    NAMESPACE = "j100_0819"
    JOY_TOPIC = f"/{NAMESPACE}/joy_teleop/joy"
    MOVE_ACTION = f"/{NAMESPACE}/move_action"
    ARM_STATE_TOPIC = f"/{NAMESPACE}/manipulators/arm_0_joint_trajectory_controller/state"
    ARM_CMD_TOPIC = f"/{NAMESPACE}/manipulators/arm_0_joint_trajectory_controller/joint_trajectory"

    # MoveIt group names
    ARM_GROUP = "arm_0"
    GRIPPER_GROUP = "arm_0_gripper"
    GRIPPER_JOINT_NAME = "arm_0_gripper_right_finger_bottom_joint"

    # Gripper positions
    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSED = 1.0

    # Joystick mapping
    ENABLE_AXIS = 5  # axes[5] == -1 to enable

    # MoveIt error code mapping
    ERROR_CODES = {
        0: "UNDEFINED/CANCELLED",
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
        -12: "GOAL_IN_COLLISION",
        -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
        -14: "GOAL_CONSTRAINTS_VIOLATED",
        -15: "INVALID_GROUP_NAME",
        -16: "INVALID_GOAL_CONSTRAINTS",
        -17: "INVALID_ROBOT_STATE",
        -18: "INVALID_LINK_NAME",
        -19: "INVALID_OBJECT_NAME",
        -21: "FRAME_TRANSFORM_FAILURE",
        -22: "COLLISION_CHECKING_UNAVAILABLE",
        -23: "ROBOT_STATE_STALE",
        -24: "SENSOR_INFO_STALE",
        -25: "COMMUNICATION_FAILURE",
        -26: "START_STATE_INVALID",
        -27: "GOAL_STATE_INVALID",
        -28: "UNRECOGNIZED_GOAL_TYPE",
        -29: "CRASH",
        -30: "ABORT",
        -31: "NO_IK_SOLUTION",
    }

    def __init__(self):
        super().__init__('joy_joint_node')

        # Declare parameters
        self.declare_parameter('deadzone', 0.2)
        self.declare_parameter('control_rate', 60.0)  # Hz
        self.declare_parameter('gripper_close_button', 0)
        self.declare_parameter('gripper_open_button', 1)

        # Get parameters
        self.deadzone = self.get_parameter('deadzone').value
        self.control_rate = self.get_parameter('control_rate').value
        self.gripper_close_button = self.get_parameter('gripper_close_button').value
        self.gripper_open_button = self.get_parameter('gripper_open_button').value

        # State variables
        self.is_enabled = False
        self.joy_joints = np.zeros(4)  # axes[0,1,3,4] -> joints 1-4
        self.joy_joint5 = 0.0  # from buttons[-2]/[-1]
        self.joy_joint6 = 0.0  # from buttons[-4]/[-3]
        self.prev_buttons = []
        self.current_joint_state = None

        # Callback group for concurrent execution
        self.cb_group = ReentrantCallbackGroup()

        # Subscribe to joystick
        self.joy_sub = self.create_subscription(
            Joy,
            self.JOY_TOPIC,
            self.joy_callback,
            10,
            callback_group=self.cb_group
        )

        # Subscribe to arm controller state
        self.arm_state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            self.ARM_STATE_TOPIC,
            self.arm_state_callback,
            10,
            callback_group=self.cb_group
        )

        # Publisher for direct joint trajectory control
        self.arm_cmd_pub = self.create_publisher(
            JointTrajectory,
            self.ARM_CMD_TOPIC,
            10
        )

        # Action client for MoveGroup (used for gripper)
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

        self.get_logger().info("JoyJointNode initialized")
        self.get_logger().info(f"  Joy topic: {self.JOY_TOPIC}")
        self.get_logger().info(f"  Arm cmd topic: {self.ARM_CMD_TOPIC}")
        self.get_logger().info(f"  Control rate: {self.control_rate} Hz")
        self.get_logger().info(f"  Enable: axes[{self.ENABLE_AXIS}] == -1")
        self.get_logger().info(f"  Joints 1-4: axes[0,1,3,4]")
        self.get_logger().info(f"  Joint 5: buttons[-2]/[-1]")
        self.get_logger().info(f"  Joint 6: buttons[-4]/[-3]")

    def joy_callback(self, msg: Joy):
        """Process joystick input for joint-space control."""
        # Check enable condition
        if len(msg.axes) > self.ENABLE_AXIS:
            self.is_enabled = (msg.axes[self.ENABLE_AXIS] == -1.0)
        else:
            self.is_enabled = False

        # self.get_logger().info(
        #     f"[RAW] enabled={self.is_enabled} "
        #     f"axes=[{', '.join(f'{a:.2f}' for a in msg.axes)}] "
        #     f"buttons={list(msg.buttons)}"
        # )

        if self.is_enabled:
            # Joints 1-4 from stick axes
            if len(msg.axes) > 4:
                self.joy_joints[0] = self._apply_deadzone(msg.axes[0])
                self.joy_joints[1] = self._apply_deadzone(msg.axes[1])
                self.joy_joints[2] = self._apply_deadzone(msg.axes[3])
                self.joy_joints[3] = self._apply_deadzone(msg.axes[4])

            # Joint 5 from buttons[-2] (positive) and buttons[-1] (negative)
            if len(msg.buttons) >= 2:
                self.joy_joint5 = float(msg.buttons[-2]) - float(msg.buttons[-1])
                # self.get_logger().info(
                #     f"[BTN] j5: btn[{len(msg.buttons)-2}]={msg.buttons[-2]} "
                #     f"btn[{len(msg.buttons)-1}]={msg.buttons[-1]} -> j5={self.joy_joint5:.1f}"
                # )

            # Joint 6 from buttons[-4] (positive) and buttons[-3] (negative)
            if len(msg.buttons) >= 4:
                self.joy_joint6 = float(msg.buttons[-4]) - float(msg.buttons[-3])
                # self.get_logger().info(
                #     f"[BTN] j6: btn[{len(msg.buttons)-4}]={msg.buttons[-4]} "
                #     f"btn[{len(msg.buttons)-3}]={msg.buttons[-3]} -> j6={self.joy_joint6:.1f}"
                # )

            # Handle gripper buttons (rising edge detection)
            if len(msg.buttons) > max(self.gripper_close_button, self.gripper_open_button):
                if len(self.prev_buttons) == len(msg.buttons):
                    if (msg.buttons[self.gripper_close_button] == 1 and
                            self.prev_buttons[self.gripper_close_button] == 0):
                        self.get_logger().info("Gripper close requested")
                        self.send_gripper_goal(self.GRIPPER_CLOSED)

                    if (msg.buttons[self.gripper_open_button] == 1 and
                            self.prev_buttons[self.gripper_open_button] == 0):
                        self.get_logger().info("Gripper open requested")
                        self.send_gripper_goal(self.GRIPPER_OPEN)
        else:
            # Clear commands when disabled
            self.joy_joints = np.zeros(4)
            self.joy_joint5 = 0.0
            self.joy_joint6 = 0.0

        # Always update prev_buttons for edge detection
        if len(msg.buttons) > 0:
            self.prev_buttons = list(msg.buttons)

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def arm_state_callback(self, msg: JointTrajectoryControllerState):
        """Store current arm joint state."""
        if self.current_joint_state is None:
            self.get_logger().info(f"Received arm joint state: {list(msg.joint_names)}")

        joint_state = JointState()
        joint_state.name = list(msg.joint_names)
        joint_state.position = list(msg.actual.positions)
        if msg.actual.velocities:
            joint_state.velocity = list(msg.actual.velocities)
        self.current_joint_state = joint_state

    def control_timer_callback(self):
        """Main control loop - runs at control_rate Hz."""
        if not self.is_enabled:
            return

        # Skip if no motion commanded
        if (np.allclose(self.joy_joints, 0) and
                self.joy_joint5 == 0.0 and self.joy_joint6 == 0.0):
            return

        self.send_joint_command()

    def send_joint_command(self):
        """Send joint trajectory directly to controller.

        Maps joystick inputs to joint velocities:
        - joy_joints[0-3] -> joints 1-4 (joint_scale)
        - joy_joint5 -> joint 5 (wrist_scale)
        - joy_joint6 -> joint 6 (wrist_scale)
        """
        if self.current_joint_state is None:
            self.get_logger().warning("No joint state received yet")
            return

        current_positions = list(self.current_joint_state.position)
        joint_names = list(self.current_joint_state.name)

        if len(current_positions) < 6:
            self.get_logger().warning("Not enough joints in state")
            return

        dt = 1.0 / self.control_rate
        joint_scale = 0.8   # rad/s for joints 1-3 (base/shoulder/elbow)
        shoulder_scale = 0.7  # rad/s for joint 4 (wrist 1, tighter limits)
        wrist_scale = 0.6   # rad/s for joints 5-6 (button-driven)

        target_positions = current_positions.copy()
        target_positions[0] += self.joy_joints[0] * joint_scale * dt
        target_positions[1] += self.joy_joints[1] * joint_scale * dt
        target_positions[2] += self.joy_joints[2] * shoulder_scale * dt
        target_positions[3] += self.joy_joints[3] * shoulder_scale * dt
        target_positions[4] += self.joy_joint5 * wrist_scale * dt
        target_positions[5] += self.joy_joint6 * wrist_scale * dt

        if abs(target_positions[3]) > 2.6:
            target_positions[3] = 2.6 * np.sign(target_positions[3]) 

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.velocities = [0.0] * len(current_positions)
        point.time_from_start = Duration(sec=0, nanosec=5_000_000)

        traj.points = [point]

        self.arm_cmd_pub.publish(traj)

        # Debug: joystick input
        # self.get_logger().info(
        #     f"[JOY] enabled={self.is_enabled} "
        #     f"axes=[{self.joy_joints[0]:.3f},{self.joy_joints[1]:.3f},"
        #     f"{self.joy_joints[2]:.3f},{self.joy_joints[3]:.3f}] "
        #     f"j5={self.joy_joint5:.1f} j6={self.joy_joint6:.1f}"
        # )
        # Debug: command sent
        self.get_logger().info(
            f"[CMD] current=[{', '.join(f'{p:.4f}' for p in current_positions)}]"
        )
        self.get_logger().info(
            f"[CMD] target =[{', '.join(f'{p:.4f}' for p in target_positions)}]"
        )
        self.get_logger().info(
            f"[CMD] delta  =[{', '.join(f'{t-c:.6f}' for t, c in zip(target_positions, current_positions))}]"
        )

    def send_gripper_goal(self, position: float):
        """Send gripper goal via MoveGroup action with joint constraint."""
        if not self.move_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning("MoveGroup action server not available for gripper")
            return

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.GRIPPER_GROUP

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

    node = JoyJointNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
