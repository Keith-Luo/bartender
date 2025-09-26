import time
import numpy as np
import math

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from joyconrobotics import JoyconRobotics
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

NUM_EPISODES = 3
FPS = 30
EPISODE_TIME_SEC = 100
TASK_DESCRIPTION = "My task description"

# --- MODIFY THIS --- #
# TODO: Replace with the actual IP of the robot
camera_config = {
    "left_arm_wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "right_arm_wrist": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=30),
    "head": OpenCVCameraConfig(index_or_path=8, width=640, height=480, fps=30),
}
robot_config = XLerobotClientConfig(remote_ip="127.0.0.1", id="xlerobot", cameras=camera_config)
robot = XLerobotClient(robot_config)

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# --- MODIFY THIS --- #
dataset = LeRobotDataset.create(
    repo_id="zonglin1104/xlerobot_joycon_dataset",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# This class is copied from examples/7_xlerobot_teleop_joycon.py
class FixedAxesJoyconRobotics(JoyconRobotics):
    def __init__(self, device, **kwargs):
        # Initialize attributes with defaults before super().__init__ to prevent a race condition.
        self.joycon_stick_v_0 = 2000
        self.joycon_stick_h_0 = 2000

        super().__init__(device, **kwargs)
        
        if self.joycon.is_right():
            self.joycon_stick_v_0 = 1900
            self.joycon_stick_h_0 = 2100
        else:  # left Joy-Con
            self.joycon_stick_v_0 = 2300
            self.joycon_stick_h_0 = 2000
        
        self.gripper_speed = 0.4
        self.gripper_direction = 1
        self.gripper_min = 0
        self.gripper_max = 90
        self.last_gripper_button_state = 0
    
    def common_update(self):
        speed_scale = 0.0003
        
        orientation_rad = self.get_orientation()
        roll, pitch, yaw = orientation_rad

        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + self.joycon_stick_v_0:
            self.position[0] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        elif joycon_stick_v < self.joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + self.joycon_stick_h_0:
            self.position[1] += speed_scale * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < self.joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += speed_scale * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
        
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == 'plus' and status == 1) or (self.joycon.is_left() and event_type == 'minus' and status == 1):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            else: 
                self.reset_button = 0
        
        gripper_button_pressed = False
        if self.joycon.is_right():
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1
        
        if gripper_button_pressed and self.last_gripper_button_state == 0:
            self.gripper_direction *= -1
        
        self.last_gripper_button_state = gripper_button_pressed
        
        if gripper_button_pressed:
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            if new_gripper_state >= self.gripper_min and new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state

        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0
        
        return self.position, self.gripper_state, self.button_control

# These classes and functions are copied from examples/7_xlerobot_teleop_joycon.py
from lerobot.model.SO101Robot import SO101Kinematics

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class SimpleTeleopArm:
    def __init__(self, joint_map, initial_obs, kinematics, home_pose, prefix="right", kp=1):
        self.home_pose = home_pose
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics
        
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        try:
            # Use forward kinematics to get the initial EE position from the initial joint angles.
            initial_x, initial_y = self.kinematics.forward_kinematics(
                self.joint_positions["shoulder_lift"],
                self.joint_positions["elbow_flex"]
            )
            self.current_x = initial_x
            self.current_y = initial_y
        except Exception as e:
            # Fallback to hardcoded values if FK fails.
            print(f"[{self.prefix}] FK failed on init: {e}. Using default EE position.")
            self.current_x = 0.1629
            self.current_y = 0.1131

        self.pitch = 0.0
        
        self.degree_step = 2
        self.xy_step = 0.005
        
        self.target_positions = self.joint_positions.copy()
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        self.target_positions = self.zero_pos.copy()
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        self.target_positions["wrist_flex"] = 0.0
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon_pose, gripper_state):
        # Calculate relative pose by subtracting the initial home pose.
        rel_pose = [joycon_pose[i] - self.home_pose[i] for i in range(len(joycon_pose))]
        x, y, z, roll_, pitch_, yaw = rel_pose

        # Deadzone on the relative pose: if the joycon is not moving, do nothing.
        pose_magnitudes = [abs(v) for v in [x, y, z, roll_, pitch_, yaw]]
        if all(m < 0.01 for m in pose_magnitudes):
            self.target_positions["gripper"] = gripper_state
            return

        pitch = -pitch_ * 60
        current_x = 0.1629 + x
        current_y = 0.1131 + z
        roll = roll_ * 45

        y_scale = 250.0
        self.target_positions["shoulder_pan"] = y * y_scale
        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(current_x, current_y)
            self.target_positions["shoulder_lift"] = joint2_target
            self.target_positions["elbow_flex"] = joint3_target
        except Exception as e:
            print(f"[{self.prefix}] IK failed: {e}")

        # Direct mapping for wrist control
        self.target_positions["wrist_flex"] = pitch
        self.target_positions["wrist_roll"] = roll
        self.target_positions["gripper"] = gripper_state

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 2  # Move 2 degrees each time
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        print("[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon):
        """Handle left Joy-Con directional pad input to control head motors"""
        # Get left Joy-Con directional pad state
        button_up = joycon.joycon.get_button_up()      # Up: head_motor_1+
        button_down = joycon.joycon.get_button_down()  # Down: head_motor_1-
        button_left = joycon.joycon.get_button_left()  # Left: head_motor_2+
        button_right = joycon.joycon.get_button_right() # Right: head_motor_2-
        
        if button_up == 1:
            self.target_positions["head_motor_2"] -= self.degree_step
        if button_down == 1:
            self.target_positions["head_motor_2"] += self.degree_step
        if button_left == 1:
            self.target_positions["head_motor_1"] -= self.degree_step
        if button_right == 1:
            self.target_positions["head_motor_1"] += self.degree_step

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

# Base speed control parameters - adjustable slopes
BASE_ACCELERATION_RATE = 2.0  # acceleration slope (speed/second)
BASE_DECELERATION_RATE = 2.5  # deceleration slope (speed/second)
BASE_MAX_SPEED = 3.0          # maximum speed multiplier

def get_joycon_base_action(joycon, robot):
    """
    Get base control commands from Joy-Con
    X: forward, B: backward, Y: left turn, A: right turn
    """
    # Get button states
    button_x = joycon.joycon.get_button_x()  # forward
    button_b = joycon.joycon.get_button_b()  # backward
    button_y = joycon.joycon.get_button_y()  # left turn
    button_a = joycon.joycon.get_button_a()  # right turn
    
    # Build key set (simulate keyboard input)
    pressed_keys = set()
    
    if button_x == 1:
        pressed_keys.add('k')  # forward
    if button_b == 1:
        pressed_keys.add('i')  # backward
    if button_y == 1:
        pressed_keys.add('u')  # left turn
    if button_a == 1:
        pressed_keys.add('o')  # right turn
    
    # Convert to numpy array and get base action
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}
    
    return base_action

def get_joycon_speed_control(joycon, state):
    """
    Get speed control from Joy-Con - linear acceleration and deceleration
    Linearly accelerate to maximum speed when holding any base control button, linearly decelerate to 0 when released
    """
    current_time = time.time()
    dt = current_time - state["last_update_time"]
    state["last_update_time"] = current_time
    
    # Check if any base control buttons are pressed
    button_x = joycon.joycon.get_button_x()  # forward
    button_b = joycon.joycon.get_button_b()  # backward
    button_y = joycon.joycon.get_button_y()  # left turn
    button_a = joycon.joycon.get_button_a()  # right turn
    
    any_base_button_pressed = any([button_x, button_b, button_y, button_a])
    
    if any_base_button_pressed:
        # Button pressed - accelerate
        if not state["is_accelerating"]:
            state["is_accelerating"] = True
        
        # Linear acceleration
        state["current_base_speed"] += BASE_ACCELERATION_RATE * dt
        state["current_base_speed"] = min(state["current_base_speed"], BASE_MAX_SPEED)
        
    else:
        # No button pressed - decelerate
        if state["is_accelerating"]:
            state["is_accelerating"] = False
        
        # Linear deceleration
        state["current_base_speed"] -= BASE_DECELERATION_RATE * dt
        state["current_base_speed"] = max(state["current_base_speed"], 0.0)
    
    return state["current_base_speed"]


import rerun as rr

robot.connect()

_init_rerun(session_name="xlerobot_record")

log_say("Initializing Joy-Cons...")
joycon_right = FixedAxesJoyconRobotics("right", dof_speed=[2, 2, 2, 1, 1, 1])
joycon_left = FixedAxesJoyconRobotics("left", dof_speed=[2, 2, 2, 1, 1, 1])
log_say("Joy-Cons initialized.")

log_say("Capturing initial Joy-Con poses as home... Keep them stationary.")
home_pose_right, _, _ = joycon_right.get_control()
home_pose_left, _, _ = joycon_left.get_control()
log_say("Home poses captured.")

obs = robot.get_observation()
kin_left = SO101Kinematics()
kin_right = SO101Kinematics()
left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, kin_left, home_pose_left, prefix="left")
right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, kin_right, home_pose_right, prefix="right")
head_control = SimpleHeadControl(obs)

#left_arm.move_to_zero_position(robot)
#right_arm.move_to_zero_position(robot)
#head_control.move_to_zero_position(robot)

base_speed_control_state = {
    "current_base_speed": 0.0,
    "last_update_time": time.time(),
    "is_accelerating": False,
}

for episode_idx in range(NUM_EPISODES):
    log_say(f"Recording episode {episode_idx}")
    start_episode_t = time.perf_counter()
    
    while time.perf_counter() - start_episode_t < EPISODE_TIME_SEC:
        start_loop_t = time.perf_counter()

        pose_right, gripper_right, control_button_right = joycon_right.get_control()
        pose_left, gripper_left, control_button_left = joycon_left.get_control()


        if control_button_right == 8:  # reset button
            log_say("Resetting to zero position.")
            right_arm.move_to_zero_position(robot)
            left_arm.move_to_zero_position(robot)
            head_control.move_to_zero_position(robot)
            continue

        right_arm.handle_joycon_input(pose_right, gripper_right)
        left_arm.handle_joycon_input(pose_left, gripper_left)
        head_control.handle_joycon_input(joycon_left)

        right_action = right_arm.p_control_action(robot)
        left_action = left_arm.p_control_action(robot)
        head_action = head_control.p_control_action(robot)
        
        base_action = get_joycon_base_action(joycon_right, robot)
        speed_multiplier = get_joycon_speed_control(joycon_right, base_speed_control_state)
        
        if base_action:
            for key in base_action:
                if 'vel' in key or 'velocity' in key:  
                    base_action[key] *= speed_multiplier
        
        action = {**left_action, **right_action, **head_action, **base_action}
        
        sent_action = robot.send_action(action)
        observation = robot.get_observation()

        log_rerun_data(observation, sent_action)

        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
        frame = {**observation_frame, **action_frame}
        frame["task"] = TASK_DESCRIPTION
        dataset.add_frame(frame)

        busy_wait(max(0, 1 / FPS - (time.perf_counter() - start_loop_t)))

    log_say(f"Episode {episode_idx} finished.")
    dataset.save_episode()

log_say("Finished recording all episodes.")
dataset.push_to_hub()

joycon_right.disconnect()
joycon_left.disconnect()
robot.disconnect()
