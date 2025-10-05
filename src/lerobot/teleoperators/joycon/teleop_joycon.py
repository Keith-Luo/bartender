#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import math
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from .configuration_joycon import JoyconTeleopConfig

logger = logging.getLogger(__name__)

try:
    from joyconrobotics import JoyconRobotics
    JOYCON_AVAILABLE = True
except ImportError:
    JOYCON_AVAILABLE = False
    logger.warning("joyconrobotics not available. Install it with 'pip install joyconrobotics'")

# Import SO101 Kinematics for inverse kinematics
try:
    from lerobot.model.SO101Robot import SO101Kinematics
    IK_AVAILABLE = True
except ImportError:
    IK_AVAILABLE = False
    logger.warning("SO101Kinematics not available. IK will use simplified mapping.")


class FixedAxesJoyconRobotics(JoyconRobotics):
    """Extended JoyconRobotics class with fixed axes and additional features for teleop"""
    
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        
        # Set different center values for left and right Joy-Cons
        if self.joycon.is_right():
            self.joycon_stick_v_0 = 1900
            self.joycon_stick_h_0 = 2100
        else:  # left Joy-Con
            self.joycon_stick_v_0 = 1900
            self.joycon_stick_h_0 = 2000
        
        # Gripper control related variables
        self.gripper_speed = 0.4  # Gripper open/close speed (degrees/frame)
        self.gripper_direction = 1  # 1 means open, -1 means close
        self.gripper_min = 0  # Minimum angle (fully closed)
        self.gripper_max = 90  # Maximum angle (fully open)
        self.last_gripper_button_state = 0  # Record previous frame button state for detecting press events
    
    def common_update(self):
        """Modified update logic: joystick only controls fixed axes"""
        speed_scale = 0.0003
        
        # Get current orientation data to print pitch
        orientation_rad = self.get_orientation()
        roll, pitch, yaw = orientation_rad

        # Vertical joystick: controls X and Z axes (forward/backward)
        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + self.joycon_stick_v_0:
            self.position[0] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range * self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range * self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        elif joycon_stick_v < self.joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range * self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range * self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        
        # Horizontal joystick: only controls Y axis (left/right)
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + self.joycon_stick_h_0:
            self.position[1] += speed_scale * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < self.joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += speed_scale * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        
        # Z-axis button control
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        # Home button reset logic (simplified version)
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
        
        # Button events handling
        for event_type, status in self.button.events():
            if self.joycon.is_right() and event_type == 'plus' and status == 1:
                # 右手柄 "+" 键：结束当前 episode 并进入下一条
                self.reset_button = 1
                # self.reset_joycon()
            elif self.joycon.is_left() and event_type == 'minus' and status == 1:
                # 左手柄 "-" 键：重录当前 episode
                self.restart_episode_button = status
            elif self.joycon.is_right() and event_type == 'a':
                # 下一条 episode
                # TODO：似乎按下 a 才是下一条 episode？
                self.next_episode_button = status
            # elif self.joycon.is_right() and event_type == 'y':
            #     # 重新录制这一条 episode（暂时保留，可能不使用）
            #     self.restart_episode_button = status
            else: 
                self.reset_button = 0
        
        # Gripper button state detection and direction control
        gripper_button_pressed = False
        if self.joycon.is_right():
            # Right Joy-Con uses ZR button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            # Left Joy-Con uses ZL button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1
        
        # Detect button press events (from 0 to 1) to change direction
        if gripper_button_pressed and self.last_gripper_button_state == 0:
            # Button just pressed, change direction
            self.gripper_direction *= -1
            logger.info(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")
        
        # Update button state record
        self.last_gripper_button_state = gripper_button_pressed
        
        # Linear control of gripper open/close when holding gripper button
        if gripper_button_pressed:
            # Check if exceeding limits
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            
            # If exceeding limits, stop moving
            if new_gripper_state >= self.gripper_min and new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state

        # Button control state
        if self.joycon.is_right():
            if self.next_episode_button == 1:
                # 按下 a
                self.button_control = 1
            elif self.restart_episode_button == 1:
                # 按下 y
                self.button_control = -1
            elif self.reset_button == 1:
                # 按下 + 号
                logger.info("debug1")
                self.button_control = 8
            else:
                self.button_control = 0
        elif self.joycon.is_left():
            # 左手柄的控制
            if self.restart_episode_button == 1:
                # 按下 - 号：重录当前 episode
                logger.info("Left Joy-Con minus button pressed: restart episode")
                self.button_control = -2  # 使用 -2 作为左手柄重录信号
            else:
                self.button_control = 0
        
        return self.position, self.gripper_state, self.button_control


class JoyConTeleop(Teleoperator):
    """
    Teleop class to use Nintendo Joy-Con controllers for robot control.
    Supports arm control, base movement, gripper control, and special episode control buttons.
    """

    config_class = JoyconTeleopConfig
    name = "joycon"

    def __init__(self, config: JoyconTeleopConfig):
        super().__init__(config)
        self.config = config
        
        self.joycon_right = None
        self.joycon_left = None
        self._is_connected = False
        
        # Base control state variables
        self.current_base_speed = 0.0
        self.last_update_time = time.time()
        self.is_accelerating = False
        
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Head control state - maintain target positions like SimpleHeadControl
        # This ensures head position is maintained across get_action() calls
        self.head_target_positions = {
            "head_motor_1": 0.0,  # Will be set by set_head_zero_position()
            "head_motor_2": 0.0,  # Will be set by set_head_zero_position()
        }
        self.head_zero_pos = {
            "head_motor_1": 0.0,   # Default yaw
            "head_motor_2": -55.0,  # Default pitch: look down at workspace
        }
        self.head_degree_step = 2.0
        
        # Initialize inverse kinematics for both arms
        if IK_AVAILABLE:
            self.right_arm_kinematics = SO101Kinematics()
            self.left_arm_kinematics = SO101Kinematics()
            logger.info("Inverse kinematics initialized for both arms")
        else:
            self.right_arm_kinematics = None
            self.left_arm_kinematics = None
            logger.warning("Inverse kinematics not available, using simplified mapping")

    @property
    def action_features(self) -> dict:
        """Define the action space based on configuration"""
        # TODO：这个 action_features 是不是没用呢？可能需要 check 一下，xlerobot_client 里有封装
        features = {}
        
        if self.config.use_arm_control:
            # Right arm control
            features.update({
                "right_arm_shoulder_pan.pos": float,
                "right_arm_shoulder_lift.pos": float,
                "right_arm_elbow_flex.pos": float,
                "right_arm_wrist_flex.pos": float,
                "right_arm_wrist_roll.pos": float,
            })
            
            # Left arm control
            features.update({
                "left_arm_shoulder_pan.pos": float,
                "left_arm_shoulder_lift.pos": float,
                "left_arm_elbow_flex.pos": float,
                "left_arm_wrist_flex.pos": float,
                "left_arm_wrist_roll.pos": float,
            })
            
        if self.config.use_gripper:
            features.update({
                "right_arm_gripper.pos": float,
                "left_arm_gripper.pos": float,
            })
        
        if self.config.use_base_control:
            features.update({
                "base_x.vel": float,
                "base_y.vel": float,
                "base_angular.vel": float,
            })
            
        if self.config.use_head_control:
            features.update({
                "head_motor_1.pos": float,
                "head_motor_2.pos": float,
            })
        
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if Joy-Con controllers are connected"""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Joy-Con controllers don't require calibration"""
        return True
    
    def set_head_zero_position(self, head_pitch: float = -30.0, head_yaw: float = 0.0):
        """
        Set the head zero position (the position to return to when reset).
        Also sets current target positions to this zero position.
        
        Args:
            head_pitch: Pitch angle in degrees (negative = look down), default -30°
            head_yaw: Yaw angle in degrees, default 0°
        """
        self.head_zero_pos = {
            "head_motor_1": head_yaw,
            "head_motor_2": head_pitch,
        }
        # Set current target to zero position
        self.head_target_positions = self.head_zero_pos.copy()
        logger.info(f"Head zero position set to: pitch={head_pitch}°, yaw={head_yaw}°")

    def connect(self) -> None:
        """Connect to Joy-Con controllers"""
        if not JOYCON_AVAILABLE:
            raise ImportError("joyconrobotics not available. Install it with 'pip install joyconrobotics'")
            
        if self.is_connected:
            logger.warning("Joy-Con controllers are already connected")
            return
            
        try:
            # Initialize Joy-Con controllers based on configuration
            if self.config.device in ["right", "both"]:
                logger.info("Initializing right Joy-Con controller...")
                self.joycon_right = FixedAxesJoyconRobotics(
                    "right",
                    dof_speed=[2, 2, 2, 1, 1, 1]
                )
                logger.info("Right Joy-Con controller connected")
                
            if self.config.device in ["left", "both"]:
                logger.info("Initializing left Joy-Con controller...")
                self.joycon_left = FixedAxesJoyconRobotics(
                    "left", 
                    dof_speed=[2, 2, 2, 1, 1, 1]
                )
                logger.info("Left Joy-Con controller connected")
                
            self._is_connected = True
            logger.info("Joy-Con teleoperator connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect Joy-Con controllers: {e}")
            raise

    def get_action(self) -> dict[str, Any]:
        """Get current action from Joy-Con input"""
        if not self.is_connected:
            return {}
            
        action = {}
        
        try:
            # Get input from controllers
            if self.joycon_right is not None:
                pose_right, gripper_right, control_button_right = self.joycon_right.get_control()
            else:
                pose_right, gripper_right, control_button_right = None, 0.0, 0
                
            if self.joycon_left is not None:
                pose_left, gripper_left, control_button_left = self.joycon_left.get_control()
            else:
                pose_left, gripper_left, control_button_left = None, 0.0, 0

            # Handle arm control
            if self.config.use_arm_control:
                action.update(self._get_arm_actions(pose_right, pose_left))
                
            # Handle gripper control
            if self.config.use_gripper:
                gripper_actions = {}
                # 根据配置决定控制哪个手臂的夹爪
                if self.config.arm_selection in ["right", "both"]:
                    gripper_actions["right_arm_gripper.pos"] = gripper_right
                if self.config.arm_selection in ["left", "both"]:
                    gripper_actions["left_arm_gripper.pos"] = gripper_left
                action.update(gripper_actions)
                
            # Handle base control
            if self.config.use_base_control and self.joycon_right is not None:
                base_action = self._get_base_action(self.joycon_right)
                action.update(base_action)
            
            # Handle head control  
            if self.config.use_head_control and self.joycon_left is not None:
                head_action = self._get_head_action(self.joycon_left)
                action.update(head_action)
                
            # Handle special episode control buttons
            # Note: control_button values from joyconrobotics library:
            # Right Joy-Con:
            #   "+" button (plus) = 8 (next episode)
            #   "A" button = 1
            #   "Y" button = -1
            # Left Joy-Con:
            #   "-" button (minus) = -2 (restart episode)
            
            # Check right Joy-Con control
            if control_button_right == 8:  # next episode，按下右手柄 "+" 号键
                logger.info("Next-episode button ('+') pressed on right Joy-Con")
                logger.info("debug2")
                action["_episode_control"] = "next"
            
            # Check left Joy-Con control  
            if control_button_left == -2:  # restart episode，按下左手柄 "-" 号键
                logger.info("Restart-episode button ('-') pressed on left Joy-Con")
                action["_episode_control"] = "restart"
                
        except Exception as e:
            logger.error(f"Error getting Joy-Con action: {e}")
            
        return action

    def _get_arm_actions(self, pose_right, pose_left):
        """Convert Joy-Con poses to arm actions using inverse kinematics"""
        actions = {}
        
        # 根据配置决定控制哪个手臂
        control_right_arm = self.config.arm_selection in ["right", "both"]
        control_left_arm = self.config.arm_selection in ["left", "both"]
        
        if pose_right is not None and control_right_arm:
            x, y, z, roll_, pitch_, yaw = pose_right
            
            # Calculate pitch control - consistent with 7_xlerobot_teleop_joycon.py
            pitch = -pitch_ * 60 + 10
            
            # Set end-effector coordinates
            current_x = 0.1629 + x
            current_y = 0.1131 + z
            
            # Calculate roll
            roll = roll_ * 45
            
            # Add y value to control shoulder_pan joint
            y_scale = 250.0
            actions["right_arm_shoulder_pan.pos"] = y * y_scale
            
            # Use inverse kinematics for shoulder_lift and elbow_flex
            if self.right_arm_kinematics is not None:
                try:
                    shoulder_lift, elbow_flex = self.right_arm_kinematics.inverse_kinematics(current_x, current_y)
                    actions["right_arm_shoulder_lift.pos"] = shoulder_lift
                    actions["right_arm_elbow_flex.pos"] = elbow_flex
                    
                    # wrist_flex calculation considers the entire kinematic chain
                    actions["right_arm_wrist_flex.pos"] = -shoulder_lift - elbow_flex + pitch
                except Exception as e:
                    logger.error(f"Right arm IK failed: {e}, using simplified mapping")
                    # Fallback to simplified mapping
                    actions["right_arm_shoulder_lift.pos"] = current_x * 100
                    actions["right_arm_elbow_flex.pos"] = current_y * 100
                    actions["right_arm_wrist_flex.pos"] = pitch
            else:
                # Simplified mapping when IK is not available
                actions["right_arm_shoulder_lift.pos"] = current_x * 100
                actions["right_arm_elbow_flex.pos"] = current_y * 100
                actions["right_arm_wrist_flex.pos"] = pitch
            
            actions["right_arm_wrist_roll.pos"] = roll
            
        if pose_left is not None and control_left_arm:
            x, y, z, roll_, pitch_, yaw = pose_left
            
            # Calculate pitch control
            pitch = -pitch_ * 60 + 10
            
            # Set end-effector coordinates
            current_x = 0.1629 + x
            current_y = 0.1131 + z
            
            # Calculate roll
            roll = roll_ * 45
            
            # Add y value to control shoulder_pan joint
            y_scale = 250.0
            actions["left_arm_shoulder_pan.pos"] = y * y_scale
            
            # Use inverse kinematics for shoulder_lift and elbow_flex
            if self.left_arm_kinematics is not None:
                try:
                    shoulder_lift, elbow_flex = self.left_arm_kinematics.inverse_kinematics(current_x, current_y)
                    actions["left_arm_shoulder_lift.pos"] = shoulder_lift
                    actions["left_arm_elbow_flex.pos"] = elbow_flex
                    
                    # wrist_flex calculation considers the entire kinematic chain
                    actions["left_arm_wrist_flex.pos"] = -shoulder_lift - elbow_flex + pitch
                except Exception as e:
                    logger.error(f"Left arm IK failed: {e}, using simplified mapping")
                    # Fallback to simplified mapping
                    actions["left_arm_shoulder_lift.pos"] = current_x * 100
                    actions["left_arm_elbow_flex.pos"] = current_y * 100
                    actions["left_arm_wrist_flex.pos"] = pitch
            else:
                # Simplified mapping when IK is not available
                actions["left_arm_shoulder_lift.pos"] = current_x * 100
                actions["left_arm_elbow_flex.pos"] = current_y * 100
                actions["left_arm_wrist_flex.pos"] = pitch
            
            actions["left_arm_wrist_roll.pos"] = roll
            
        return actions

    def _get_base_action(self, joycon):
        """Get base control commands from Joy-Con"""
        # Get button states
        button_x = joycon.joycon.get_button_x()  # forward
        button_b = joycon.joycon.get_button_b()  # backward
        button_y = joycon.joycon.get_button_y()  # left turn
        button_a = joycon.joycon.get_button_a()  # right turn
        
        # Calculate speed multiplier with acceleration/deceleration
        speed_multiplier = self._get_speed_control(joycon)
        
        base_action = {}
        
        if button_x == 1:  # forward
            base_action["base_x.vel"] = 1.0 * speed_multiplier
        elif button_b == 1:  # backward
            base_action["base_x.vel"] = -1.0 * speed_multiplier
        else:
            base_action["base_x.vel"] = 0.0
            
        if button_y == 1:  # left turn
            base_action["base_angular.vel"] = 1.0 * speed_multiplier
        elif button_a == 1:  # right turn
            base_action["base_angular.vel"] = -1.0 * speed_multiplier
        else:
            base_action["base_angular.vel"] = 0.0
            
        base_action["base_y.vel"] = 0.0  # No lateral movement
        
        return base_action

    def _get_speed_control(self, joycon):
        """Get speed control with linear acceleration and deceleration"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Check if any base control buttons are pressed
        button_x = joycon.joycon.get_button_x()
        button_b = joycon.joycon.get_button_b()
        button_y = joycon.joycon.get_button_y()
        button_a = joycon.joycon.get_button_a()
        
        any_base_button_pressed = any([button_x, button_b, button_y, button_a])
        
        if any_base_button_pressed:
            # Button pressed - accelerate
            if not self.is_accelerating:
                self.is_accelerating = True
            
            # Linear acceleration
            self.current_base_speed += self.config.base_acceleration_rate * dt
            self.current_base_speed = min(self.current_base_speed, self.config.base_max_speed)
        else:
            # No button pressed - decelerate
            if self.is_accelerating:
                self.is_accelerating = False
            
            # Linear deceleration
            self.current_base_speed -= self.config.base_deceleration_rate * dt
            self.current_base_speed = max(self.current_base_speed, 0.0)
        
        return self.current_base_speed

    def _get_head_action(self, joycon):
        """
        Handle left Joy-Con directional pad input to control head motors.
        Maintains target_positions state like SimpleHeadControl in 7_xlerobot_teleop_joycon.py.
        Returns target positions (not increments) so position is maintained across calls.
        """
        # Get left Joy-Con directional pad state
        button_up = joycon.joycon.get_button_up()      # Up: head_motor_2+ (pitch up)
        button_down = joycon.joycon.get_button_down()  # Down: head_motor_2- (pitch down)
        button_left = joycon.joycon.get_button_left()  # Left: head_motor_1+ (yaw left)
        button_right = joycon.joycon.get_button_right() # Right: head_motor_1- (yaw right)
        
        # Update target positions based on button presses (use if, not elif, for simultaneous presses)
        if button_up == 1:
            self.head_target_positions["head_motor_2"] += self.head_degree_step
        if button_down == 1:
            self.head_target_positions["head_motor_2"] -= self.head_degree_step
        if button_left == 1:
            self.head_target_positions["head_motor_1"] += self.head_degree_step
        if button_right == 1:
            self.head_target_positions["head_motor_1"] -= self.head_degree_step
        
        # Clamp to reasonable limits
        self.head_target_positions["head_motor_1"] = np.clip(
            self.head_target_positions["head_motor_1"], -60.0, 60.0
        )
        self.head_target_positions["head_motor_2"] = np.clip(
            self.head_target_positions["head_motor_2"], -60.0, 60.0
        )
        
        # Return current target positions (NOT increments or 0)
        # This ensures position is maintained even when no buttons are pressed
        return {
            "head_motor_1.pos": self.head_target_positions["head_motor_1"],
            "head_motor_2.pos": self.head_target_positions["head_motor_2"],
        }

    def disconnect(self) -> None:
        """Disconnect from Joy-Con controllers"""
        try:
            if self.joycon_right is not None:
                self.joycon_right.disconnect()
                self.joycon_right = None
                
            if self.joycon_left is not None:
                self.joycon_left.disconnect()
                self.joycon_left = None
                
            self._is_connected = False
            logger.info("Joy-Con teleoperator disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting Joy-Con controllers: {e}")

    def calibrate(self) -> None:
        """Joy-Con controllers don't require calibration"""
        pass

    def configure(self) -> None:
        """Configure Joy-Con controllers - no additional configuration needed"""
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to Joy-Con controllers - not supported"""
        pass