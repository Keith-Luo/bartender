#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("joycon")
@dataclass
class JoyconTeleopConfig(TeleoperatorConfig):
    mock: bool = False
    
    # JoyCon specific configurations
    device: str = "both"  # "right", "left", or "both"
    use_base_control: bool = True  # Whether to include base movement control
    use_gripper: bool = True  # Whether to include gripper control
    use_arm_control: bool = True  # Whether to include arm control
    use_head_control: bool = False  # Whether to include head control
    
    # Arm control selection - NEW
    arm_selection: str = "both"  # "left", "right", "both" - which arms to control
    
    # Control parameters
    speed_scale: float = 0.0003  # Speed scaling factor for position control
    gripper_speed: float = 0.4  # Gripper open/close speed
    gripper_min: float = 0.0  # Minimum gripper angle
    gripper_max: float = 90.0  # Maximum gripper angle
    
    # Base control parameters
    base_acceleration_rate: float = 2.0  # Base acceleration rate
    base_deceleration_rate: float = 2.5  # Base deceleration rate  
    base_max_speed: float = 3.0  # Maximum base speed multiplier

