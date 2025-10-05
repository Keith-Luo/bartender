import time
import numpy as np
import math

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop, build_dataset_frame
from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.teleoperators.joycon import JoyConTeleop, JoyconTeleopConfig, FixedAxesJoyconRobotics
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def move_robot_to_zero_position(robot, teleop: JoyConTeleop, head_pitch_angle: float = -30.0):
    """
    Reset robot to zero position by updating teleop position and sending action.
    Resets both position (x,y,z) and orientation (roll,pitch,yaw) to zero.
    
    Args:
        robot: Robot instance
        teleop: Teleop instance
        head_pitch_angle: Head pitch angle in degrees (negative = look down)
                         head_motor_2 controls pitch (up/down)
                         Typical range: -60 to +60 degrees
                         Default -30 means looking down at workspace
    """
    # Reset teleop positions to their offsets (zero position)
    if teleop.joycon_left is not None:
        teleop.joycon_left.position = teleop.joycon_left.offset_position_m.copy()
        # Reset orientation (roll, pitch, yaw) to zero
        teleop.joycon_left.orientation_rad = teleop.joycon_left.offset_euler_rad.copy()
        
    if teleop.joycon_right is not None:
        teleop.joycon_right.position = teleop.joycon_right.offset_position_m.copy()
        # Reset orientation (roll, pitch, yaw) to zero
        teleop.joycon_right.orientation_rad = teleop.joycon_right.offset_euler_rad.copy()
    
    # Set head zero position and update target positions
    # This ensures the head position is maintained in subsequent get_action() calls
    teleop.set_head_zero_position(head_pitch=head_pitch_angle, head_yaw=0.0)
    
    # Get the action from teleop (which now reflects zero position with correct head angle)
    action = teleop.get_action()
    
    # Remove episode control signal if present
    if "_episode_control" in action:
        del action["_episode_control"]
    
    # Send the zero position action to robot
    robot.send_action(action)
    log_say(f"Robot moved to zero position (head pitch: {head_pitch_angle}°)")
    


def record_loop_with_joycon_episode_control(
    robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop: JoyConTeleop | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    """
    Custom record loop that handles Joy-Con episode control signals.
    Based on the original record_loop but modified to process episode control from Joy-Con.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    dt = 1 / fps
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        # Get action from Joy-Con teleop
        action = teleop.get_action()
        
        # Check for episode control signals from Joy-Con
        if "_episode_control" in action:
            episode_control = action["_episode_control"]
            # Remove the control signal from action before sending to robot
            del action["_episode_control"]
            
            if episode_control == "next":
                # Right Joy-Con "+" button: move to next episode
                log_say("Next episode requested via Joy-Con '+' button")
                break
            elif episode_control == "restart":
                # Left Joy-Con "-" button: restart/rerecord current episode
                log_say("Restart episode requested via Joy-Con '-' button")
                events["rerecord_episode"] = True
                events["exit_early"] = True
                break

        # Send action to robot (action can be clipped using max_relative_target)
        sent_action = robot.send_action(action)

        if dataset is not None:
            # Use original dataset processing logic
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            # Use original display logic
            log_rerun_data(observation, action)

        # Use original timing control
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
            
        timestamp = time.perf_counter() - start_episode_t

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 50
RESET_TIME_SEC = 5
# TODO
TASK_DESCRIPTION = "pick the wine bottle and pour wine into the cup"

# 头部俯仰角度配置 (Head pitch angle configuration)
# head_motor_2 控制俯仰 (pitch: up/down tilt)
# 负值 = 往下看 (look down), 正值 = 往上看 (look up)
# 推荐范围: -60° 到 +60°
# 示例值:
#   -30°: 轻微往下看工作台 (slightly look down at workspace)
#   -45°: 明显往下看工作台 (clearly look down at workspace)
#   -60°: 大角度往下看 (steep look down)
#   0°:   水平看 (look straight)
HEAD_PITCH_ANGLE = 55.0  # 默认往下看 30 度，可根据实际工作台高度调整

# 选择要使用的相机（注释掉不需要的相机）
camera_config = {
    "left_arm_wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "right_arm_wrist": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=30),  # 注释掉不录制
    "head": OpenCVCameraConfig(index_or_path=8, width=640, height=480, fps=30),
}

robot_config = XLerobotClientConfig(remote_ip="127.0.0.1", id="xlerobot", cameras=camera_config)
robot = XLerobotClient(robot_config)

# Configure Joy-Con teleoperator
joycon_config = JoyconTeleopConfig(
    device="both",  # Use both left and right Joy-Con
    use_arm_control=True,
    use_gripper=True,
    use_base_control=True,
    use_head_control=True,
    arm_selection="both",  # "left", "right", "both" - 控制哪个手臂
)

# 单臂控制示例配置（如果只要录制左臂数据）:
# joycon_config = JoyconTeleopConfig(
#     device="both",
#     use_arm_control=True,
#     use_gripper=True,
#     use_base_control=False,  # 单臂时可能不需要底盘控制
#     use_head_control=False,
#     arm_selection="left",  # 只控制左臂
# )

# 或者只控制右臂:
# joycon_config = JoyconTeleopConfig(
#     device="both",
#     use_arm_control=True, 
#     use_gripper=True,
#     use_base_control=False,
#     use_head_control=False,
#     arm_selection="right",  # 只控制右臂
# )
joycon_teleop = JoyConTeleop(joycon_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="Keith-Luo/pick_wine_bottle_and_pour_5",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# To connect you already should have this script running on XLerobot: `python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=xlerobot`
robot.connect()
joycon_teleop.connect()

_init_rerun(session_name="xlerobot_record")

listener, events = init_keyboard_listener()

if not robot.is_connected or not joycon_teleop.is_connected:
    raise ValueError("Robot or Joy-Con is not connected!")

# Move to zero position at the start
move_robot_to_zero_position(robot, joycon_teleop, HEAD_PITCH_ANGLE)

time.sleep(2)

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {recorded_episodes}")

    # Run the record loop
    record_loop_with_joycon_episode_control(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=joycon_teleop,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Logic for reset env
    # 重置环境的逻辑
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        # log_say("Reset the environment")
        move_robot_to_zero_position(robot, joycon_teleop, HEAD_PITCH_ANGLE)

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    # Save the episode
    dataset.save_episode()
    recorded_episodes += 1

# Upload to hub and clean up
# dataset.push_to_hub()

robot.disconnect()
joycon_teleop.disconnect()
listener.stop()
