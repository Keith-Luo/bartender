import time
import numpy as np
import math

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop, build_dataset_frame
from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.teleoperators.joycon import JoyConTeleop, JoyconTeleopConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


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
                log_say("Next episode requested via Joy-Con '+' button")
                break
            # 移除重录逻辑
            # elif episode_control == "restart":
            #     log_say("Restart episode requested via Joy-Con")
            #     events["rerecord_episode"] = True
            #     events["exit_early"] = True
            #     break

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

NUM_EPISODES = 3
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
# TODO
TASK_DESCRIPTION = "My task description"

# 选择要使用的相机（注释掉不需要的相机）
camera_config = {
    "left_arm_wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    # "right_arm_wrist": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=30),  # 注释掉不录制
    "head": OpenCVCameraConfig(index_or_path=8, width=640, height=480, fps=30),
}

# 或者创建不同的配置
# 方案1：只录制左臂相机
# camera_config = {
#     "left_arm_wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
# }

# 方案2：只录制头部相机  
# camera_config = {
#     "head": OpenCVCameraConfig(index_or_path=8, width=640, height=480, fps=30),
# }

# 方案3：高分辨率配置
# camera_config = {
#     "left_arm_wrist": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=30),
#     "head": OpenCVCameraConfig(index_or_path=8, width=1920, height=1080, fps=30),
# }
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
    repo_id="zonglin1104/xlerobot_joycon_dataset",
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
        log_say("Reset the environment")
        record_loop_with_joycon_episode_control(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=joycon_teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    recorded_episodes += 1

# Upload to hub and clean up
dataset.push_to_hub()

robot.disconnect()
joycon_teleop.disconnect()
listener.stop()
