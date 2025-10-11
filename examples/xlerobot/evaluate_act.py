from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.record import record_loop
from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

import time


def move_robot_to_initial_position(robot):
    """
    Move robot to initial position for evaluation.
    Modify the joint angles here to match your desired starting pose.
    """
    # TODO: 修改这些值以匹配你想要的起始位姿
    initial_action = {
        # 左臂关节角度
        "left_arm_shoulder_pan.pos": 0.15,
        "left_arm_shoulder_lift.pos": -99.14,
        "left_arm_elbow_flex.pos": 78.16,  
        "left_arm_wrist_flex.pos": 41.65,
        "left_arm_wrist_roll.pos": 0.14,
        "left_arm_gripper.pos": 1.44,  
        
        # 右臂关节角度
        "right_arm_shoulder_pan.pos": -0.14,
        "right_arm_shoulder_lift.pos": 36.03,
        "right_arm_elbow_flex.pos": 24.81,  
        "right_arm_wrist_flex.pos": -47.72,
        "right_arm_wrist_roll.pos": 0.24,
        "right_arm_gripper.pos": 1.64,  
        
        # 头部位置
        "head_motor_1.pos": -0.29,   # yaw (左右转)
        "head_motor_2.pos": 55.0,  # pitch (上下俯仰)
        
        # 底盘速度 (重置时设为0)
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }
    
    robot.send_action(initial_action)
    log_say("Robot moved to initial position")

    time.sleep(2)  # 等待2秒以确保机器人到达位置

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 20
TASK_DESCRIPTION = "Pick up the lemon and place in the pink cup"

# Create the robot and teleoperator configurations
robot_config = XLerobotClientConfig(remote_ip="localhost", id="xlerobot")
robot = XLerobotClient(robot_config)

# TODO
policy = ACTPolicy.from_pretrained("zonglin1104/pick_cherry_and_place_act_all")

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="Keith-Luo/pick_cherry_and_place_act_all_10.11",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()

_init_rerun(session_name="recording")

listener, events = init_keyboard_listener()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

# Move robot to initial position before starting
move_robot_to_initial_position(robot)

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Running inference, recording eval episode {recorded_episodes} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Logic for reset env
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        # Move robot back to initial position
        move_robot_to_initial_position(robot)
        # Wait for manual environment reset if needed
        # record_loop(
        #     robot=robot,
        #     events=events,
        #     fps=FPS,
        #     control_time_s=EPISODE_TIME_SEC,
        #     single_task=TASK_DESCRIPTION,
        #     display_data=True,
        # )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    recorded_episodes += 1

# Upload to hub and clean up
# dataset.push_to_hub()

robot.disconnect()
listener.stop()
