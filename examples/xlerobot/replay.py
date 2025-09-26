import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

EPISODE_IDX = 0

# --- MODIFY THIS --- #
# TODO: Replace with the actual IP of the robot
robot_config = XLerobotClientConfig(remote_ip="localhost", id="xlerobot")
robot = XLerobotClient(robot_config)

# --- MODIFY THIS --- #
# Make sure this is the same dataset repo id used in record.py
dataset = LeRobotDataset("<hf_username>/xlerobot_dataset", episodes=[EPISODE_IDX])
actions = dataset.hf_dataset.select_columns("action")

robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

log_say(f"Replaying episode {EPISODE_IDX}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }
    robot.send_action(action)

    busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()
