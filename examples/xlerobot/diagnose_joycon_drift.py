#!/usr/bin/env python
"""
诊断 Joy-Con 漂移问题的脚本
实时打印左右 Joy-Con 的 position 和姿态数据，帮助定位漂移根源
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from lerobot.teleoperators.joycon import FixedAxesJoyconRobotics

def print_joycon_status(joycon_right, joycon_left, iteration):
    """打印两个 Joy-Con 的当前状态"""
    print(f"\n{'='*80}")
    print(f"Iteration: {iteration}")
    print(f"{'='*80}")
    
    if joycon_right:
        pose_right, gripper_right, control_right = joycon_right.get_control()
        print(f"\n[RIGHT Joy-Con]")
        print(f"  Position (x, y, z):     {pose_right[0]:8.5f}, {pose_right[1]:8.5f}, {pose_right[2]:8.5f}")
        print(f"  Orientation (r, p, y):  {pose_right[3]:8.5f}, {pose_right[4]:8.5f}, {pose_right[5]:8.5f}")
        print(f"  Internal position attr: {joycon_right.position}")
        print(f"  Gripper state:          {gripper_right:.3f}")
        
        # 获取摇杆原始读数
        stick_v = joycon_right.joycon.get_stick_right_vertical()
        stick_h = joycon_right.joycon.get_stick_right_horizontal()
        print(f"  Stick (V, H):           {stick_v}, {stick_h}")
        print(f"  Stick center (V, H):    {joycon_right.joycon_stick_v_0}, {joycon_right.joycon_stick_h_0}")
    
    if joycon_left:
        pose_left, gripper_left, control_left = joycon_left.get_control()
        print(f"\n[LEFT Joy-Con]")
        print(f"  Position (x, y, z):     {pose_left[0]:8.5f}, {pose_left[1]:8.5f}, {pose_left[2]:8.5f}")
        print(f"  Orientation (r, p, y):  {pose_left[3]:8.5f}, {pose_left[4]:8.5f}, {pose_left[5]:8.5f}")
        print(f"  Internal position attr: {joycon_left.position}")
        print(f"  Gripper state:          {gripper_left:.3f}")
        
        # 获取摇杆原始读数
        stick_v = joycon_left.joycon.get_stick_left_vertical()
        stick_h = joycon_left.joycon.get_stick_left_horizontal()
        print(f"  Stick (V, H):           {stick_v}, {stick_h}")
        print(f"  Stick center (V, H):    {joycon_left.joycon_stick_v_0}, {joycon_left.joycon_stick_h_0}")

    # 计算 position 差异
    if joycon_right and joycon_left:
        print(f"\n[Position Drift Analysis]")
        dx = pose_left[0] - pose_right[0]
        dy = pose_left[1] - pose_right[1]
        dz = pose_left[2] - pose_right[2]
        print(f"  Left - Right Delta:     dx={dx:8.5f}, dy={dy:8.5f}, dz={dz:8.5f}")

def main():
    print("="*80)
    print("Joy-Con 漂移诊断工具")
    print("="*80)
    print("\n请按照以下步骤操作：")
    print("1. 将左右两个 Joy-Con 静止放在桌面上")
    print("2. 不要移动或触碰 Joy-Con")
    print("3. 观察 position (x, y, z) 是否会随时间变化")
    print("4. 特别注意左 Joy-Con 的 x 值是否在往负方向漂移")
    print("\n按 Ctrl+C 结束测试\n")
    
    # 初始化 Joy-Con
    print("初始化右 Joy-Con...")
    joycon_right = FixedAxesJoyconRobotics(
        "right",
        dof_speed=[2, 2, 2, 1, 1, 1]
    )
    print("✓ 右 Joy-Con 已连接\n")
    
    print("初始化左 Joy-Con...")
    joycon_left = FixedAxesJoyconRobotics(
        "left",
        dof_speed=[2, 2, 2, 1, 1, 1]
    )
    print("✓ 左 Joy-Con 已连接\n")
    
    time.sleep(1)
    
    print("开始监测... (更新间隔: 1秒)")
    print("="*80)
    
    try:
        iteration = 0
        start_time = time.time()
        
        # 记录初始位置
        initial_pose_right, _, _ = joycon_right.get_control()
        initial_pose_left, _, _ = joycon_left.get_control()
        
        print(f"\n[初始位置]")
        print(f"Right: x={initial_pose_right[0]:.5f}, y={initial_pose_right[1]:.5f}, z={initial_pose_right[2]:.5f}")
        print(f"Left:  x={initial_pose_left[0]:.5f}, y={initial_pose_left[1]:.5f}, z={initial_pose_left[2]:.5f}")
        
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            
            print_joycon_status(joycon_right, joycon_left, iteration)
            
            # 计算从初始位置的漂移
            current_pose_right, _, _ = joycon_right.get_control()
            current_pose_left, _, _ = joycon_left.get_control()
            
            drift_right_x = current_pose_right[0] - initial_pose_right[0]
            drift_left_x = current_pose_left[0] - initial_pose_left[0]
            
            print(f"\n[累积漂移 (从启动开始)]")
            print(f"  Elapsed time:           {elapsed:.1f} seconds")
            print(f"  Right X drift:          {drift_right_x:+.5f}")
            print(f"  Left X drift:           {drift_left_x:+.5f}")
            print(f"  Drift difference:       {abs(drift_left_x) - abs(drift_right_x):+.5f}")
            
            if abs(drift_left_x) > 0.01:
                print(f"\n⚠️  警告: 左 Joy-Con X 轴漂移超过 0.01!")
            
            time.sleep(1)  # 每秒更新一次
            
    except KeyboardInterrupt:
        print("\n\n测试结束。")
    finally:
        print("\n断开 Joy-Con 连接...")
        joycon_right.disconnect()
        joycon_left.disconnect()
        print("完成。")

if __name__ == "__main__":
    main()
