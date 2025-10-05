#!/usr/bin/env python
"""
更底层的 Joy-Con 诊断脚本
直接使用 joyconrobotics 库，不经过 LeRobot 的封装
监测原始的 IMU 读数和 position 累积过程
"""

import time
import sys
sys.path.insert(0, '/Users/luokaijun/Desktop/hackathon/household_robotics/joycon-robotics')

from joyconrobotics import JoyconRobotics

def main():
    print("="*80)
    print("Joy-Con 底层漂移诊断工具 (直接使用 joyconrobotics 库)")
    print("="*80)
    print("\n请将左右两个 Joy-Con 静止放在桌面上，不要移动")
    print("观察 position 和方向向量是否会持续变化\n")
    print("按 Ctrl+C 结束测试\n")
    
    # 初始化（使用和 FixedAxesJoyconRobotics 相同的参数）
    print("初始化右 Joy-Con...")
    joycon_right = JoyconRobotics(
        device="right",
        dof_speed=[2, 2, 2, 1, 1, 1],
        without_rest_init=False  # 进行初始校准
    )
    print("✓ 右 Joy-Con 已连接\n")
    
    print("初始化左 Joy-Con...")
    joycon_left = JoyconRobotics(
        device="left",
        dof_speed=[2, 2, 2, 1, 1, 1],
        without_rest_init=False  # 进行初始校准
    )
    print("✓ 左 Joy-Con 已连接\n")
    
    time.sleep(2)
    
    print("开始监测... (更新间隔: 0.5秒)")
    print("="*80)
    
    try:
        iteration = 0
        start_time = time.time()
        
        # 记录初始位置
        initial_right, _, _ = joycon_right.get_control()
        initial_left, _, _ = joycon_left.get_control()
        
        print(f"\n[初始位置]")
        print(f"Right: x={initial_right[0]:.6f}, y={initial_right[1]:.6f}, z={initial_right[2]:.6f}")
        print(f"Left:  x={initial_left[0]:.6f}, y={initial_left[1]:.6f}, z={initial_left[2]:.6f}")
        
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            
            # 获取当前状态
            pose_right, gripper_right, _ = joycon_right.get_control()
            pose_left, gripper_left, _ = joycon_left.get_control()
            
            print(f"\n{'='*80}")
            print(f"Iteration: {iteration:3d}  |  Time: {elapsed:6.1f}s")
            print(f"{'='*80}")
            
            # 右 Joy-Con
            print(f"\n[RIGHT Joy-Con]")
            print(f"  Position:             [{pose_right[0]:8.6f}, {pose_right[1]:8.6f}, {pose_right[2]:8.6f}]")
            print(f"  Orientation (rad):    [{pose_right[3]:8.6f}, {pose_right[4]:8.6f}, {pose_right[5]:8.6f}]")
            print(f"  Internal .position:   {joycon_right.position}")
            print(f"  Direction vector:     {joycon_right.direction_vector}")
            print(f"  Direction vector R:   {joycon_right.direction_vector_right}")
            
            # 获取摇杆原始值
            stick_v_r = joycon_right.joycon.get_stick_right_vertical()
            stick_h_r = joycon_right.joycon.get_stick_right_horizontal()
            print(f"  Stick raw (V/H):      {stick_v_r:4d} / {stick_h_r:4d}")
            print(f"  Stick center (V/H):   {joycon_right.joycon_stick_v_0 if hasattr(joycon_right, 'joycon_stick_v_0') else 'N/A'}")
            
            # 左 Joy-Con
            print(f"\n[LEFT Joy-Con]")
            print(f"  Position:             [{pose_left[0]:8.6f}, {pose_left[1]:8.6f}, {pose_left[2]:8.6f}]")
            print(f"  Orientation (rad):    [{pose_left[3]:8.6f}, {pose_left[4]:8.6f}, {pose_left[5]:8.6f}]")
            print(f"  Internal .position:   {joycon_left.position}")
            print(f"  Direction vector:     {joycon_left.direction_vector}")
            print(f"  Direction vector R:   {joycon_left.direction_vector_right}")
            
            # 获取摇杆原始值
            stick_v_l = joycon_left.joycon.get_stick_left_vertical()
            stick_h_l = joycon_left.joycon.get_stick_left_horizontal()
            print(f"  Stick raw (V/H):      {stick_v_l:4d} / {stick_h_l:4d}")
            print(f"  Stick center (V/H):   {joycon_left.joycon_stick_v_0 if hasattr(joycon_left, 'joycon_stick_v_0') else 'N/A'}")
            
            # 漂移分析
            drift_right_x = pose_right[0] - initial_right[0]
            drift_left_x = pose_left[0] - initial_left[0]
            
            print(f"\n[漂移分析]")
            print(f"  Right X drift:        {drift_right_x:+.6f}")
            print(f"  Left X drift:         {drift_left_x:+.6f}")
            print(f"  Difference:           {drift_left_x - drift_right_x:+.6f}")
            
            # 检测异常
            if abs(drift_left_x) > 0.01:
                print(f"\n  ⚠️  左 Joy-Con X 漂移超过 0.01!")
            if abs(drift_right_x) > 0.01:
                print(f"\n  ⚠️  右 Joy-Con X 漂移超过 0.01!")
            
            # 检查方向向量是否异常
            if joycon_left.direction_vector:
                dir_mag = (joycon_left.direction_vector[0]**2 + 
                          joycon_left.direction_vector[1]**2 + 
                          joycon_left.direction_vector[2]**2) ** 0.5
                print(f"\n  Left direction vector magnitude: {dir_mag:.6f}")
                if abs(dir_mag - 1.0) > 0.01:
                    print(f"  ⚠️  方向向量长度异常 (应该接近1.0)")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n测试结束。")
    finally:
        print("\n断开 Joy-Con 连接...")
        joycon_right.disconnect()
        joycon_left.disconnect()
        print("完成。")

if __name__ == "__main__":
    main()
