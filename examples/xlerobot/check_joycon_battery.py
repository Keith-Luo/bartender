#!/usr/bin/env python3
"""
Joy-Con Battery Level Monitor
检测 Joy-Con 手柄的电池电量

Usage:
    PYTHONPATH=src python examples/xlerobot/check_joycon_battery.py
"""

import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from joyconrobotics import JoyconRobotics
    JOYCON_AVAILABLE = True
except ImportError:
    print("❌ joyconrobotics library not found. Please install it with: pip install joyconrobotics")
    JOYCON_AVAILABLE = False

def get_battery_level(joycon):
    """
    Get battery level from Joy-Con controller
    Joy-Con battery levels are typically reported as 0-4 (empty to full)
    """
    try:
        # Try to access the underlying joycon object
        jc = joycon.joycon

        # Method 1: Try direct battery attributes
        if hasattr(jc, 'battery_level'):
            return jc.battery_level
        if hasattr(jc, 'battery'):
            return jc.battery

        # Method 2: Try SPI flash read (common method for Joy-Con battery)
        try:
            # Read battery level from SPI flash address 0x86
            battery_data = jc.spi_flash_read(0x86, 1)
            if battery_data and len(battery_data) > 0:
                # Battery level is usually in lower 4 bits
                battery_level = battery_data[0] & 0x0F
                return battery_level
        except Exception as e:
            print(f"SPI flash read failed: {e}")

        # Method 3: Try device info
        try:
            info = jc.get_info()
            if isinstance(info, dict):
                if 'battery' in info:
                    return info['battery']
                if 'battery_level' in info:
                    return info['battery_level']
        except Exception as e:
            print(f"Device info read failed: {e}")

        # Method 4: Try input report (some Joy-Cons report battery in input reports)
        try:
            # Get current input report
            report = jc.read_input_report()
            if report and len(report) > 0:
                # Battery info is often in specific bytes of the input report
                # This varies by Joy-Con firmware version
                battery_byte = report[2] if len(report) > 2 else None
                if battery_byte is not None:
                    # Extract battery level (usually 4 bits)
                    battery_level = (battery_byte >> 4) & 0x0F
                    return battery_level
        except Exception as e:
            print(f"Input report read failed: {e}")

        return None
    except Exception as e:
        return f"Error: {e}"

def battery_level_to_percentage(level):
    """
    Convert Joy-Con battery level (0-4) to percentage
    """
    if isinstance(level, str):
        return level  # Return error message as-is

    if level is None:
        return "Unknown"

    # Joy-Con typically reports 0-4 levels
    if isinstance(level, int) and 0 <= level <= 4:
        percentage = (level / 4.0) * 100
        return ".0f"
    else:
        return f"{level}% (raw)"

def get_battery_status_description(level):
    """
    Get human-readable battery status
    """
    if isinstance(level, str):
        return level

    if level is None:
        return "Unknown"

    if isinstance(level, int) and 0 <= level <= 4:
        if level == 0:
            return "🔴 Empty - Please charge!"
        elif level == 1:
            return "🟠 Low - Charge soon"
        elif level == 2:
            return "🟡 Medium"
        elif level == 3:
            return "🟢 Good"
        elif level == 4:
            return "🟢 Full"
    else:
        if level < 20:
            return "🔴 Very Low"
        elif level < 40:
            return "🟠 Low"
        elif level < 60:
            return "🟡 Medium"
        elif level < 80:
            return "🟢 Good"
        else:
            return "🟢 Full"

def print_battery_report(joycon_right, joycon_left):
    """
    Print a formatted battery report
    """
    print("\n📊 Battery Status Report:")
    print("=" * 50)
    print("<12")
    print("-" * 50)

    if joycon_right:
        right_battery = get_battery_level(joycon_right)
        right_percentage = battery_level_to_percentage(right_battery)
        right_status = get_battery_status_description(right_battery)
        print("<12")

    if joycon_left:
        left_battery = get_battery_level(joycon_left)
        left_percentage = battery_level_to_percentage(left_battery)
        left_status = get_battery_status_description(left_battery)
        print("<12")

    print("=" * 50)

def main():
    if not JOYCON_AVAILABLE:
        return

    print("🎮 Joy-Con Battery Monitor")
    print("=" * 40)

    joycon_left = None
    joycon_right = None

    try:
        # Try to connect to Joy-Con controllers
        print("🔍 Searching for Joy-Con controllers...")

        try:
            joycon_right = JoyconRobotics("right")
            print("✅ Right Joy-Con connected")
        except Exception as e:
            print(f"❌ Right Joy-Con not found: {e}")

        try:
            joycon_left = JoyconRobotics("left")
            print("✅ Left Joy-Con connected")
        except Exception as e:
            print(f"❌ Left Joy-Con not found: {e}")

        if not joycon_right and not joycon_left:
            print("❌ No Joy-Con controllers found!")
            print("💡 Make sure Joy-Cons are paired via Bluetooth")
            print("💡 Try running: sudo systemctl start bluetooth")
            return

        # Print initial battery report
        print_battery_report(joycon_right, joycon_left)

        # Ask user if they want continuous monitoring
        try:
            response = input("\n🔄 Enable continuous monitoring? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("\n⏰ Monitoring battery levels every 30 seconds...")
                print("Press Ctrl+C to stop monitoring\n")

                while True:
                    time.sleep(30)  # Wait 30 seconds
                    print_battery_report(joycon_right, joycon_left)
            else:
                print("\n👋 Single check completed. Exiting...")

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user.")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Clean up
        try:
            if joycon_right:
                joycon_right.disconnect()
                print("✅ Right Joy-Con disconnected")
        except:
            pass

        try:
            if joycon_left:
                joycon_left.disconnect()
                print("✅ Left Joy-Con disconnected")
        except:
            pass

if __name__ == "__main__":
    main()