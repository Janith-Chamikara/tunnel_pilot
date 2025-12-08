#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from mavros_msgs.msg import State
import time


class GPSTakeoffNode(Node):
    def __init__(self):
        super().__init__('gps_takeoff_test')

        # --- Subscribers ---
        self.state_sub = self.create_subscription(
            State, 'mavros/state', self.state_cb, 10)

        # --- Service Clients ---
        self.arming_client = self.create_client(
            CommandBool, 'mavros/cmd/arming')
        self.set_mode_client = self.create_client(
            SetMode, 'mavros/set_mode')
        self.takeoff_client = self.create_client(
            CommandTOL, 'mavros/cmd/takeoff')

        self.current_state = State()

        # Wait for services to be available
        self.get_logger().info("Waiting for MAVROS services...")
        self.arming_client.wait_for_service(1.0)
        self.set_mode_client.wait_for_service(1.0)
        self.takeoff_client.wait_for_service(1.0)
        self.get_logger().info("✅ Services Ready!")

    def state_cb(self, msg):
        self.current_state = msg

    def run_mission(self):
        # 1. Wait for Connection
        while not self.current_state.connected:
            self.get_logger().info("Waiting for FCU connection...", throttle_duration_sec=2)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("✅ FCU Connected.")

        # 2. Switch to GUIDED Mode
        # Note: If this fails, it usually means you don't have a good GPS Lock yet.
        self.get_logger().info("Attempting to switch to GUIDED mode...")
        self.set_mode("GUIDED")

        # Robust check loop
        start_wait = time.time()
        while self.current_state.mode != "GUIDED":
            rclpy.spin_once(self, timeout_sec=0.1)
            # Retry every 1 second
            if int(time.time()) % 2 == 0:
                self.set_mode("GUIDED")

            if time.time() - start_wait > 10.0:
                self.get_logger().error("❌ FAILED to enter GUIDED mode.")
                self.get_logger().error("Check: Do you have a solid Green LED (GPS Lock)?")
                return

        self.get_logger().info("✅ Mode is GUIDED.")

        # 3. ARM the Drone
        self.get_logger().warn("⚠️ ARMING IN 5 SECONDS... STAND CLEAR!")
        time.sleep(5)

        if self.arm_drone(True):
            self.get_logger().info("🚀 MOTORS ARMED 🚀")
            time.sleep(2)  # Allow motors to stabilize
        else:
            self.get_logger().error("❌ Failed to Arm. Check safety switch or battery.")
            return

        # 4. TAKEOFF to 1.0 Meter
        self.get_logger().info("🛫 TAKING OFF to 1.0 Meter...")
        if self.takeoff(1.0):
            self.get_logger().info("✅ Takeoff command accepted.")
        else:
            self.get_logger().error("❌ Takeoff rejected.")
            self.arm_drone(False)
            return

        # 5. HOVER (Wait 10 Seconds)
        self.get_logger().info("⏳ Hovering for 10 seconds...")

        # We use a loop here so we keep processing ROS callbacks
        hover_start = time.time()
        while time.time() - hover_start < 10.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        # 6. LAND
        self.get_logger().info("🛬 LANDING...")
        self.set_mode("LAND")

        # Monitor landing
        while self.current_state.armed:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info("Descending...", throttle_duration_sec=2)

        self.get_logger().info("✅ Landed and Disarmed.")

    # --- Helper Functions ---

    def set_mode(self, mode_name):
        req = SetMode.Request()
        req.custom_mode = mode_name
        self.set_mode_client.call_async(req)

    def arm_drone(self, arm_cmd):
        req = CommandBool.Request()
        req.value = arm_cmd
        future = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    def takeoff(self, altitude):
        req = CommandTOL.Request()
        req.altitude = float(altitude)
        req.latitude = 0.0  # Not used for guided takeoff
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0

        future = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success


def main(args=None):
    rclpy.init(args=args)
    node = GPSTakeoffNode()
    try:
        node.run_mission()
    except KeyboardInterrupt:
        print("Cancelled.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
