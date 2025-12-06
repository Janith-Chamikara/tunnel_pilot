#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
import time


class MotorTestNode(Node):
    def __init__(self):
        super().__init__('motor_test')

        # 1. Subscribers
        self.state_sub = self.create_subscription(
            State, 'mavros/state', self.state_cb, 10)

        # 2. Service Clients
        self.arming_client = self.create_client(
            CommandBool, 'mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, 'mavros/set_mode')

        self.current_state = State()

        # Wait for services
        self.get_logger().info("Waiting for MAVROS services...")
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for arming service...")

        self.get_logger().info("Services Ready! Waiting for Connection...")

    def state_cb(self, msg):
        self.current_state = msg

    def run_test(self):
        # Step 1: Wait for FCU Connection
        while not self.current_state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info("Waiting for FCU connection...", throttle_duration_sec=2)

        self.get_logger().info("FCU Connected.")

        # Step 2: Set Mode to STABILIZED
        # (We use Stabilized because OFFBOARD requires a constant stream of setpoints)
        self.set_mode("STABILIZED")
        time.sleep(2)  # Give it time to switch

        # Step 3: ARM (Motors Start Spinning)
        self.get_logger().warn("⚠️ ARMING MOTORS IN 3 SECONDS... (HOPE PROPS ARE OFF)")
        time.sleep(1)
        self.get_logger().warn("2...")
        time.sleep(1)
        self.get_logger().warn("1...")
        time.sleep(1)

        if self.arm_drone(True):
            self.get_logger().info("🚀 MOTORS SPINNING (Idle Speed) 🚀")
            self.get_logger().info("Observe for 5 seconds...")
            time.sleep(5)

            # Step 4: DISARM (Stop)
            self.get_logger().info("🛑 DISARMING...")
            self.arm_drone(False)
            self.get_logger().info("Test Complete.")
        else:
            self.get_logger().error("Failed to Arm. Check safety switch or battery.")

    def set_mode(self, mode_name):
        req = SetMode.Request()
        req.custom_mode = mode_name
        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().mode_sent

    def arm_drone(self, arm_cmd):
        req = CommandBool.Request()
        req.value = arm_cmd
        future = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success


def main(args=None):
    rclpy.init(args=args)
    node = MotorTestNode()

    try:
        node.run_test()
    except KeyboardInterrupt:
        node.arm_drone(False)
        print("\nForce Disarmed.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
