#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State, Thrust
from geometry_msgs.msg import PoseStamped
import time


class MotorTestNode(Node):
    def __init__(self):
        super().__init__('motor_test')

        # --- Publishers ---
        # We need to send Attitude (Angle) and Thrust commands continuously
        self.att_pub = self.create_publisher(
            PoseStamped, 'mavros/setpoint_attitude/attitude', 10)
        self.thrust_pub = self.create_publisher(
            Thrust, 'mavros/setpoint_attitude/thrust', 10)

        # --- Subscribers ---
        self.state_sub = self.create_subscription(
            State, 'mavros/state', self.state_cb, 10)

        # --- Service Clients ---
        self.arming_client = self.create_client(
            CommandBool, 'mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, 'mavros/set_mode')

        self.current_state = State()
        self.target_thrust = 0.0  # Start at 0

        # Create a timer to publish setpoints at 20Hz (Required for OFFBOARD)
        self.timer = self.create_timer(0.05, self.publish_setpoints)

        self.get_logger().info("Waiting for MAVROS services...")
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for services...")
        self.get_logger().info("Services Ready!")

    def state_cb(self, msg):
        self.current_state = msg

    def publish_setpoints(self):
        # 1. Publish Attitude (Keep it Flat/Level)
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        # Quaternion for "Level" (x=0, y=0, z=0, w=1)
        pose.pose.orientation.w = 1.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        self.att_pub.publish(pose)

        # 2. Publish Thrust
        thrust_msg = Thrust()
        thrust_msg.header.stamp = self.get_clock().now().to_msg()
        thrust_msg.thrust = self.target_thrust
        self.thrust_pub.publish(thrust_msg)

    def run_test(self):
        # Step 1: Wait for Connection
        while not self.current_state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info("Waiting for FCU connection...", throttle_duration_sec=2)

        self.get_logger().info("FCU Connected.")

        # Step 2: Prepare for OFFBOARD
        # We must stream setpoints for a bit before switching mode
        self.target_thrust = 0.0
        for _ in range(20):  # Stream for 1 second
            rclpy.spin_once(self, timeout_sec=0.05)

        # Step 3: Set Mode to OFFBOARD
        self.set_mode("OFFBOARD")
        time.sleep(1)

        # Step 4: ARM
        self.get_logger().warn("⚠️ ARMING IN 3 SECONDS...")
        time.sleep(3)

        if self.arm_drone(True):
            self.get_logger().info("🚀 MOTORS ARMED (Idle) 🚀")
            time.sleep(2)  # Wait a moment at idle

            # Step 5: Increase Speed to 20%
            self.get_logger().info("⚡ INCREASING THROTTLE TO 20% ⚡")
            self.target_thrust = 0.2  # <--- THIS SETS THE SPEED (0.0 to 1.0)

            # Keep the node spinning for 20 seconds
            start_time = time.time()
            while (time.time() - start_time) < 20.0:
                rclpy.spin_once(self, timeout_sec=0.1)
                # We stay in this loop so the Timer keeps firing

            # Step 6: Stop
            self.get_logger().info("🛑 Time up. Stopping...")
            self.target_thrust = 0.0
            time.sleep(1)  # Let them spin down
            self.arm_drone(False)
            self.get_logger().info("Test Complete.")

        else:
            self.get_logger().error("Failed to Arm.")

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
        node.target_thrust = 0.0
        node.arm_drone(False)
        print("\nForce Disarmed.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
