#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from mavros_msgs.msg import State, OverrideRCIn
from mavros_msgs.srv import CommandBool, SetMode

import time


class MotorTestNode(Node):
    def __init__(self):
        super().__init__('motor_test_dynamic_stabilized')

        # --- CONFIG ---
        # Mode that does NOT require GPS (ArduCopter: STABILIZE, ACRO, ALT_HOLD, etc.)
        self.target_mode = 'STABILIZE'
        # Approx 20% throttle (1000–2000 µs range)
        self.throttle_pwm_20pct = 1200
        # How long to hold 20% throttle (seconds)
        self.test_duration_sec = 5.0
        # ----------------

        self.state = State()
        self.test_started = False
        self.test_finished = False

        # Subscribers
        self.state_sub = self.create_subscription(
            State, 'mavros/state', self.state_cb, 10
        )

        # Publishers
        self.rc_pub = self.create_publisher(
            OverrideRCIn, 'mavros/rc/override', 10
        )

        # Service clients
        self.arming_client = self.create_client(
            CommandBool, 'mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, 'mavros/set_mode')

        self.timer = self.create_timer(0.1, self.update)  # 10 Hz

        self.get_logger().warn(
            "SAFETY: Make sure PROPS ARE REMOVED for first tests!"
        )

    def state_cb(self, msg: State):
        self.state = msg

    def wait_for_service(self, client, name: str):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {name} service...')

    def set_mode(self, mode: str) -> bool:
        self.wait_for_service(self.set_mode_client, 'set_mode')
        req = SetMode.Request()
        req.custom_mode = mode
        self.get_logger().info(f"Setting mode to {mode}...")
        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().mode_sent:
            self.get_logger().info(f"Mode {mode} set successfully.")
            return True
        else:
            self.get_logger().error("Failed to set mode.")
            return False

    def arm(self, value: bool) -> bool:
        self.wait_for_service(self.arming_client, 'arming')
        req = CommandBool.Request()
        req.value = value
        action = "Arming" if value else "Disarming"
        self.get_logger().info(f"{action}...")
        future = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().success:
            self.get_logger().info(f"{action} succeeded.")
            return True
        else:
            self.get_logger().error(f"{action} failed.")
            return False

    def send_throttle(self, pwm: int):
        msg = OverrideRCIn()
        # 8 channels, 65535 = ignore channel
        msg.channels = [65535] * 8
        # Channel 3 is throttle on ArduCopter (index 2)
        msg.channels[2] = pwm
        self.rc_pub.publish(msg)

    def update(self):
        # Wait until connected
        if not self.state.connected:
            self.get_logger().info_throttle(2.0, "Waiting for FCU connection...")
            return

        # Only run test once
        if self.test_finished:
            return

        # Step 1: set mode
        if self.state.mode != self.target_mode and not self.test_started:
            ok = self.set_mode(self.target_mode)
            if not ok:
                self.get_logger().error(
                    "Cannot set mode. Check pre-arm checks / params."
                )
                self.test_finished = True
            return

        # Step 2: arm
        if not self.state.armed and not self.test_started:
            ok = self.arm(True)
            if not ok:
                self.get_logger().error(
                    "Cannot arm. Check safety switch / pre-arm checks."
                )
                self.test_finished = True
            return

        # Step 3: run motor test once we are armed and in correct mode
        if self.state.armed and not self.test_started:
            self.get_logger().warn(
                "Starting motor test: 20% throttle for "
                f"{self.test_duration_sec} seconds."
            )
            self.test_started = True
            self.start_time = time.time()

        if self.test_started and not self.test_finished:
            elapsed = time.time() - self.start_time
            if elapsed < self.test_duration_sec:
                # Hold 20% throttle
                self.send_throttle(self.throttle_pwm_20pct)
            else:
                # Bring throttle back to idle, then disarm and finish
                self.get_logger().info("Motor test done, returning to idle.")
                self.send_throttle(1000)  # idle
                self.arm(False)
                self.test_finished = True
                self.get_logger().info("Test finished. Node will now idle.")


def main(args=None):
    rclpy.init(args=args)
    node = MotorTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Motor test interrupted by user.")
        node.arm(False)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
