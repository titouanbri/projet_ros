#!/usr/bin/env python3
from typing import List
import rclpy
from rclpy.node import Node, Publisher
from std_msgs.msg import Bool


class Scheduler(Node):
    
    def __init__(self):
        super().__init__("scheduler")
        self.get_logger().info(f"Initializing Scheduler")

        self.functions = [
            "homing",
            "eye_in_hand",
            "detection",
            "servoing",
        ]

        self.cmd_pubs = {}
        self.feedback = {}

        for fn in self.functions:
            self.cmd_pubs[fn] = self.create_publisher(
                Bool, f"{fn}_cmd", 3
            )
            self.create_subscription(
                Bool,
                f"{fn}_feedback",
                lambda msg, fn=fn: self.feedback_cb(msg, fn),
                2,
            )
            self.feedback[fn] = False

        self.sequence = [   #FOR GRAFCET LIKE BEHAVIOR
            "homing",
            "cam_calib",
            "eye_in_hand",
            "detection",
            "servoing",
        ]

        self.current_idx = 0
        self.active_fn = None

        self.timer = self.create_timer(0.1, self.step)

    def feedback_cb(self, msg : Bool, fn):
        self.feedback[fn] = msg.data

    def send_cmd(self, fn : str, value: bool):
        msg = Bool()
        msg.data = value
        pub : Publisher = self.cmd_pubs[fn]
        pub.publish(msg)

    def step(self):
        self.get_logger().info(f"Stepping...",throttle_duration_sec=60)

        homed = self.feedback["homing"]
        eye_in_hand_calibrated = self.feedback["eye_in_hand"]

        #probably replace these with a dedicated function and state vars
        if not homed:
            self.send_cmd("homing", True)
            self.get_logger().warn(f"Homing....",throttle_duration_sec=10)
        else:
            self.send_cmd("homing", False)
            self.get_logger().warn(f"Homing finished!",throttle_duration_sec=10)

        if homed and not eye_in_hand_calibrated:
            self.send_cmd("eye_in_hand", True)
            self.send_cmd("servoing", True)
            self.get_logger().warn(f"Doing eye-in-hand calibration....",throttle_duration_sec=10)
        elif homed:
            self.send_cmd("eye_in_hand", False)
            self.send_cmd("servoing", False)
            self.get_logger().warn(f"Eye-in-hand calibration finished!",throttle_duration_sec=10)

        
     



def main():
    rclpy.init()
    node = Scheduler()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
