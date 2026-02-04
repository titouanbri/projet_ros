#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, String, Bool
import tkinter as tk

class EEVelTeleopGUI(Node):
    """Cartesian velocity teleop GUI for UR3e"""
    def __init__(self):
        super().__init__("cartesian_velocity_teleop_gui")

        # Parameters
        self.open_loop = False  # change if needed
        self.arm_id = "ur3e"
        self.cmd_pub_topic = "/ee_velocity_cmd"
        self.work_frame = f"wrist_3_link"

        if self.open_loop:
            self.cmd_pub_topic = "/ee_velocity_command"

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.cmd_pub_topic, 1)
        self.frame_pub = self.create_publisher(String, "/work_frame", 1)
        self.filter_pub = self.create_publisher(Float64, "/joint_velocity_controller/joint_velocity_filter", 1)
        self.pose_ctrl_disabler = self.create_publisher(Bool, '/auto_pose_control_enabled', 1)
        # GUI setup
        self.root = tk.Tk()
        self.root.title("UR3e Cartesian Velocity Teleop")
        self.active_cmd = None
        self.cmd = Twist()

        # Default scaling coefficients
        self.lin_scale_val = tk.DoubleVar(value=1.0)
        self.ang_scale_val = tk.DoubleVar(value=1.0)
        self.old_filter_val = 0.1
        self.filter_val = tk.DoubleVar(value=self.old_filter_val)

        self.error_var = tk.StringVar()
        self.error_var.set("Error: N/A")

        self.make_gui()

        # Subscribers
        self.create_subscription(Float64, "/rcm_to_axis_error", self.error_callback, 10)

        # Close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.loop()

    def make_gui(self):
        # Movement buttons
        self.make_button("← X-", lambda: self.set_cmd(linear=(-1, 0, 0)), 1, 0)
        self.make_button("→ X+", lambda: self.set_cmd(linear=(1, 0, 0)), 1, 2)
        self.make_button("↓ Y-", lambda: self.set_cmd(linear=(0, -1, 0)), 2, 1)
        self.make_button("↑ Y+", lambda: self.set_cmd(linear=(0, 1, 0)), 0, 1)
        self.make_button("⇣ Z-", lambda: self.set_cmd(linear=(0, 0, -1)), 3, 0)
        self.make_button("⇡ Z+", lambda: self.set_cmd(linear=(0, 0, 1)), 3, 2)

        # Rotation buttons
        self.make_button("⟲ Rx-", lambda: self.set_cmd(angular=(-1, 0, 0)), 4, 0)
        self.make_button("⟳ Rx+", lambda: self.set_cmd(angular=(1, 0, 0)), 4, 2)
        self.make_button("⟱ Ry-", lambda: self.set_cmd(angular=(0, -1, 0)), 5, 0)
        self.make_button("⟰ Ry+", lambda: self.set_cmd(angular=(0, 1, 0)), 5, 2)
        self.make_button("⤵ Rz-", lambda: self.set_cmd(angular=(0, 0, -1)), 6, 0)
        self.make_button("⤴ Rz+", lambda: self.set_cmd(angular=(0, 0, 1)), 6, 2)

        # Linear & angular sliders
        tk.Label(self.root, text="Linear Scale").grid(row=8, column=0, pady=(10,0))
        lin_slider = tk.Scale(self.root, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.lin_scale_val)
        lin_slider.grid(row=8, column=1, columnspan=2, sticky="we")

        tk.Label(self.root, text="Angular Scale").grid(row=9, column=0)
        ang_slider = tk.Scale(self.root, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.ang_scale_val)
        ang_slider.grid(row=9, column=1, columnspan=2, sticky="we")

        # Low-pass filter slider
        tk.Label(self.root, text="Filter parameter").grid(row=10, column=0)
        filter_slider = tk.Scale(self.root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                 variable=self.filter_val, command=self.change_filter)
        filter_slider.grid(row=10, column=1, columnspan=2, sticky="we")

        # Error display
        error_label = tk.Label(self.root, textvariable=self.error_var, fg="red", font=("Arial", 10))
        error_label.grid(row=11, column=0, columnspan=3, pady=(5,10))

    def change_filter(self, new_val):
        val = float(new_val)
        if val != self.old_filter_val:
            self.old_filter_val = val
            self.filter_pub.publish(Float64(data=val))

    def make_button(self, text, press_fn, row, col):
        btn = tk.Button(self.root, text=text, width=6, height=2)
        btn.grid(row=row, column=col, padx=5, pady=5)
        btn.bind("<ButtonPress>", lambda e: self.start_motion(press_fn))
        btn.bind("<ButtonRelease>", lambda e: self.stop_motion())
        return btn

    def set_cmd(self, linear=(0,0,0), angular=(0,0,0)):
        lin_coef = self.lin_scale_val.get()
        ang_coef = self.ang_scale_val.get()

        self.cmd.linear.x  = lin_coef * 0.02 * linear[0]
        self.cmd.linear.y  = lin_coef * 0.02 * linear[1]
        self.cmd.linear.z  = lin_coef * 0.02 * linear[2]
        self.cmd.angular.x = ang_coef * 0.06 * angular[0]
        self.cmd.angular.y = ang_coef * 0.06 * angular[1]
        self.cmd.angular.z = ang_coef * 0.06 * angular[2]

    def start_motion(self, cmd_fn):
        self.pose_ctrl_disabler.publish(Bool(data=False))
        cmd_fn()
        self.active_cmd = self.cmd

    def stop_motion(self):
        self.cmd = Twist()
        self.cmd_pub.publish(self.cmd)
        self.active_cmd = None

    def error_callback(self, msg: Float64):
        self.error_var.set(f"Error: {msg.data*1e3:.2f} mm")

    def loop(self):
        if self.active_cmd:
            self.cmd_pub.publish(self.active_cmd)
        self.root.after(10, self.loop)

    def on_close(self):
        self.stop_motion()
        self.root.destroy()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = EEVelTeleopGUI()
    try:
        node.root.mainloop()
    except KeyboardInterrupt:
        node.on_close()
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
