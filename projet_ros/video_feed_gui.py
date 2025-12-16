import rclpy
import cv2
import tkinter as tk  

from PIL import ImageTk
from PIL import Image as pilimg 
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32,Polygon


prev_point= None
start_point= None
window_w=0
window_h=0


class Front(Node):
    def __init__(self):
        super().__init__("video_feed_gui") 
        self.sub_cam=self.create_subscription(Image,'/yolo/smartphone_result',self.process,10) #subscribe to /camera topic
        self.cam_debug=self.create_publisher(CompressedImage,'/cam_debug',10) #used to check if subbing is working
        self.cv_bridge=CvBridge()
        self.img= None

    def process(self,ros_im):  
        global window_h,window_w 
        cv_im=self.cv_bridge.imgmsg_to_cv2(ros_im, desired_encoding='bgr8') #converting ROS Image to cv2 image
        cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB) #converting to RGB
        img= pilimg.fromarray(cv_im) #converting to PIL image type
        img= img.resize((window_w,window_h),) #add pilimg.Resampling.LANCZOS for AA after tuple to apply anti-aliasing
        imgtk = ImageTk.PhotoImage(image=img) #converting to Tkinter image format
        self.img= imgtk
        smsg= String()
        smsg.data="publishing OK"

        if img != None:
            compressed_data = cv2.imencode('.jpg', cv_im)[1].tobytes()
            ret_im= CompressedImage()
            ret_im.format= "jpeg"
            ret_im.data= compressed_data
            self.cam_debug.publish(ret_im)
        
    def yield_img(self):
        return self.img

class MainWindow():                             #class for the main UI window
    def __init__(self, window : tk.Tk, node : Front ):
        global window_w,window_h
        self.window = window                    #AKA root
        self.node  = node
        self.width = window_w
        self.height = window_h
        self.interval = 20                      # Interval in ms to get the latest frame

        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width-5, height=self.height-5)
        self.canvas.grid(row=0, column=0)
        self.update_image()

        
    def update_image(self):
        global window_h, window_w
        self.image = self.node.yield_img()

        window_w , window_h = self.window.winfo_width(),self.window.winfo_height()

        self.canvas.config(width=window_w ,height=window_h )
        self.width = window_w
        self.height = window_h

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image, tag="image")        # Update image
        rclpy.spin_once(self.node)

        self.window.after(self.interval, self.update_image)         # update again every 'interval' ms
 


def main(args=None):
    global window_h,window_w
    root = tk.Tk()

    if window_h <=0 or window_w <= 0:
            window_h= root.winfo_screenheight()//2
            window_w= root.winfo_screenwidth()//2

    root.geometry(f"{window_w}x{window_h}")
    root.title("Video feed GUI")

    rclpy.init(args=args)
    front= Front()
    rclpy.spin_once(front)
    window=MainWindow(root, front)

    root.mainloop()
    

if __name__ == '__main__':
    main()