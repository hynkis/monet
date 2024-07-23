#!/usr/bin/env python

# """
# Date : 2019.11.21
# Author : Hyunki Seong

# Odometry pacakge
#     - subscribe vehicle state from CARLA
#     - publish odom message
# """

import math
import time
import numpy as np

# for ROS
import rospy
from std_msgs.msg import Int16, Float64, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped

import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion

STEER_FACTOR = np.deg2rad(20)/400.0   # max_steer(turn_left:21/turn_right:-18) : 20 deg / rc_cmd : (-)        1100 ~ 1500 ~ 1900 (+)
THROTTLE_FACTOR = 0.030               # 0.016                  / rc_cmd : (backward) 1900 ~ 1500 ~ 1100 (forward)
BIAS_CORRECTION_TIME = 5              # data acquisition time for bias reduction

WHEELBASE = 0.255 # [m] Tamiya RC car

# ===== Calculation ===== #
def pi_to_pi(angle):
    """
    normalize angle to -pi ~ +pi
    """
    if angle > math.pi:
        angle -= 2*math.pi
    elif angle < -math.pi:
        angle += 2*math.pi

    return angle

def transform(x, y, yaw, tx, ty, tyaw):
    """
    x, y, yaw : current states
    tx, ty, tyaw : target states
    """
    yaw = pi_to_pi(yaw)
    tyaw = pi_to_pi(tyaw)
    theta = -tyaw
    p = -tx
    q = -ty
    r = 0
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    # inverse rotation > inverse translation
    # x' := Rot * Trans * x
    rot_matrix = np.array([[c_theta, -s_theta,   0, p*c_theta - q*s_theta],
                            [s_theta, c_theta,   0, p*s_theta + q*c_theta],
                            [      0,       0,   1,                     r],
                            [      0,       0,   0,                     1]])
    output = np.matmul(rot_matrix, np.array([[x], [y], [0], [1]]))
    
    return output[0][0], output[1][0], pi_to_pi(yaw-tyaw) # for scalar value

# ===== Filter ===== #
class Filter(object):
    def __init__(self, w, data_init):
        self.w = w
        self.data_prev = data_init

    def filtering(self, data):
        output = data * self.w + self.data_prev * (1 - self.w)
        data_prev = data
        return output

# ==============================================================================
# -- ROS -----------------------------------------------------------------------
# ==============================================================================

class ROSGateway(object):
    """
    ROS gateway
    """

    def __init__(self):
        
        rospy.init_node('odom_publish_node', anonymous=True)

        # state
        self.x_ = 0.0
        self.y_ = 0.0
        self.yaw_ = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        
        # Accel
        self.ax = 0.0
        self.ay = 0.0

        # Accel bias
        self.ax_bias = None
        self.ay_bias = None

        self.steer = 0.0

        self.steer_auto = 0.0
        self.steer_manual = 0.0
        self.vx_auto = 0.0
        self.vx_manual = 0.0

        self.init_x   = None
        self.init_y   = None
        self.init_yaw = None

        self.auto_mode = False

        self.rate = rospy.Rate(100)

        self.tic = time.time()

        self.speed_filter = Filter(w=0.75, data_init=0.0)

        self.stamp = None

        # ----- Subscriber ----- #
        self.sub_imu      = rospy.Subscriber('/imu/imu', Imu, self.callback_imu)

        # ----- Publisher ----- #
        self.pub_odom        = rospy.Publisher('/odom_localization', Odometry, queue_size=1)

    def callback_imu(self, msg):
        """
        # IMU was attached with rotation (x-axis 180 deg) => (yaw_rate -> -yaw_rate / ay -> -ay)
        """
        self.stamp = msg.header.stamp
        # print("Sensor Time :", self.stamp.to_sec())
        orient_q = msg.orientation
        (roll, pitch, yaw) = euler_from_quaternion([orient_q.x, orient_q.y, orient_q.z, orient_q.w])
        self.yaw_ = yaw
        self.vyaw = - msg.angular_velocity.z
        self.ax = - msg.linear_acceleration.x # from filtered imu
        self.ay = msg.linear_acceleration.y   # from filtered imu

    def publish_odom(self, x, y, rot_q):
        """
        Callback agent state
            - publishing /odom message
            - broadcasting tf of agent
        """
        
        # ========== Publishing odom message ========== #
        timestamp = rospy.Time.now()
        if self.stamp is not None:
            timestamp = self.stamp

        # ROS message payload (NED to ENU coordinate)
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id  = "base_link"
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0
        odom_msg.pose.pose.orientation.x = rot_q[0] # x
        odom_msg.pose.pose.orientation.y = rot_q[1] # y
        odom_msg.pose.pose.orientation.z = rot_q[2] # z
        odom_msg.pose.pose.orientation.w = rot_q[3] # w
        # # covariance
        # odom_msg.pose.covariance[0]  = 0.0 # for x
        # odom_msg.pose.covariance[7]  = 0.0 # for y
        # odom_msg.pose.covariance[35] = 0.0 # for yaw

        # velocity
        # odom_msg.twist.twist.linear.x = self.vx
        # odom_msg.twist.twist.linear.y = self.vy
        #odom_msg.twist.twist.angular.z = yaw_dot
        odom_msg.twist.twist.angular.z = self.vyaw

        # Publish solution
        self.pub_odom.publish(odom_msg)

        _, _, yaw = euler_from_quaternion(rot_q)
        print("Publishing odom. x: %.3f, y: %.3f, yaw: %.3f, yaw_rate: %.3f" %(x, y, yaw, self.vyaw))

        # print("process time: %.2f, dt: %.4f" %(1/max(time.time()-self.tic, 1e-5), dt))
        self.tic = time.time()


def main():
    # ========== Main loop ========== #
    rosgateway = ROSGateway()
    imu_accel_x = []
    imu_accel_y = []

    listener = tf.TransformListener()

    while not rospy.is_shutdown():
        # Get current pose
        try:
            (trans, rot_q) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("No tf")
            continue
        x, y, _ = trans
        _, _, yaw = euler_from_quaternion(rot_q)
        
        rosgateway.publish_odom(x, y, rot_q)
        rosgateway.rate.sleep()
    

if __name__ == '__main__':
    main()
