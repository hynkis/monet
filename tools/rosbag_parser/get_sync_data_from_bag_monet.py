#! /usr/bin/python2
"""
Data generation

    1. Route plan & re-record
        {img, bev, rc_cmd/steer, rc_cmd/throttle,
         imu/accel_x, imu/accel_y, imu/accel_z, imu/yaw_rate, odom/x, odom/y, odom/yaw}
        rosbag record /camera/infra1/image_rect_raw/compressed bev/image/compressed /imu/imu /odom_localization /rc_cmd/steer /rc_cmd/throttle

    1. Filtering rosbag
    - rosbag filter 2021-05-11-17-53-56.bag filtered_2021-05-11-17-53-56.bag "t.secs >=1619885848.342863 and t.secs <= 1619885905.273909"
    : https://answers.ros.org/question/99711/how-to-split-a-recorded-rosbag-file/

    2. set topic name, bagfile path (BAG_DIR)

    3. make directories: img, csv, bev

    4. run this code
    - python2 get_sync_data_from_bag.py

Flow
    - 

etc (23.05.09)
    - neutral steer: 1412~1416 --> 1414
    - neutral throttle: 1500 (usually 1455)

Data
    0 : 'img_stamp'
    1 : 'img_path'
    2 : 'steer_t'
    3 : 'throttle_t'
    4 : 'imu_accel_x'
    5 : 'imu_accel_y'
    6 : 'imu_yaw_rate'
    7 : 'pose_x'
    8 : 'pose_y'
    9 : 'pose_yaw'
"""
import rosbag
import pandas as pd
from sensor_msgs.msg import Image
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from cv_bridge import CvBridge, CvBridgeError
import cv2
import csv
from utils_parser import *

from gen_topological_map import TopoMapTransfer, road_map

bridge = CvBridge()

SHOW_PARSED_DATA = True
SHOW_IMG = True
SHOW_BEV = True

SAVE_IMG = True

INITIAL_DATA_PASS_SIZE = 0 # for passing initial several data
# SAVE_INTERVAL = 10 # for training dataset
SAVE_INTERVAL = 101 # for eval dataset (let the train dataset 8 times more than valid dataset)

topic_img      = '/camera/infra1/image_rect_raw/compressed'
topic_bev      = 'bev/image/compressed'
topic_steer    = '/rc_cmd/steer'
topic_throttle = '/rc_cmd/throttle'
topic_pose     = '/odom_localization'
topic_imu      = '/imu/imu'

# BAG_DIR = '../../dataset/rosbag/2021-05-02'
# BAG_DIR = '../../dataset/rosbag/2021-06-02_03'
# BAG_DIR = '../../dataset/rosbag/2021-06-23_24'
# BAG_DIR = '../../dataset/rosbag/2021-05_06'

BAG_DIR = '../../dataset/rosbag_n1_3rd_5th/230505_n1_5th/postprocessed'
# BAG_DIR = '../../dataset/rosbag_n1_3rd_5th/230507_n1_3rd/postprocessed'

bag_file_names = get_file_name_in_dir(BAG_DIR, 'bag')
num_total_bag = len(bag_file_names)

IMG_DIR = '../../dataset/img'
CSV_DIR = '../../dataset/csv'
BEV_DIR = '../../dataset/bev'

for idx, bag_file_name in enumerate(bag_file_names):
    print("Start to parse rosbag file: ", bag_file_name)
    print("Data Generation progress : %d / %d" %(idx, num_total_bag))
    # BAG_NAME = 'filter-2021-05-17-20-59-46'
    BAG_NAME = bag_file_name[:-4] # without '.bag'
    BAG_PATH = BAG_DIR + '/' + BAG_NAME + '.bag'
    bag = rosbag.Bag(BAG_PATH)

    STATE_DATA_PATH = CSV_DIR + '/' + 'state_' + BAG_NAME + '.csv'

    num_data = 0

    # for data container
    # : get img msg first and then get the latest other msgs 
    img_t        = None
    img_t_header = None
    bev_t        = None
    steer_t      = None
    throttle_t   = None
    pose_t       = None
    imu_t        = None

    closest_wpt_x = None
    closest_wpt_y = None

    # ==================================== #
    # ===== 1. Parsing from bag file ===== #
    # ==================================== #
    f = open(STATE_DATA_PATH, 'w')
    thewriter = csv.writer(f)

    # with open(STATE_DIR+'/state.csv','ab') as f:
    #     thewriter = csv.writer(f)
    thewriter.writerow(['img_stamp',
                        'img_path',
                        'steer_t',
                        'throttle_t',
                        'imu_accel_x',
                        'imu_accel_y',
                        'imu_yaw_rate',
                        'pose_x',
                        'pose_y',
                        'pose_yaw',
                        ])
    
    data_length = len(list(bag.read_messages()))
    for data_idx, (topic, msg, t) in enumerate(bag.read_messages()):
        if(topic == topic_img):
            # Get current image before getting steering angle
            if img_t is None:
                try:
                    # Convert compressed image to RAW & Save it at Container
                    img_t = bridge.compressed_imgmsg_to_cv2(msg)
                    img_t_header = msg.header
                    bev_t = None
                    steer_t = None
                    throttle_t = None
                    pose_t = None
                    imu_t = None
                    
                    if SHOW_IMG:
                        cv2.imshow('image', img_t)
                        cv2.waitKey(10)

                except CvBridgeError as e:
                    print(e)

        if(topic == topic_bev):
            # Get current bev image after getting front image
            if img_t is not None and bev_t is None:
                try:
                    # Convert compressed image to RAW & save it at Container
                    bev_t = bridge.compressed_imgmsg_to_cv2(msg)
                    
                    if SHOW_BEV:
                        cv2.imshow('bev', bev_t)
                        cv2.waitKey(10)

                except CvBridgeError as e:
                    print(e)

        if(topic == topic_steer):
            # Get current steering angle after getting image 
            if img_t is not None and steer_t is None:
                # print("t :", t.to_sec(), "Steer :", msg.data)
                steer_t = msg.data

        if(topic == topic_throttle):
            # Get current throttle after getting image 
            if img_t is not None and throttle_t is None:
                # print("t :", t.to_sec(), "Throttle :", msg.data)
                throttle_t = msg.data

        if(topic == topic_pose):
            # Get current throttle after getting image 
            if img_t is not None and pose_t is None:
                pose_t = msg.pose.pose # msg.pose.pose.position
                # print("t :", t.to_sec(), "Pose :", msg.pose.pose.position.x, msg.pose.pose.position.y, "Closest wpt :", closest_wpt_x, closest_wpt_y, "closest_d :", closest_d, "is_arrive :", is_arrive)

        if(topic == topic_imu):
            # Get current throttle after getting image 
            if img_t is not None and imu_t is None:
                # print("t :", t.to_sec(), "Imu :", msg.linear_acceleration.x, msg.linear_acceleration.y, msg.angular_velocity.z)
                imu_t = msg

        # Save data in container
        if img_t is not None and \
            bev_t is not None and \
            steer_t is not None and \
            throttle_t is not None and \
            pose_t is not None and \
            imu_t is not None:
            # print("got all data. t :", t.to_sec(), "Image num :", num_data, "image_size :", img_t.shape)
            num_data += 1

            # Passing data condition
            # : initial data pass
            # : data interval
            if data_idx > INITIAL_DATA_PASS_SIZE and num_data % SAVE_INTERVAL == 0:
                # Save image: Writing image to the directory mentioned while executing script
                if SAVE_IMG:
                    img_path = 'frame_' + BAG_NAME + '_' + str(img_t_header.stamp) + '.jpg'
                    cv2.imwrite(IMG_DIR +'/' + img_path, img_t)
                    cv2.imwrite(BEV_DIR + '/' + img_path, bev_t)

                # Save imu, route data information as per requirement
                orient_q = pose_t.orientation
                _, _, pose_yaw = euler_from_quaternion([orient_q.x, orient_q.y, orient_q.z, orient_q.w])

                # [img_stamp, img_path, imu_accel_x, imu_accel_y, imu_yaw_rate, pose_x, pose_y, pose_yaw, closest_wpt_x, closest_wpt_y, is_arrive]
                img_path = 'frame_' + BAG_NAME + '_' + str(img_t_header.stamp) + '.jpg'
                thewriter.writerow([str(img_t_header.stamp),
                                    img_path,
                                    steer_t,
                                    throttle_t,
                                    imu_t.linear_acceleration.x,
                                    imu_t.linear_acceleration.y,
                                    imu_t.angular_velocity.z,
                                    pose_t.position.x,
                                    pose_t.position.y,
                                    pose_yaw,
                                    ])

                if SHOW_PARSED_DATA:
                    print("Save img, steer, throttle, pose, imu")
                    print("img :", img_t.shape, "bev :", bev_t.shape, "steer :", steer_t, "throttle :", throttle_t, "pose :", pose_t.position.x, "imu :", imu_t.linear_acceleration.x)

                print("Saved data: num_data: ", num_data, "img :", img_t.shape, "bev :", bev_t.shape, "steer :", steer_t, "throttle :", throttle_t)

            else:
                # print("Not save this data! data_idx :", data_idx)
                continue

                # if num_data >= MAX_NUM_DATA:
                #     print("Max num of data.")
                #     break

            # Reset
            img_t      = None
            bev_t      = None
            steer_t    = None
            throttle_t = None
            pose_t     = None
            imu_t      = None
    
    bag.close()
    f.close()
