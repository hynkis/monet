#! /usr/bin/python2
"""
Data generation
    1. Filtering rosbag
    - rosbag filter 2021-05-11-17-53-56.bag filtered_2021-05-11-17-53-56.bag "t.secs >=1619885848.342863 and t.secs <= 1619885905.273909"
    : https://answers.ros.org/question/99711/how-to-split-a-recorded-rosbag-file/

    2. set topic name, bagfile path (BAG_DIR)

    3. make directories: img, csv, bev

    4. run this code
    - python2 get_sync_data_from_bag.py

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
    10 : 'closest_wpt_x'
    11 : 'closest_wpt_y'
    12 : 'is_arrive'
    13 : 'target_wpt_x_t'
    14 : 'target_wpt_y_t'
    15 : 'target_wpt_x_t1'
    16 : 'target_wpt_y_t1'
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

SHOW_PARSED_DATA = False
SHOW_IMG = False
SHOW_BEV_MAP = True

SAVE_IMG = True
SAVE_POSTPROCESSED_TARGET_WPT = True
SAVE_POSTPROCESSED_BEV_MAP = True

PARSING_ROSBAG_DATA = True
POSTPROCESSING_TARGET_WPT = True
POSTPROCESSING_BEV_MAP = True

INITIAL_DATA_PASS_SIZE = 50 # for passing initial several data
SAVE_INTERVAL = 10

topic_img      = '/camera/infra1/image_rect_raw/compressed'
topic_steer    = '/rc_cmd/steer'
topic_throttle = '/rc_cmd/throttle'
topic_pose     = '/odom_localization'
topic_imu      = '/imu/imu'

# BAG_DIR = '../../dataset/rosbag/2021-05-02'
# BAG_DIR = '../../dataset/rosbag/2021-06-02_03'
# BAG_DIR = '../../dataset/rosbag/2021-06-23_24'
BAG_DIR = '../../dataset/rosbag/2021-05_06'

bag_file_names = get_file_name_in_dir(BAG_DIR, 'bag')
num_total_bag = len(bag_file_names)

for idx, bag_file_name in enumerate(bag_file_names):
    print("Start to parse rosbag file: ", bag_file_name)
    print("Progress :", idx, num_total_bag)
    # BAG_NAME = 'filter-2021-05-17-20-59-46'
    BAG_NAME = bag_file_name[:-4] # without '.bag'
    BAG_PATH = BAG_DIR + '/' + BAG_NAME + '.bag'
    bag = rosbag.Bag(BAG_PATH)

    IMG_DIR = '../../dataset/img'
    CSV_DIR = '../../dataset/csv'
    BEV_DIR = '../../dataset/bev'
    STATE_DATA_PATH = CSV_DIR + '/' + 'state_' + BAG_NAME + '.csv'

    num_data = 0

    # for data container
    img_t = None
    img_t_header = None
    steer_t = True
    throttle_t = True
    pose_t = True
    imu_t = True

    closest_wpt_x = None
    closest_wpt_y = None

    # ==================================== #
    # ===== 1. Parsing from bag file ===== #
    # ==================================== #
    if PARSING_ROSBAG_DATA:
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
                            'closest_wpt_x',
                            'closest_wpt_y',
                            'is_arrive',
                            'done',
                            ])
        
        data_length = len(list(bag.read_messages()))
        for data_idx, (topic, msg, t) in enumerate(bag.read_messages()):
            if(topic == topic_img):
                # Get current image after getting steering angle
                if steer_t is not None and throttle_t is not None and pose_t is not None and imu_t is not None:
                    try:
                        # Convert compressed image to RAW
                        img = bridge.compressed_imgmsg_to_cv2(msg)
                        num_data += 1
                        # print("t :", t.to_sec(), "Image num :", num_data, "image_size :", img.shape)
                        
                        if SHOW_IMG:
                            cv2.imshow('image', img)
                            cv2.waitKey(0)
                        
                        # Save current image at Container
                        img_t = img
                        img_t_header = msg.header
                        steer_t = None
                        throttle_t = None
                        pose_t = None
                        imu_t = None

                    except CvBridgeError, e:
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
                    # Calculate min distance wpt
                    # closest_wpt_x, closest_wpt_y, closest_d = find_closest_wpt(pose_t.position.x, pose_t.position.y)
                    is_arrive, closest_wpt_x, closest_wpt_y, closest_d = find_arrving_wpt(pose_t.position.x, pose_t.position.y)
                    if closest_d < MINIMUM_DIST:
                        MINIMUM_DIST = closest_d
                    # print("t :", t.to_sec(), "Pose :", msg.pose.pose.position.x, msg.pose.pose.position.y, "Closest wpt :", closest_wpt_x, closest_wpt_y, "closest_d :", closest_d, "is_arrive :", is_arrive)


            if(topic == topic_imu):
                # Get current throttle after getting image 
                if img_t is not None and imu_t is None:
                    # print("t :", t.to_sec(), "Imu :", msg.linear_acceleration.x, msg.linear_acceleration.y, msg.angular_velocity.z)
                    imu_t = msg

            # Save data in container
            if img_t is not None and steer_t is not None and throttle_t is not None and pose_t is not None and imu_t is not None:

                if data_idx < INITIAL_DATA_PASS_SIZE:
                    print("Not save this data! data_idx :", data_idx)
                    continue
                    
                # # Save interval
                # if num_data % SAVE_INTERVAL != 0:
                #     # Reset
                #     img_t = None
                #     steer_t = True
                #     throttle_t = True
                #     pose_t = True
                #     imu_t = True
                #     continue

                # Save image: Writing image to the directory mentioned while executing script
                if SAVE_IMG:
                    cv2.imwrite(IMG_DIR+'/frame_' + BAG_NAME + '_' + str(img_t_header.stamp) + '.jpg', img)

                # Save imu, route data information as per requirement
                orient_q = pose_t.orientation
                _, _, pose_yaw = euler_from_quaternion([orient_q.x, orient_q.y, orient_q.z, orient_q.w])

                # Set done False (check terminal state in custom_dataset)
                done = False

                # [img_stamp, img_path, imu_accel_x, imu_accel_y, imu_yaw_rate, pose_x, pose_y, pose_yaw, closest_wpt_x, closest_wpt_y, is_arrive]
                thewriter.writerow([str(img_t_header.stamp),
                                    'frame_' + BAG_NAME + '_' + str(img_t_header.stamp)+'.jpg',
                                    steer_t,
                                    throttle_t,
                                    imu_t.linear_acceleration.x,
                                    imu_t.linear_acceleration.y,
                                    imu_t.angular_velocity.z,
                                    pose_t.position.x,
                                    pose_t.position.y,
                                    pose_yaw,
                                    closest_wpt_x,
                                    closest_wpt_y,
                                    is_arrive,
                                    done,
                                    ])

                # if num_data >= MAX_NUM_DATA:
                #     print("Max num of data.")
                #     break

                if SHOW_PARSED_DATA:
                    print("Save img, steer, throttle, pose, imu")
                    print("img :", img.shape, "steer :", steer_t, "throttle :", throttle_t, "pose :", pose_t.position.x, "imu :", imu_t.linear_acceleration.x)

                # Reset
                img_t = None
                steer_t = True
                throttle_t = True
                pose_t = True
                imu_t = True
        
        bag.close()
        f.close()

    # ================================================ #
    # ===== 2. Post-processing (target waypoint) ===== #
    # ================================================ #
    if POSTPROCESSING_TARGET_WPT:
        state_data = pd.read_csv(STATE_DATA_PATH)
        processed_data = pd.read_csv(STATE_DATA_PATH)
        # print(state_data)

        # - Find arriving waypoint and split data w.r.t. arriving point
        # - Set the terminal route point
        # - 1) if not arriving the terminal point yet, copy the terminal point
        # - 2) if arrving the terminal point, find the new terminal point whose heading error is min
        state_data_pose_x   = state_data['pose_x'].to_numpy()
        state_data_pose_y   = state_data['pose_y'].to_numpy()
        state_data_pose_yaw = state_data['pose_yaw'].to_numpy()
        state_data_closest_wpt_x = state_data['closest_wpt_x'].to_numpy()
        state_data_closest_wpt_y = state_data['closest_wpt_y'].to_numpy()
        state_data_is_arrive     = state_data['is_arrive'].to_numpy()

        state_data_target_x_t = np.ones(state_data_closest_wpt_x.shape)
        state_data_target_y_t = np.ones(state_data_closest_wpt_y.shape)

        i = 0                   # for indexing
        check_arrive_ind = True # for finding first is_arrive index
        prev_arrive_ind = 0     # for indexing from prev~curr arriving points
        for closest_wpt_x, closest_wpt_y, is_arrive in zip(state_data_closest_wpt_x, state_data_closest_wpt_y, state_data_is_arrive):
            # if is_arrive, set the target wpt as closest_wpt_x,y until that arriving waypoint.
            if is_arrive:
                if check_arrive_ind:
                    state_data_target_x_t[prev_arrive_ind:i] = closest_wpt_x
                    state_data_target_y_t[prev_arrive_ind:i] = closest_wpt_y
                    # update previous arrving point index
                    prev_arrive_ind = i
                    check_arrive_ind = False # no checking after finding first arriving point
                else:
                    pass
            else:
                check_arrive_ind = True # checking the next first arrving point
            
            # if terminal point, check whether the terminal point is arrived or not
            if i == len(state_data_target_x_t) - 1:
                # if closest_wpt != prev_arrived_wpt, not yet arriving the terminal wpt
                if (closest_wpt_x != state_data_closest_wpt_x[prev_arrive_ind]) and (closest_wpt_y != state_data_closest_wpt_y[prev_arrive_ind]):
                    state_data_target_x_t[prev_arrive_ind:] = closest_wpt_x
                    state_data_target_y_t[prev_arrive_ind:] = closest_wpt_y
                # if closest_wpt == prev_arrived_wpt, arrived the terminal wpt.
                else:
                    # find a wpt whose heading error is min.
                    min_head_wpt_x, min_head_wpt_y = find_min_heading_dist_cost_error_wpt(state_data_pose_x[-1], state_data_pose_y[-1], state_data_pose_yaw[-1])
                    state_data_target_x_t[prev_arrive_ind:] = min_head_wpt_x
                    state_data_target_y_t[prev_arrive_ind:] = min_head_wpt_y

            # for indexing
            i += 1
        # save processed data
        processed_data['target_wpt_x_t'] = state_data_target_x_t
        processed_data['target_wpt_y_t'] = state_data_target_y_t

        # - Set the two target waypoint
        state_data_target_x_t1 = np.ones(state_data_closest_wpt_x.shape)
        state_data_target_y_t1 = np.ones(state_data_closest_wpt_y.shape)

        i = 0                   # for indexing
        check_arrive_ind = True # for finding first is_arrive index
        prev_arrive_ind = 0     # for indexing from prev~curr arriving points
        for target_x_t, target_y_t, is_arrive in zip(state_data_target_x_t, state_data_target_y_t, state_data_is_arrive):
            if is_arrive:
                if check_arrive_ind:
                    state_data_target_x_t1[prev_arrive_ind:i] = target_x_t
                    state_data_target_y_t1[prev_arrive_ind:i] = target_y_t
                    # update previous arrving point index
                    prev_arrive_ind = i
                    check_arrive_ind = False # no checking after finding first arriving point
                else:
                    pass
            else:
                check_arrive_ind = True # checking the next first arrving point

            # if terminal point, set the target_wpt_t1 as the target_wpt_t
            if i == len(state_data_target_x_t1) - 1:
                state_data_target_x_t1[prev_arrive_ind:] = target_x_t
                state_data_target_y_t1[prev_arrive_ind:] = target_y_t

            # for indexing
            i += 1
        # Save processed data
        processed_data['target_wpt_x_t1'] = state_data_target_x_t1
        processed_data['target_wpt_y_t1'] = state_data_target_y_t1

        # - Get the trajectory as training data
        print(processed_data)
        if SAVE_POSTPROCESSED_TARGET_WPT:
            processed_data.to_csv(STATE_DATA_PATH, mode='w', index=False)

    # ====================================================== #
    # ===== 3. Post-processing (topological BEV image) ===== #
    # ====================================================== #
    if POSTPROCESSING_BEV_MAP:
        # Global route from collected trajectory
        global_route = []
        
        # - initial wpt (closest wpt of roadmap and closest wpt of )
        current_wpt_x = processed_data['closest_wpt_x'][0]
        current_wpt_y = processed_data['closest_wpt_y'][0]
        global_route.append((current_wpt_x, current_wpt_y))

        # - loop for stacking waypoints
        for target_x, target_y in zip(processed_data['target_wpt_x_t'], processed_data['target_wpt_y_t']):
            # -- stack wpt only if wpt is different from current wpt
            if (current_wpt_x != target_x) and (current_wpt_y != target_y):
                global_route.append((target_x, target_y))
                # -- update current wpt
                current_wpt_x = target_x
                current_wpt_y = target_y

        # Global route to Bird-Eye's View (BEV)
        # - define bev topo map transfer
        initial_ego_pose_x = processed_data['pose_x'][0]
        initial_ego_pose_y = processed_data['pose_y'][0]
        
        topo_map_transfer = TopoMapTransfer(init_pose_x=initial_ego_pose_x,
                                            init_pose_y=initial_ego_pose_y,
                                            road_map_wpts=road_map,
                                            global_route_wpts=global_route,
                                            bev_img_resize=(60,60),
                                            bev_route_progress=15)
        for pose_x, pose_y, pose_yaw, raw_img_path in zip(processed_data['pose_x'], processed_data['pose_y'], processed_data['pose_yaw'], processed_data['img_path']):
            # - from current pose to bev map
            bev_map_resize = topo_map_transfer.get_bev_map(pose_x, pose_y, pose_yaw)
            if SHOW_BEV_MAP:
                cv2.imshow('bev_image', bev_map_resize)
                cv2.waitKey(3)
            # - save bev map (same name with raw image)
            if SAVE_POSTPROCESSED_BEV_MAP:
                cv2.imwrite(BEV_DIR + '/' + raw_img_path, bev_map_resize)

