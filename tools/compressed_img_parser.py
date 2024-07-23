#! /usr/bin/python

import os
import sys
import cv2
import csv
import rospy, rosbag
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Verifying output directory where Images and metadata(csv) will be stored
img_dir = None
label_dir = None

if len(sys.argv) == 3:
    img_dir = sys.argv[1]
    label_dir = sys.argv[2]
    if not os.path.exists(sys.argv[1]):
        os.makedirs(img_dir)

    else:
        print sys.argv[1], sys.argv[2], 'already exists.'
        # sys.exit()

else:
    print 'specify img directory and label directory'
    sys.exit()

bridge = CvBridge()

def main():

    # Topic name 
    topicName0 = '/gmsl_camera1/compressed'

    # Bag file 
    inbag_name ='/media/seong/Storage_2/rosbag/intersection_test/200929/seong_bag/fine_lstm_sac/2020-09-29-18-41-26.bag'
    bag = rosbag.Bag(inbag_name)

    # Data num limit
    MAX_NUM_DATA = 1e+10
    SAVE_INTERVAL = 2
    num_data = 0

    for topic, msg, t in bag.read_messages():

        if(topic == topicName0 ):
            try:
                # Convert compressed image to RAW
                img = bridge.compressed_imgmsg_to_cv2(msg)
                num_data += 1
                print("Image num :", num_data, "image_size :", img.shape)

            except CvBridgeError, e:
                print(e)

            else:
                # Save interval
                if SAVE_INTERVAL is not None:
                    if num_data % SAVE_INTERVAL != 0:
                        continue
                    
                # Writing image to the directory mentioned while executing script
                cv2.imwrite(img_dir+'/frame_' + str(msg.header.stamp) + '.jpg', img)

                # Saving metadata information as per requirement
                with open(label_dir+'/metadata.csv','ab') as f:
                    thewriter = csv.writer(f)
                    thewriter.writerow([str(msg.header.seq), str(msg.header.stamp), 'frame_'+ str(msg.header.stamp)+'.jpg'])

                if num_data >= MAX_NUM_DATA:
                    print("Max num of data.")
                    break

    bag.close()
    rospy.spin()

if __name__ == "__main__":
    main()
