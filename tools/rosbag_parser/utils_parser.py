import numpy as np
import os
import glob
import natsort

# Parameters
MINIMUM_DIST = 2

# Waypoints
WPT00 = (7.26, -1.18)
WPT01 = (11.07, 21.13)
WPT02 = (-3.56, 23.85)
WPT03 = (-7.31, 1.44)
WPT04 = (4.18, -20.41)
WPT05 = (2.88, -25.11)
WPT06 = (1.7, -34.15)
WPT07 = (9.32, -26.19)
WPT08 = (7.91, -35.04)
WPT09 = (30.71, -39.12)
WPT10 = (35.57, -11.50)
WPT11 = (37.40, -0.52) # 37.52, -0.12
WPT12 = (40.65, 15.94)
WPT13 = (32.14, -1.85)
WPT14 = (26.39, 9.33)
WPT15 = (28.80, 18.07)
WPT16 = (22.95, 19.34)

waypoints = []
waypoints.append(WPT00)
waypoints.append(WPT01)
waypoints.append(WPT02)
waypoints.append(WPT03)
waypoints.append(WPT04)
waypoints.append(WPT05)
waypoints.append(WPT06)
waypoints.append(WPT07)
waypoints.append(WPT08)
waypoints.append(WPT09)
waypoints.append(WPT10)
waypoints.append(WPT11)
waypoints.append(WPT12)
waypoints.append(WPT13)
waypoints.append(WPT14)
waypoints.append(WPT15)
waypoints.append(WPT16)

waypoints = np.array(waypoints)

def find_closest_wpt(x, y):
    position_t = np.array([x, y])
    position_diff = waypoints - position_t
    dist = np.sqrt(np.power(position_diff[:,0], 2) + np.power(position_diff[:,1], 2))
    min_ind = np.argmin(dist)
    min_dist = dist[min_ind]
    return waypoints[min_ind, 0], waypoints[min_ind, 1], min_dist

def find_arrving_wpt(x, y):
    min_x, min_y, min_d = find_closest_wpt(x, y)
    if min_d < MINIMUM_DIST:
        return True, min_x, min_y, min_d
    else:
        return False, min_x, min_y, min_d

def find_min_heading_dist_cost_error_wpt(x, y, yaw):
    position_t = np.array([x, y])
    position_diff = waypoints - position_t
    # dist error
    dist_error = np.sqrt(np.power(position_diff[:,0], 2) + np.power(position_diff[:,1], 2))
    # heading error
    azimuth_to_wpt = np.arctan2(position_diff[:,1], position_diff[:,0])
    heading_error = np.absolute(azimuth_to_wpt - yaw)
    # total error
    error = heading_error + 0.1 * dist_error
    min_ind = np.argmin(error)
    return waypoints[min_ind, 0], waypoints[min_ind, 1]

def get_file_path_in_dir(dir_path, file_type):
    file_path_list = glob.glob(dir_path + '/*.' + file_type)
    file_path_list = natsort.natsorted(file_path_list)
    return file_path_list

def get_file_name_in_dir(dir_path, file_type):
    file_name_list = [f for f in os.listdir(dir_path) if f.endswith('.'+file_type)]
    file_name_list = natsort.natsorted(file_name_list)
    return file_name_list


def main():
    x, y = 4.38, -14.27
    find_closest_wpt(x, y)

    file_path = get_file_path_in_dir('/home/seong/behavior_aware_imitation_learning/dataset/bag/2021-05-02-01-25-10', 'bag')
    file_name = get_file_name_in_dir('/home/seong/behavior_aware_imitation_learning/dataset/bag/2021-05-02-01-25-10', 'bag')

    print(file_path)
    print(file_name)

if __name__ == '__main__':
    main()
