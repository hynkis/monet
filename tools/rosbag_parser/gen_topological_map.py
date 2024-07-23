import math
import time
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt

LOOK_BEHIND_IND = 100 # [index] 0 only, otherwise, delayed route path (red)
LOOK_AHEAD_IND = 100
LOOK_AHEAD_PROGRESS = 25 # [m]

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

road_map = []
road_map.append(WPT00)
road_map.append(WPT01)
road_map.append(WPT02)
road_map.append(WPT03)
road_map.append(WPT00)
road_map.append(WPT04)
road_map.append(WPT05)
road_map.append(WPT06)
road_map.append(WPT08)
road_map.append(WPT09)
road_map.append(WPT10)
road_map.append(WPT11)
road_map.append(WPT12)
road_map.append(WPT15)
road_map.append(WPT16)

road_map.append(WPT01)
road_map.append(WPT00)
road_map.append(WPT04)
road_map.append(WPT07)
road_map.append(WPT08)
road_map.append(WPT06)
road_map.append(WPT05)
road_map.append(WPT07)

road_map.append(WPT08)
road_map.append(WPT09)
road_map.append(WPT10)
road_map.append(WPT13)
road_map.append(WPT14)
road_map.append(WPT16)
road_map.append(WPT15)
road_map.append(WPT14)
road_map.append(WPT13)
road_map.append(WPT11)

global_route = []
global_route.append(WPT03)
global_route.append(WPT00)
global_route.append(WPT04)
global_route.append(WPT07)
global_route.append(WPT08)
global_route.append(WPT06)
global_route.append(WPT05)
global_route.append(WPT07)
global_route.append(WPT08)
global_route.append(WPT09)

# Global functions
def global2local(ego_x, ego_y, ego_yaw, x_list, y_list):
    # Translational transform
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    x_list = x_list - ego_x
    y_list = y_list - ego_y

    # Rotational transform
    rot_theta = -ego_yaw
    c_theta = np.cos(rot_theta)
    s_theta = np.sin(rot_theta)

    rot_mat = np.array([[c_theta, -s_theta],
                        [s_theta, c_theta]])

    output_xy_list = np.matmul(rot_mat, np.array([x_list, y_list]))
    output_x_list = output_xy_list[0,:]
    output_y_list = output_xy_list[1,:]

    return output_x_list, output_y_list
    
def find_nearest_point(ego_x, ego_y, x_list, y_list):
    dist = np.zeros(len(x_list))
    for i in range(len(x_list)):
        dist[i] = calc_dist(x_list[i], y_list[i], ego_x, ego_y)
    
    near_ind = np.argmin(dist)

    return x_list[near_ind], y_list[near_ind], near_ind
    
def calc_dist(tx, ty, ix, iy):
    return math.sqrt( (tx-ix)**2 + (ty-iy)**2 )
    
def pix2local(x_pix, y_pix, width, height):
    return -(y_pix - height), -(x_pix - 0.5*width)

def local2pixel(local_x, local_y, local_bias_x, img_width, img_height, img_res):
    """
    img_res: n pixel / 1 meter
    """
    pixel_x = 0.5*img_width - img_res * local_y
    pixel_y = img_height - img_res * (local_x) + img_res * local_bias_x # local_bias_x*img_res for bias
    
    pixel_x = np.maximum(0, np.minimum(pixel_x, img_width))
    pixel_y = np.maximum(0, np.minimum(pixel_y, img_height))

    return np.int32(pixel_x), np.int32(pixel_y)

class TopoMapTransfer(object):
    def __init__(self,
                 init_pose_x,
                 init_pose_y,
                 road_map_wpts,
                 global_route_wpts,
                 bev_img_resize = (60,60),
                 bev_route_progress = LOOK_AHEAD_PROGRESS):
        # Min/Max for topological bev map
        self.min_local_x = -4.0 # -0.0
        self.max_local_x =  8.0 # 10.0
        self.min_local_y = -6.0 # -5.0
        self.max_local_y =  6.0 #  5.0

        self.img_size = (100,100)
        self.img_res = 10 # n pixel / 1 meter

        self.bev_img_resize = bev_img_resize
        self.bev_route_progress = bev_route_progress # consider max local x and y length

        # for road map
        self.wpts = road_map_wpts # list [(x0,y0), (x0,y0), ...]
        self.wpts_x = np.array([x for (x, _) in road_map_wpts])
        self.wpts_y = np.array([y for (_, y) in road_map_wpts])
        self.wpts_xi, self.wpts_yi = self.interpolate_wpts(self.wpts_x, self.wpts_y) # np array [[x,y], [x,y], ...]
        
        # for global route
        # - get extended global route (from initial ego's position)
        nearest_roadmap_x, nearest_roadmap_y, _ = find_nearest_point(init_pose_x, init_pose_y, self.wpts_xi, self.wpts_yi)
        global_route_wpts.insert(0, (nearest_roadmap_x, nearest_roadmap_y))

        self.route_wpts = global_route_wpts # list [(x0,y0), (x0,y0), ...]
        self.route_wpts_x = np.array([x for (x, _) in self.route_wpts])
        self.route_wpts_y = np.array([y for (_, y) in self.route_wpts])
        self.route_wpts_xi, self.route_wpts_yi = self.interpolate_wpts(self.route_wpts_x, self.route_wpts_y) # np array [[x,y], [x,y], ...]

    def interpolate_wpts(self, wpts_x, wpts_y):
        total_wpts_xi, total_wpts_yi = [], []
        num_wpts = len(wpts_x)

        # Stack piecewise linear intepolated points
        for i in range(num_wpts-1):
            wpt_x0, wpt_y0 = wpts_x[i], wpts_y[i]
            wpt_x1, wpt_y1 = wpts_x[i+1], wpts_y[i+1]

            # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
            # is needed in order to force the spline fit to pass through all the input points.
            
            tck, _ = interpolate.splprep([[wpt_x0, wpt_x1], [wpt_y0, wpt_y1]], s=0.0, k=1)
            # tck, u = interpolate.splprep([wpts_x, wpts_y], s=0, per=1, k=1)

            # evaluate the spline fits for 1000 evenly spaced distance values
            wpts_xi, wpts_yi = interpolate.splev(np.linspace(0, 1, 100), tck)

            # cumulate lists
            total_wpts_xi = np.hstack([total_wpts_xi, wpts_xi])
            total_wpts_yi = np.hstack([total_wpts_yi, wpts_yi])

        return total_wpts_xi, total_wpts_yi

    def get_local_wpts(self, ego_x, ego_y, ego_yaw, x_list, y_list):
        # Translational transform
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        x_list = x_list - ego_x
        y_list = y_list - ego_y

        # Rotational transform
        rot_theta = -ego_yaw
        c_theta = np.cos(rot_theta)
        s_theta = np.sin(rot_theta)

        rot_mat = np.array([[c_theta, -s_theta],
                            [s_theta, c_theta]])

        output_xy_list = np.matmul(rot_mat, np.array([x_list, y_list]))
        output_x_list = output_xy_list[0,:]
        output_y_list = output_xy_list[1,:]

        return output_x_list, output_y_list

    def get_truncated_wpts(self, x_list, y_list, progress_dist, start_ind=0):
        xy_dist = np.zeros(len(x_list))
        # compute distances
        xy_list = np.vstack([x_list, y_list]) # (N,2)
        xy_del = xy_list[:, 1:] - xy_list[:, :-1] # (N-1,2)
        xy_pow = np.sum(np.power(xy_del, 2), axis=0) # (N-1,2)
        xy_dist[1:] = np.sqrt(xy_pow)
        # compute cumulative distances
        xy_dist = np.cumsum(xy_dist)
        # remove bias (cumsum of the start_ind)
        xy_dist -= xy_dist[start_ind]
        # find index where cum.dist ~= progress_dist.
        # - (with considering start_ind: cumulative sum of start_ind should be 0!)
        # - should search from the closest point
        # - return truncated index if
        # -- 1) the index where cum_dist > progress_dist
        # -- 2) the terminal index if rest of wpts are shorter than progress_dist 
        truncated_ind = 0
        for ind, dist in enumerate(xy_dist):
            # pass before start_ind
            if ind < start_ind:
                continue
            truncated_ind = ind
            dist_error = dist - progress_dist
            if dist_error > 0:
                break

        return x_list[:truncated_ind], y_list[:truncated_ind]
    
    def region_of_interest(self, local_wpts_x, local_wpts_y):
        # Region of interest w.r.t. x axis
        roi_x_mask = (local_wpts_x > self.min_local_x) & (local_wpts_x < self.max_local_x)
        roi_local_wpts_x = local_wpts_x[roi_x_mask]
        roi_local_wpts_y = local_wpts_y[roi_x_mask]

        # Region of interest w.r.t. y axis
        roi_y_mask = (roi_local_wpts_y > self.min_local_y) & (roi_local_wpts_y < self.max_local_y)
        roi_local_wpts_x = roi_local_wpts_x[roi_y_mask]
        roi_local_wpts_y = roi_local_wpts_y[roi_y_mask]

        return roi_local_wpts_x, roi_local_wpts_y

    def get_bev_map(self, ego_x, ego_y, ego_yaw):
        # BEV image
        # - road  : white
        # - route : red
        # for topological bev map
        bev_map = np.zeros((self.img_size[0], self.img_size[1], 3), np.uint8)

        # Find nearest point in global route
        ego_x_near, ego_y_near, near_ind = find_nearest_point(ego_x, ego_y, self.route_wpts_xi, self.route_wpts_yi)

        # Get local map (road map & global route)
        local_road_wpts_x, local_road_wpts_y = self.get_local_wpts(ego_x_near, ego_y_near, ego_yaw, self.wpts_xi, self.wpts_yi)
        local_route_wpts_x, local_route_wpts_y = self.get_local_wpts(ego_x_near, ego_y_near, ego_yaw, self.route_wpts_xi, self.route_wpts_yi)
        
        # Get truncated local route
        # - cut until bev route progress distance
        local_route_wpts_x, local_route_wpts_y = self.get_truncated_wpts(local_route_wpts_x,
                                                            local_route_wpts_y, self.bev_route_progress, start_ind=near_ind)
        # - start from the closest point with look-behind index
        look_behind_ind = LOOK_BEHIND_IND
        start_ind = max(0, near_ind-look_behind_ind)
        local_route_wpts_x = local_route_wpts_x[start_ind:]
        local_route_wpts_y = local_route_wpts_y[start_ind:]

        # ROI (road map & global route)
        local_road_wpts_x, local_road_wpts_y = self.region_of_interest(local_road_wpts_x, local_road_wpts_y)
        local_route_wpts_x, local_route_wpts_y = self.region_of_interest(local_route_wpts_x, local_route_wpts_y)

        # position to pixel (road map & global route)
        pixel_road_x, pixel_road_y = local2pixel(local_road_wpts_x, local_road_wpts_y, self.min_local_x,
                                                 self.img_size[0], self.img_size[1], self.img_res)
        pixel_route_x, pixel_route_y = local2pixel(local_route_wpts_x, local_route_wpts_y, self.min_local_x,
                                                   self.img_size[0], self.img_size[1], self.img_res)
        pixel_ego_x, pixel_ego_y = local2pixel(0, 0, self.min_local_x,
                                                 self.img_size[0], self.img_size[1], self.img_res)
        # Transform to pair
        pixel_road_pair  = np.vstack([pixel_road_x, pixel_road_y]).T # (N,2)
        pixel_route_pair = np.vstack([pixel_route_x, pixel_route_y]).T # (N,2)

        num_pixel_road = pixel_road_pair.shape[0]
        for i in range(num_pixel_road):
            # white for roadmap
            cv2.circle(bev_map, tuple(pixel_road_pair[i,:]), radius=3, color=(255,255,255), thickness=-1) # BGR

        num_pixel_route = pixel_route_pair.shape[0]
        for i in range(num_pixel_route):
            # red for route
            cv2.circle(bev_map, tuple(pixel_route_pair[i,:]), radius=3, color=(0,0,255), thickness=-1) # BGR

        # green for ego
        cv2.circle(bev_map, tuple([pixel_ego_x, pixel_ego_y]), radius=4, color=(0,255,0), thickness=-1) # BGR

        # Numpy to CV image
        bev_map_resize = cv2.resize(bev_map, self.bev_img_resize)

 
        return bev_map_resize


def main():

    # Example ego pose
    ego_x = 4.6   # 5.3 #33
    ego_y = -20.9 # -17.0 #-5
    ego_yaw = np.deg2rad(-100)
    init_pose_x = 10.0 # 33
    init_pose_y = 0.0 # -5
    
    topo_map_transfer = TopoMapTransfer(init_pose_x, init_pose_y, road_map, global_route)

    tic = time.time()
    # Get topological bev map
    bev_map_resize = topo_map_transfer.get_bev_map(ego_x, ego_y, ego_yaw)
    toc = time.time()
    print("time :", toc - tic)


    plt.figure()
    plt.axis('equal')
    plt.grid()
    plt.plot(topo_map_transfer.wpts_x, topo_map_transfer.wpts_y, 'r.')
    plt.plot(topo_map_transfer.wpts_xi, topo_map_transfer.wpts_yi, 'b-')
    plt.title('global map')
    plt.show()

    cv2.imshow('bev_map_resize', bev_map_resize)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
