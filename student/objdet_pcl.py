# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import zlib
import open3d as o3d
import torch
import matplotlib.pyplot as plt

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # 1. Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # 2. Assign the points from your numpy array
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
    
    # 3. Visualize and block execution until the window is closed
    print("Displaying point cloud. Close the window to allow the script to continue.")
    o3d.visualization.draw_geometries([pcd])

    # step 1 : initialize open3d with key callback and create window
    
    # step 2 : create instance of open3d point-cloud class

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)

    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")
# step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    
    # step 2 : extract the range and the intensity channel from the range image
    ri_range = ri[:,:,0]
    ri_intensity = ri[:,:,1]
    
    # step 3 : set values <0 to zero
    ri_range[ri_range < 0] = 0.0
    ri_intensity[ri_intensity < 0] = 0.0
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    ri_range = (ri_range / np.amax(ri_range) * 255).astype(np.uint8)

    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    p1, p99 = np.percentile(ri_intensity, [1, 99])
    ri_intensity = np.clip((ri_intensity - p1) * 255 / (p99 - p1), 0, 255).astype(np.uint8)

    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack((ri_range, ri_intensity))
    
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity



def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates   
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)
    # y-coordinates are shifted such that the ego vehicle is centered in the BEV map.

    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    # Sorting by -z ensures that the point with the highest z-value (the top-most point) comes first for each (x,y) grid cell.
    idx_height = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_top = lidar_pcl_cpy[idx_height]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ## also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, unique_indices, counts = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_pcl_top[unique_indices]

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    # Before normalization, we map the intensity values to the correct BEV grid cells.
    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3]

    ## step 5 : normalize the intensity map using percentiles to mitigate the influence of outliers
    # This enhances contrast and makes objects like vehicles stand out more clearly.
    p1, p99 = np.percentile(intensity_map[intensity_map > 0], [1, 99])
    intensity_map = np.clip((intensity_map - p1) * 255 / (p99 - p1), 0, 255)

    ####### ID_S2_EX2 END #######


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height, configs.bev_width))

    ## step 2 : assign the height value of each unique entry in lidar_pcl_top to the height map
    # We normalize the height by the predefined z-limits to scale the values between 0 and 255.
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(configs.lim_z[1] - configs.lim_z[0]) * 255

    ####### ID_S2_EX3 END #######       

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height, configs.bev_width))
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) * 255
    unique_bev_coords = lidar_pcl_top[:, 0:2].astype(int)
    density_map[unique_bev_coords[:, 0], unique_bev_coords[:, 1]] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map # r_map (density)
    bev_map[1, :, :] = height_map  # g_map (height)
    bev_map[0, :, :] = intensity_map # b_map (intensity)

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps) # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

    # Create intensity heatmap
    plt.figure(figsize=(15, 5))
    
    # Intensity map visualization
    plt.subplot(1, 2, 1)
    plt.imshow(intensity_map, cmap='hot', origin='lower')
    plt.colorbar(label='Intensity Values')
    plt.title('BEV Intensity Map')
    plt.xlabel('BEV X Coordinate')
    plt.ylabel('BEV Y Coordinate')
    
    # Height map visualization  
    plt.subplot(1, 2, 2)
    plt.imshow(height_map, cmap='viridis', origin='lower')
    plt.colorbar(label='Height Values')
    plt.title('BEV Height Map')
    plt.xlabel('BEV X Coordinate')
    plt.ylabel('BEV Y Coordinate')
    
    plt.tight_layout()
    plt.show()

    return input_bev_maps
