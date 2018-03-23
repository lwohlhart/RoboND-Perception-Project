#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    noise_filter = pcl_cloud.make_statistical_outlier_filter()
    noise_filter.set_mean_k(10)
    noise_filter.set_std_dev_mul_thresh(0.001)
    cloud_filtered = noise_filter.filter()
    #rospy.loginfo("point_cloud_size: {} noise_filtered_size:{}".format(pcl_cloud.size, cloud_filtered.size))

    # TODO: Voxel Grid Downsampling
    voxel_filter = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    voxel_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = voxel_filter.filter()

    # TODO: PassThrough Filter
    passthrough_filter = cloud_filtered.make_passthrough_filter()
    passthrough_filter.set_filter_field_name('z')
    passthrough_filter.set_filter_limits(0.6, 1.1)
    cloud_filtered = passthrough_filter.filter()

    passthrough_filter = cloud_filtered.make_passthrough_filter()
    passthrough_filter.set_filter_field_name('y')
    passthrough_filter.set_filter_limits(-0.5, 0.5)
    cloud_filtered = passthrough_filter.filter()

    # TODO: RANSAC Plane Segmentation
    ransac_plane_segmenter = cloud_filtered.make_segmenter()  
    ransac_plane_segmenter.set_model_type(pcl.SACMODEL_PLANE)
    ransac_plane_segmenter.set_method_type(pcl.SAC_RANSAC)
    ransac_plane_segmenter.set_distance_threshold(0.01)

    plane_indices, _ = ransac_plane_segmenter.segment()     


    # TODO: Extract inliers and outliers
    objects_cloud = cloud_filtered.extract(plane_indices, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(objects_cloud)
    tree = white_cloud.make_kdtree()

    euclid_cluster_extractor = white_cloud.make_EuclideanClusterExtraction()
    euclid_cluster_extractor.set_ClusterTolerance(0.01)
    euclid_cluster_extractor.set_MinClusterSize(100)
    euclid_cluster_extractor.set_MaxClusterSize(5000)
    euclid_cluster_extractor.set_SearchMethod(tree)

    object_clusters_indices = euclid_cluster_extractor.Extract()

    clusters_color = get_color_list(len(object_clusters_indices))    
    color_clusters_point_list = []

    for j, clusters_indices in enumerate(object_clusters_indices):
        for i, point_index in enumerate(clusters_indices):
            color_clusters_point_list.append([white_cloud[point_index][0],
                                            white_cloud[point_index][1],
                                            white_cloud[point_index][2],
                                            rgb_to_float(clusters_color[j])])

    clusters_cloud = pcl.PointCloud_PointXYZRGB()
    clusters_cloud.from_list(color_clusters_point_list)

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # TODO: Convert PCL data to ROS messages
    ros_objects_cloud = pcl_to_ros(objects_cloud)
    ros_clusters_cloud = pcl_to_ros(clusters_cloud)
    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_objects_cloud)
    pcl_clusters_pub.publish(ros_clusters_cloud)
    

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    
    detected_objects_labels = []
    detected_objects = []
    
    for j, cluster_indices in enumerate(object_clusters_indices):
        # for i, point_index in enumerate(clusters_indices):
        # Grab the points for the cluster
        cluster_points = objects_cloud.extract(cluster_indices, negative=False)
        ros_cluster_points = pcl_to_ros(cluster_points)
        # Compute the associated feature vector
        col_hist = compute_color_histograms(ros_cluster_points, using_hsv=True)
        cluster_normals = get_normals(ros_cluster_points)
        norm_hist = compute_normal_histograms(cluster_normals)
        features = np.concatenate((col_hist, norm_hist))
        # Make the prediction

        prediction = clf.predict(scaler.transform(features.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_position = list(np.mean(cluster_points, axis=0)[:3])
        label_position[2] += 0.4
        object_markers_pub.publish(make_label(label, label_position, j, color=np.array(clusters_color[j])/255.))

        #pcl.computeCentroid()
        # Add the detected object to the list of detected objects.
        detected_object = DetectedObject()
        detected_object.cloud = ros_cluster_points
        detected_object.label = label 
        detected_objects.append(detected_object)
    
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters
    pick_list = rospy.get_param('/object_list')    

    # prepare dropbox definitions and place_poses to scatter the contents in the box
    dropboxes = {}
    for dropbox in rospy.get_param('/dropbox'):
        dropbox['droplist'] = []
        dx = np.linspace(-0.2, 0.2, 5)
        dy = np.linspace(-0.1, 0.1, 3)
        dropbox['place_poses'] = zip(*(x.flat for x in np.meshgrid(dx,dy,0))) + np.array(dropbox['position'])
        np.random.shuffle(dropbox['place_poses'])
        dropboxes[dropbox['group']] = dropbox

    test_scene_num = Int32()
    test_scene_num.data = 1
    if len(pick_list) == 5:
        test_scene_num.data = 2
    elif len(pick_list) == 8:
        test_scene_num.data = 3

    # TODO: Parse parameters into individual variables
    
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    for i, detected_object in enumerate(object_list):
        labels.append(detected_object.label)
        points_arr = ros_to_pcl(detected_object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])

    pick_object_yamls = []
    # TODO: Loop through the pick list
    for i, pick_object in enumerate(pick_list):
        object_name = String()
        object_name.data = pick_object['name']
        object_group = pick_object['group']

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        pick_pose = Pose()
        try:
            idx = labels.index(object_name.data)
            pick_pose.position.x = np.asscalar(centroids[idx][0])
            pick_pose.position.y = np.asscalar(centroids[idx][1])
            pick_pose.position.z = np.asscalar(centroids[idx][2])
            del labels[idx]
            del centroids[idx]
        except ValueError:
            # object not found in detected list; skip it
            rospy.loginfo('Object {} not found in detected objects, will be skipped in picklist'.format(object_name.data))
            continue
        
        # TODO: Create 'place_pose' for the object
        place_poses = dropboxes[object_group]['place_poses']
        pp = place_poses[np.mod(len(dropboxes[object_group]['droplist']), len(place_poses))]
        place_pose = Pose()        
        place_pose.position.x = np.asscalar(pp[0])
        place_pose.position.y = np.asscalar(pp[1])
        place_pose.position.z = np.asscalar(pp[2])
        dropboxes[object_group]['droplist'].append(object_name.data)

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        arm_name.data = dropboxes[object_group]['name']

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        pick_object_yaml = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        pick_object_yamls.append(pick_object_yaml)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('output_{}.yaml'.format(test_scene_num.data), pick_object_yamls)



if __name__ == '__main__':

    # TODO: ROS node initialization
    nh = rospy.init_node('pick_place_perception', anonymous=True)
    
    # TODO: Create Subscribers
    pcl_world_sub = rospy.Subscriber('/pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    pcl_clusters_pub = rospy.Publisher('/pcl_clusters', PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)


    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))    
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']


    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
