#!/usr/bin/env python

# Original Author of AB3DMOT repo: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# Modified for ROS by: Pardis Taghavi
# email: taghavi.pardis@gmail.com

from __future__ import print_function
import numpy as np, copy, math,sys,argparse

import os
import sys
import rospy
import torch
import time

# Get the directory path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the desired path relative to the current file's location
libs_dir = os.path.join(current_dir, "AB3DMOT/AB3DMOT_libs")
xinshuo_lib=os.path.join(current_dir,"AB3DMOT/Xinshuo_PyToolbox")
# Add the path to sys.path
sys.path.append(libs_dir)
sys.path.append(xinshuo_lib)



from AB3DMOT_libs.matching import data_association
from AB3DMOT_libs.vis import vis_obj

from kalman_filter import KF
from kitti_oxts import get_ego_traj, egomotion_compensation_ID
from kitti_oxts import load_oxts ,_poses_from_oxts
from kitti_calib import Calibration
from model import AB3DMOT
from box import Box3D


#import std_msgs.msg
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.core.points import get_points_type
#from geometry_msgs.msg import Quaternion
from scipy.spatial.transform import Rotation as R
import ros_numpy
from collections import namedtuple
from sensor_msgs.msg import PointCloud2, Imu, NavSatFix#, Image, CameraInfo
#from geometry_msgs.msg import TwistStamped, Quaternion
import message_filters
#import torchvision.transforms as transforms
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
#from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray, VisionInfo
from visualization_msgs.msg import Marker,MarkerArray
lim_x=[-50, 50]
lim_y=[-25,25]
lim_z=[-3,3]
np.set_printoptions(suppress=True, precision=3)
points_class = get_points_type('LIDAR')

calib_path= os.path.join(current_dir, "data/0000.txt")
config_path=os.path.join(current_dir, "mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py")
model_path=os.path.join(current_dir, "mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth")


# A Baseline of 3D Multi-Object Tracking
class AB3DMOT():			  	
	def __init__(self,  ID_init=0, model=None, pointcloud_topic=None):                    
		calib=calib_path #give the path 
		calib=Calibration(calib)
		self.cats=["Pedestrian", "Cyclist", "Car"]
		
		#self.vis = cfg.vis
		self.model=model
		self.vis_dir = False
		self.vis = False
		self.dets=[]
		#self.oxts_packets=[]
		self.ego_com = True			# ego motion compensation
		self.affi_process =True	# post-processing affinity
		self.dataset="KITTI"
		self.det_name='pointrcnn'
		self.debug_id = None
		self.ID_start=1 
		torch.set_num_threads(4)
		self.poincloud_topic=pointcloud_topic
		self.pcdSub=message_filters.Subscriber("/kitti/velo/pointcloud", PointCloud2)
		self.imuSub=message_filters.Subscriber("/kitti/oxts/imu", Imu)
		self.gpsFixSub=message_filters.Subscriber("/kitti/oxts/gps/fix", NavSatFix)
		#self.gpsVelSub=message_filters.Subscriber("/kitti/oxts/gps/vel", TwistStamped)
		#self.imgSub=message_filters.Subscriber("/kitti/camera_color_left/image_raw", Image)
		ds=message_filters.ApproximateTimeSynchronizer(([self.pcdSub, self.imuSub, self.gpsFixSub ]),1, 0.1)
		ds.registerCallback(self.callbackFunction)
		self.marker_pub = rospy.Publisher( 'visualization_marker_array', MarkerArray)
		self.bbox_publish = rospy.Publisher("tracking_bboxes", BoundingBoxArray, queue_size=10)
		
		# counter
		self.trackers = []
		self.frame_count = 0
		self.ID_count = [ID_init]
		self.id_now_output = []

		#cfg file
		cfg=namedtuple('cfg', ['description','speed','save_root', 'dataset','split','det_name', 'cat_list',
			'score_threshold', 'num_hypo', 'ego_com','vis','affi_pro'])
		cfg.description='AB3DMOT'
		cfg.speed=1
		cfg.dataset='KITTI'
		cfg.split='val'
		cfg.det_name='pointrcnn'
		cfg.cat_list=['Car', 'Pedestrian', 'Cyclist']
		cfg.score_threshold=-100000 # can be changed
		cfg.num_hypo=1
		cfg.ego_com=True
		cfg.vis=False
		cfg.affi_pro=True
		#print("-cfg is set-")
		
		# config
		self.cat = "Car"
		self.ego_com = cfg.ego_com 		# ego motion compensation'
		self.affi_process = cfg.affi_pro	# post-processing affinity
		self.get_param(cfg, self.cat)

		rospy.spin()

	def get_param(self, cfg, cat):
		# get parameters for each dataset

		if cfg.dataset == 'KITTI':
			if cfg.det_name == 'pvrcnn':				# tuned for PV-RCNN detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
				else: assert False, 'error'
			elif cfg.det_name == 'pointrcnn':			# tuned for PointRCNN detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.6, 1, 4 		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
				else: assert False, 'error'
			elif cfg.det_name == 'deprecated':			
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 1, 3, 2		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
				else: assert False, 'error'
			else: assert False, 'error'
		
		else: assert False, 'no such dataset'

		# add negative due to it is the cost
		if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1	
		self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
			algm, metric, thres, max_age, min_hits

		# define max/min values for the output affinity matrix
		if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: self.max_sim, self.min_sim = 0.0, -100.
		elif self.metric in ['iou_2d', 'iou_3d']:   	   self.max_sim, self.min_sim = 1.0, 0.0
		elif self.metric in ['giou_2d', 'giou_3d']: 	   self.max_sim, self.min_sim = 1.0, -1.0

	
	def crop_pointcloud(self, pointcloud):

		mask = np.where((pointcloud[:, 0] >= lim_x[0]) & (pointcloud[:, 0] <= lim_x[1]) & (pointcloud[:, 1] >= lim_y[0]) & (pointcloud[:, 1] <= lim_y[1]) & (pointcloud[:, 2] >= lim_z[0]) & (pointcloud[:, 2] <= lim_z[1]))
		pointcloud = pointcloud[mask]
		return pointcloud

	def callbackFunction(self, lidarMsg, imuMsg, gpsFixMsg):
		start=rospy.Time.now().to_sec()
		start1=rospy.Time.now().to_sec()

		pc = ros_numpy.numpify(lidarMsg)
		points = np.column_stack((pc['x'], pc['y'],pc['z'], pc['i']))

		pc_arr=self.crop_pointcloud((points)) #reduce computational expense
		pointcloud_np = points_class(pc_arr, points_dim=pc_arr.shape[-1], attribute_dims=None)
		result, _  = inference_detector(self.model, pointcloud_np)

		#detections
		box = result[0]['boxes_3d'].tensor.numpy()
		scores = result[0]['scores_3d'].numpy()
		label = result[0]['labels_3d'].numpy()

		# box format : [ x, y, z, xdim(l), ydim(w), zdim(h), orientation] + label score
		# dets format : hwlxyzo + class
		dets=box[ :, [5,4,3,0,1,2,6]]
		info_data=[]
		dic_dets={}
		info_data = np.stack((label, scores), axis=1)
	
		dic_dets={'dets': dets, 'info': info_data}
		#print("-detections have been collected-")
		#print("number of detections from pointpillar", len(dets))
		
		#odom and gps data
		lat, lon, alt= gpsFixMsg.latitude, gpsFixMsg.longitude, gpsFixMsg.altitude
		roll, pitch, yaw= euler_from_quaternion(imuMsg.orientation.x, imuMsg.orientation.y, imuMsg.orientation.z, imuMsg.orientation.w)
		#roll, pitch, yaw= euler_from_quaternion(imuMsg.orientation.x, imuMsg.orientation.y, imuMsg.orientation.z, imuMsg.orientation.w)
		vn, ve=0,0
		vf, vl, vu= 0,0,0 #gpsVelMsg.twist.linear.x, gpsVelMsg.twist.linear.y,gpsVelMsg.twist.linear.z
		ax, ay, az=0,0,0 #imuMsg.linear_acceleration.x, imuMsg.linear_acceleration.y,imuMsg.linear_acceleration#0,0,0
		af, al, au=0,0,0 #imuMsg.linear_acceleration.x, imuMsg.linear_acceleration.y,imuMsg.linear_acceleration.z
		wx, wy, wz= 0,0,0 #gpsVelMsg.twist.angular.x, gpsVelMsg.twist.angular.y,gpsVelMsg.twist.angular.z
		wf, wl, wu=0,0,0 #imuMsg.angular_velocity.x, imuMsg.angular_velocity.y, imuMsg.angular_velocity.z
		wf, wl, wu= 0,0,0 #gpsVelMsg.twist.angular.x, gpsVelMsg.twist.angular.y,gpsVelMsg.twist.angular.z
		pos_accuracy, vel_accuracy =0,0 
		navstat, numsats= 0,0 #gpsFixMsg.status.status, 4
		posmode, velmode, orimode=  0,0,0
		OxtsPacket = namedtuple('OxtsPacket',
							'lat, lon, alt, ' +
							'roll, pitch, yaw, ' +
							'vn, ve, vf, vl, vu, ' +
							'ax, ay, az, af, al, au, ' +
							'wx, wy, wz, wf, wl, wu, ' +
							'pos_accuracy, vel_accuracy, ' +
							'navstat, numsats, ' +
							'posmode, velmode, orimode')
		self.oxts_packets = []
		oxtsdata=OxtsPacket(lat, lon, alt, roll, pitch, yaw, vn, ve, vf,vl,vu
			,ax,ay,az,af, al,au, wx,wy,wz,wf,wl,wu,pos_accuracy,vel_accuracy,navstat, numsats, posmode, velmode, orimode)
		self.oxts_packets.append(oxtsdata)
		self.oxts = _poses_from_oxts(self.oxts_packets) #imu and gps data		
		
		start=rospy.Time.now().to_sec()
		
		#for category classification
		#cat_detection= {k: dic_dets[k] for k in dic_dets.keys()}
		#idxcat=np.where(dic_dets['info'][:,0]==2)[0] #uncomment to only track cars
		#cat_detection= {k: dic_dets[k][idxcat] for k in dic_dets.keys()} #uncomment to only track cars

		cat_res, _=self.track(dic_dets)		
		#cat_res=np.c_[cat_res[0], 2* np.ones((cat_res[0].shape[0],1))]
		

		self.ID_start=max(self.ID_start, self.ID_count[0]) ##global counter
		trk_result=cat_res[0]
		#print("*******", trk_result)
		end=rospy.Time.now().to_sec()
		print("time for tracking",(end-start))

		#track detections - now we are considering Car class model for all classes - ToDo : add cyclist and pedestrian categories
		# h,w,l,x,y,z,theta, ID, other info, confidence

		bbox_array=BoundingBoxArray()
		cats=["Pedestrian", "Cyclist", "Car"]
		idx = 0
		self.markerArray = MarkerArray()
		#print(trk_result)
		for i, trk in enumerate(trk_result):
			if np.size(trk) == 0:
				continue
			
			q = yaw_to_quaternion(trk[6])
			bbox = BoundingBox()
			marker = Marker()

			marker.header = lidarMsg.header
			marker.type = marker.TEXT_VIEW_FACING
			marker.id = int(trk[7])
			marker.text = f"{int(trk[7])} {cats[int(trk[8])]}"
			marker.action = marker.ADD
			marker.frame_locked = True
			marker.lifetime = rospy.Duration(0.1)
			marker.scale.x, marker.scale.y,marker.scale.z = 0.8, 0.8, 0.8
			marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 1.0, 1.0, 1.0
			marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = trk[3], trk[4], trk[5] + 2
			
			bbox.header = lidarMsg.header #.seq = int(trk[7])
			#bbox.header.stamp = lidarMsg.header.stamp
			#bbox.header.frame_id = lidarMsg.header.frame_id
			bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z = trk[3], trk[4], trk[5]
			bbox.pose.orientation.w, bbox.pose.orientation.x, bbox.pose.orientation.y, bbox.pose.orientation.z = q[3], q[0], q[1], q[2]
			bbox.dimensions.x, bbox.dimensions.y, bbox.dimensions.z = trk[2], trk[1], trk[0]
			bbox.value = 0
			bbox.label = int(trk[8])
			bbox_array.header = bbox.header
			bbox_array.boxes.append(bbox)
			self.markerArray.markers.append(marker)
		
		#bbox_array.header.frame_id = lidarMsg.header.frame_id
		#print("len of bbox array from tracking", len(bbox_array.boxes))
		if len(bbox_array.boxes) != 0:
			self.bbox_publish.publish(bbox_array)
			self.marker_pub.publish(self.markerArray)
			bbox_array.boxes = []

		else:
			bbox_array.boxes = []
			self.bbox_publish.publish(bbox_array)
			#self.marker_pub.publish(self.markerArray)
		end1=rospy.Time.now().to_sec()
		print("time for publishing",(end1-start1))


	def process_dets(self, dets, info):
	
		dets_new = []
		for i, det in enumerate(dets):
			det_tmp = Box3D.array2bbox_raw(det, info[i,:])
			dets_new.append(det_tmp)
		#dets_new = [Box3D.array2bbox_raw(det,info) for det in dets]	
		return dets_new

	def within_range(self, theta):

		if theta >= np.pi: theta -= np.pi * 2    
		if theta < -np.pi: theta += np.pi * 2

		return theta

	def orientation_correction(self, theta_pre, theta_obs):

		theta_pre = self.within_range(theta_pre)
		theta_obs = self.within_range(theta_obs)

		if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:     
			theta_pre += np.pi       
			theta_pre = self.within_range(theta_pre)

		if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
			if theta_obs > 0: theta_pre += np.pi * 2
			else: theta_pre -= np.pi * 2

		return theta_pre, theta_obs

	def ego_motion_compensation(self, frame, trks):
		
		assert len(self.trackers) == len(trks), 'error'
		if get_ego_traj(self.oxts, frame, 1, 1, only_fut=True, inverse=True) !=None:
			print("ego trajectory is compensated - step 2")
			ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(self.oxts, frame, 1, 1, only_fut=True, inverse=True) 
			for index in range(len(self.trackers)):
				trk_tmp = trks[index]
				xyz = np.array([trk_tmp.x, trk_tmp.y, trk_tmp.z]).reshape((1, -1))
				compensated = egomotion_compensation_ID(xyz, self.calib, ego_rot_imu, ego_xyz_imu, left, right)
				trk_tmp.x, trk_tmp.y, trk_tmp.z = compensated[0]

		return trks

	
	def prediction(self):

		trks = []
		for t in range(len(self.trackers)):
			
			# propagate locations
			kf_tmp = self.trackers[t]
			kf_tmp.kf.predict()
			kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])
			# update statistics
			kf_tmp.time_since_update += 1 		
			trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
			trks.append(Box3D.array2bbox(trk_tmp))

		return trks

	def update(self, matched, unmatched_trks, dets, info):
		# update matched trackers with assigned detections
		
		dets = copy.copy(dets)
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
				assert len(d) == 1, 'error'

				# update statistics
				trk.time_since_update = 0		# reset because just updated
				trk.hits += 1

				# update orientation in propagated tracks and detected boxes so that they are within 90 degree
				bbox3d = Box3D.bbox2array(dets[d[0]])
				trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])
				# kalman filter update with observation
				trk.kf.update(bbox3d)
				trk.kf.x[3] = self.within_range(trk.kf.x[3])
				trk.info = info[d, :][0]


	def birth(self, dets, info, unmatched_dets):

		new_id_list = list()					# new ID generated for unmatched detections
		for i in unmatched_dets:        			# a scalar of index
			trk = KF(Box3D.bbox2array(dets[i]), info[i, :], self.ID_count[0])
			self.trackers.append(trk)
			new_id_list.append(trk.id)
			self.ID_count[0] += 1

		return new_id_list

	def output(self):

		num_trks = len(self.trackers)
		results = []
		for trk in reversed(self.trackers):
			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
			d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
			d = Box3D.bbox2array_raw(d)

			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
				results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1)) 		
			num_trks -= 1

			# deadth, remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(num_trks)

		return results

	def process_affi(self, affi, matched, unmatched_dets, new_id_list):

		trk_id = self.id_past 			# ID in the trks for matching

		det_id = [-1 for _ in range(affi.shape[0])]		# initialization
		for match_tmp in matched:		
			det_id[match_tmp[0]] = trk_id[match_tmp[1]]

		count = 0
		assert len(unmatched_dets) == len(new_id_list), 'error'
		for unmatch_tmp in unmatched_dets:
			det_id[unmatch_tmp] = new_id_list[count] 	# new_id_list is in the same order as unmatched_dets
			count += 1
		assert not (-1 in det_id), 'error, still have invalid ID in the detection list'
		affi = affi.transpose() 			

		permute_row = list()
		'''for output_id_tmp in self.id_past_output:
			index = trk_id.index(output_id_tmp)
			permute_row.append(index)'''
		permute_row = [trk_id.index(output_id_tmp) for output_id_tmp in self.id_past_output]
		affi = affi[permute_row, :]	
		assert affi.shape[0] == len(self.id_past_output), 'error'

		max_index = affi.shape[1]
		permute_col = list()
		to_fill_col, to_fill_id = list(), list() 		# append new columns at the end, also remember the ID for the added ones
		for output_id_tmp in self.id_now_output:
			try:
				index = det_id.index(output_id_tmp)
			except:		# some output ID does not exist in the detections but rather predicted by KF
				index = max_index
				max_index += 1
				to_fill_col.append(index); to_fill_id.append(output_id_tmp)
			permute_col.append(index)

		# expand the affinity matrix with newly added columns
		append = np.zeros((affi.shape[0], max_index - affi.shape[1]))
		append.fill(self.min_sim)
		affi = np.concatenate([affi, append], axis=1)

		# find out the correct permutation for the newly added columns of ID
		for count in range(len(to_fill_col)):
			fill_col = to_fill_col[count]
			fill_id = to_fill_id[count]
			row_index = self.id_past_output.index(fill_id)

			# construct one hot vector because it is proapgated from previous tracks, so 100% matching
			affi[row_index, fill_col] = self.max_sim		
		affi = affi[:, permute_col]

		return affi

	def track(self, dets_all):

		dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
		#if self.debug_id: print('\nframe is %s' % frame)
		self.frame_count += 1

		# recall the last frames of outputs for computing ID correspondences during affinity processing
		self.id_past_output = copy.copy(self.id_now_output)
		self.id_past = [trk.id for trk in self.trackers]

		# process detection format
		dets = self.process_dets(dets, info)
		#print(dets[0].cls,"dets with classes")

		# tracks propagation based on velocity
		trks = self.prediction()
		#print("trks after prediction: ", trks)
		# ego motion compensation, adapt to the current frame of camera coordinate
		'''if (self.frame_count > 0) and (self.ego_com) and (self.oxts is not None):
			#print("step 1 ::: ")
			trks = self.ego_motion_compensation(self.frame_count, trks)'''

		# matching
		trk_innovation_matrix = None
		if self.metric == 'm_dis':
			trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 
		matched, unmatched_dets, unmatched_trks, cost, affi = \
			data_association(dets, trks, self.metric, self.thres, self.algm, trk_innovation_matrix)
		
		'''print("matched : ", matched)
		print("unmatched_dets : ", unmatched_dets)
		print("unmatched_trks : ", unmatched_trks)'''
		
		self.update(matched, unmatched_trks, dets, info)
		new_id_list = self.birth(dets, info, unmatched_dets)

		results = self.output()
		if len(results) > 0: results = [np.concatenate(results)]		# h,w,l,x,y,z,theta, ID, other info, confidence
		else:            	 results = [np.empty((0, 10))]
		self.id_now_output = results[0][:, 7].tolist()				# only the active tracks that are outputed
		if self.affi_process:
			affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)

		return results, affi

def euler_from_quaternion(x, y, z, w):

	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll_x = math.atan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch_y = math.asin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)
	return roll_x, pitch_y, yaw_z # in radians 


def yaw_to_quaternion(yaw):
    r = R.from_euler('z', yaw, degrees=False)
    return r.as_quat()


if __name__ == '__main__':
	rospy.init_node("trackingNode")
	print("tracking node initialzied")
	config_file = config_path
	checkpoint_file =model_path
	device= torch.device('cuda:0')
	topic="/kitti/velo/pointcloud"
	model = init_model(config_file, checkpoint_file, device)
	ID_start=1
	AB3DMOT( ID_init=ID_start,model=model, pointcloud_topic=topic)
