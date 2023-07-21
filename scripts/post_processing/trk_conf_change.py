# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, sys, argparse
import numpy as np
from AB3DMOT_libs.utils import get_threshold
from AB3DMOT_libs.kitti_trk import Tracklet_3D
from AB3DMOT_libs.kitti_obj import read_label
from xinshuo_io import load_txt_file, load_list_from_folder, mkdir_if_missing, fileparts

max_dists = {
	'Bicycle': 40,
	'Motorcycle': 40,
	'Pedestrian':40,
	'Bus': 50,
	'Car': 50,
	'Trailer': 50,
	'Truck': 50
}

def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--result_sha', type=str, default='pointrcnn_Car_test_thres', help='name of the result folder')
    parser.add_argument('--split', type=str, default='val', help='train, val, test, mini_train, mini_val, custom_val')
    parser.add_argument('--title', type=str, default='defaultTH', help='results folder sufix')
    parser.add_argument('--dataset', type=str, default='nuScenes', help='KITTI, nuScenes')
    parser.add_argument('--num_hypo', type=int, default=1, help='number of hypothesis used')
    args = parser.parse_args()
    return args

def conf_thresholding(data_dir, save_dir, thres_dict, num_hypo):
	
	# loop through all hypotheses
	for hypo_index in range(num_hypo):
		
		############### collect all trajectories and their scores/categories at every frame
		# trk_id_score, cat_dict = dict(), dict()
		eval_dir = os.path.join(data_dir, 'data_%d' % (hypo_index))
		seq_list, num_seq = load_list_from_folder(eval_dir)
		# delete_ids_per_frame = [[] for _ in range(41)]
		save_dir_tmp = os.path.join(save_dir, 'data_%d' % (hypo_index)); mkdir_if_missing(save_dir_tmp)
		for seq_file in seq_list:

			# save file
			seq_name = fileparts(seq_file)[1]
			seq_file_save = os.path.join(save_dir_tmp, seq_name+'.txt'); seq_file_save = open(seq_file_save, 'w')

			# loading tracklets in this sequence
			results = Tracklet_3D(seq_file)
			seq_data = results.data

			# loop through every tracklet
			for frame, frame_data in seq_data.items():
				for id_tmp, obj in frame_data.items():

					max_dist = max_dists[obj.type]
					obj_dist = np.linalg.norm([obj.x, obj.z])
					if obj_dist > 1.05*max_dist:
						# delete_ids_per_frame[frame].append(id_tmp)
						continue

					new_score_coef = 1.4 - (1 / max_dist) * obj_dist
					obj.s = min(1.0, new_score_coef*obj.s)

					seq_file_save.write(obj.convert_to_trk_output_str(frame) + '\n')
			seq_file_save.close()

					# # read object score
					# if id_tmp not in trk_id_score.keys(): trk_id_score[id_tmp] = list()
					# trk_id_score[id_tmp].append(obj.s)

					# # read object category
					# if id_tmp not in cat_dict.keys(): cat_dict[id_tmp] = obj.type
					
		############## collect the ID to remove based on the category-specific confidence score
		# to_delete_id = list()
		# for track_id, score_list in trk_id_score.items():
		# 	average_score = sum(score_list) / float(len(score_list))
		# 	obj_type = cat_dict[track_id]
		# 	if average_score < thres_dict[obj_type]:
		# 		to_delete_id.append(track_id)

		############# remove the ID in the data_0 folder
		# save_dir_tmp = os.path.join(save_dir, 'data_%d' % (hypo_index)); mkdir_if_missing(save_dir_tmp)
		# for seq_file in seq_list:

		# 	# save file
		# 	seq_name = fileparts(seq_file)[1]
		# 	seq_file_save = os.path.join(save_dir_tmp, seq_name+'.txt'); seq_file_save = open(seq_file_save, 'w')

		# 	# loading tracklets in this sequence
		# 	results = Tracklet_3D(seq_file)
		# 	seq_data = results.data

		# 	# loop through every tracklet
		# 	for frame, frame_data in seq_data.items():
		# 		for id_tmp, obj in frame_data.items():
		# 			if id_tmp not in delete_ids_per_frame[frame]:
		# 					seq_file_save.write(obj.convert_to_trk_output_str(frame) + '\n')
		# 	seq_file_save.close()

		############# remove the ID in the trk_with_id folder for detection evaluation and tracking visualization
		trk_id_dir = os.path.join(data_dir, 'trk_withid_%d' % (hypo_index))
		seq_dir_list, num_seq = load_list_from_folder(trk_id_dir)
		save_dir_tmp = os.path.join(save_dir, 'trk_withid_%d' % (hypo_index))
		
		# load every sequence
		for seq_dir in seq_dir_list:
			frame_list, num_frame = load_list_from_folder(seq_dir)
			seq_name = fileparts(seq_dir)[1]
			save_frame_dir = os.path.join(save_dir_tmp, seq_name); mkdir_if_missing(save_frame_dir)
			
			# load every frame
			for frame in frame_list:
				frame_index = fileparts(frame)[1]
				frame_file_save = os.path.join(save_frame_dir, frame_index+'.txt'); frame_file_save = open(frame_file_save, 'w')	
				
				# load and save results for every object if not falling into the to delete list based on ID
				results = read_label(frame) 		
				for obj in results:
					max_dist = max_dists[obj.type]
					obj_dist = np.linalg.norm([obj.x, obj.z])
					if obj_dist > 1.05*max_dist:
						# delete_ids_per_frame[frame].append(id_tmp)
						continue

					new_score_coef = 1.4 - (1 / max_dist) * obj_dist
					obj.s = min(1.0, new_score_coef*obj.s)

					frame_file_save.write(obj.convert_to_det_str() + '\n')
				frame_file_save.close()

if __name__ == '__main__':
	
	# get config
	args = parse_args()
	result_sha = args.result_sha
	num_hypo = args.num_hypo
	det_name = result_sha.split('_')[0]
	subfolder_all = f'{det_name}_{args.split}_H{num_hypo}'

	# get threshold for filtering
	thres_dict = get_threshold(args.dataset, det_name)

	# get directories
	file_path = os.path.dirname(os.path.realpath(__file__))
	root_dir = os.path.join(file_path, '../../results', args.dataset)
	data_dir = os.path.join(root_dir, result_sha, subfolder_all)
	save_dir = os.path.join(root_dir, f'{result_sha}_{args.title}', subfolder_all)
	mkdir_if_missing(save_dir)

	# run thresholding
	conf_thresholding(data_dir, save_dir, thres_dict, num_hypo)