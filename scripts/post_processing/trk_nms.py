# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, sys, argparse
import pandas as pd
import numpy as np
# from AB3DMOT_libs.utils import get_threshold
from AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.kitti_trk import Tracklet_3D
from AB3DMOT_libs.kitti_obj import read_label
from AB3DMOT_libs.dist_metrics import iou, iom
from xinshuo_io import load_txt_file, load_list_from_folder, mkdir_if_missing, fileparts

def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--result_sha', type=str, default='pointrcnn_Car_test_thres', help='name of the result folder')
    parser.add_argument('--split', type=str, default='val', help='train, val, test, mini_train, mini_val, custom_val')
    parser.add_argument('--title', type=str, default='defaultTH', help='results folder sufix')
    parser.add_argument('--dataset', type=str, default='nuScenes', help='KITTI, nuScenes')
    parser.add_argument('--num_hypo', type=int, default=1, help='number of hypothesis used')
    parser.add_argument('--per_frame', action='store_true', help='delete object only in frames where it is coupled')
    args = parser.parse_args()
    return args

def obj_to_Box3D(obj):
	bbox = Box3D()
	bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s = obj.h, obj.w, obj.l, obj.x, obj.y, obj.z, obj.ry, obj.s
	return bbox
    
def run_nms(data_dir, save_dir, num_hypo, del_per_frame):
	
	# loop through all hypotheses
	for hypo_index in range(num_hypo):
		
		############### collect all trajectories and their scores/categories at every frame
		trk_id_scores = dict()
		eval_dir = os.path.join(data_dir, 'data_%d' % (hypo_index))
		seq_list, num_seq = load_list_from_folder(eval_dir)
		delete_ids_per_frame = [[] for _ in range(41)]
		to_delete_id = list()
		for seq_file in seq_list:

			# loading tracklets in this sequence
			results = Tracklet_3D(seq_file)
			seq_data = results.data
			seq_coupling_df = pd.DataFrame(columns=['A', 'B', 'frames', 'scores_A', 'scores_B', 'cat_A', 'cat_B'])

			# loop through every tracklet
			for frame, frame_data in seq_data.items():
				frame_df = pd.DataFrame(columns=['id', 'xz', 'obj'])
				for id_tmp, obj in frame_data.items():

					cur_line_df = pd.DataFrame([[id_tmp, [obj.x, obj.z], obj]], columns=['id', 'xz', 'obj'])
					frame_df = pd.concat([frame_df, cur_line_df], ignore_index=True)

					# read object score
					if id_tmp not in trk_id_scores.keys(): trk_id_scores[id_tmp] = list()
					trk_id_scores[id_tmp].append(obj.s)

					# # read object category
					# if id_tmp not in cat_dict.keys(): cat_dict[id_tmp] = obj.type

				all_objects_xz = np.array(frame_df['xz'].tolist())
				for b, cur_obj_xz in enumerate(all_objects_xz):
					xz_diffs = all_objects_xz[:b] - cur_obj_xz
					couple_cand = np.where(np.all(np.abs(xz_diffs) < 3.0, axis=1))	# !!!!!!!! TODO: set as parameter !!!!!!!!
					for a in couple_cand[0]:
						obj_a = frame_df.loc[a]['obj']
						obj_b = frame_df.loc[b]['obj']
						if obj_a.id > obj_b.id:
							obj_a = frame_df.loc[b]['obj']
							obj_b = frame_df.loc[a]['obj']

						# iou_3d = iou(obj_to_Box3D(obj_a), obj_to_Box3D(obj_b), metric='iou_3d')
						# iou_2d = iou(obj_to_Box3D(obj_a), obj_to_Box3D(obj_b), metric='iou_2d')
						iom_2d = iom(obj_to_Box3D(obj_a), obj_to_Box3D(obj_b), metric='iom_2d')
						if iom_2d > 0.25:	# !!!!!!!! TODO: set as parameter !!!!!!!!
							categories = [obj_a.type, obj_b.type]
							if 'Pedestrian' in categories and iom_2d < 0.5:	# !!!!!!!! TODO: set as parameter !!!!!!!!
								continue
							idx = seq_coupling_df.index[(seq_coupling_df['A'] == obj_a.id) & (seq_coupling_df['B'] == obj_b.id)]
							if idx.empty:
								new_couple_df = pd.DataFrame([[obj_a.id, obj_b.id, [frame], [obj_a.s], [obj_b.s], obj_a.type, obj_b.type]], 
						columns=['A', 'B', 'frames', 'scores_A', 'scores_B', 'cat_A', 'cat_B'])
								seq_coupling_df = pd.concat([seq_coupling_df, new_couple_df], ignore_index=True)
							else:
								# cur_row = seq_coupling_df.loc[idx]
								seq_coupling_df.loc[idx]['frames'].values[0].append(frame)
								seq_coupling_df.loc[idx]['scores_A'].values[0].append(obj_a.s)
								seq_coupling_df.loc[idx]['scores_B'].values[0].append(obj_b.s)
								# cur_row['frames'] = np.append(cur_row['frames'], frame)
								# seq_coupling_df.loc[idx] = cur_row

				f = 2
			for index, coupling_row in seq_coupling_df.iterrows():
				# categories = [coupling_row['cat_A'], coupling_row['cat_B']]
				# if ('Pedestrian' in categories) and ('Bicycle' in categories):
				# 	continue
				# max_a = max(coupling_row['scores_A'])
				# max_b = max(coupling_row['scores_B'])
				# # if max([max_a, max_b]) < 0.1:	# !!!!!!!! TODO: set as parameter !!!!!!!!
				# # 	if ('Car' in categories) and ('Truck' in categories):
				# # 		delete_id = coupling_row[categories.index('Car')]
				# # 	elif ('Car' in categories) and ('Bus' in categories):
				# # 		delete_id = coupling_row[categories.index('Bus')]
				# # 	elif ('Truck' in categories) and ('Trailer' in categories):
				# # 		delete_id = coupling_row[categories.index('Truck')]
				# # 	else:
				# # 		delete_id = coupling_row['B'] if max_a >= max_b else coupling_row['A']
				# # else:
				# delete_id = coupling_row['B'] if max_a >= max_b else coupling_row['A']

				scores_a = trk_id_scores[coupling_row['A']]
				scores_b = trk_id_scores[coupling_row['B']]
				max_a = max(scores_a)
				max_b = max(scores_b)
				if (max(max_a, max_b) - min(max_a, max_b)) > 0.1:	# !!!!!!!! TODO: set as parameter !!!!!!!!
					delete_id = coupling_row['B'] if max_a >= max_b else coupling_row['A']
				else:
					mean_a = np.mean(scores_a)
					mean_b = np.mean(scores_b)
					if mean_a == mean_b:
						delete_id = coupling_row['B'] if len(scores_a) >= len(scores_b) else coupling_row['A']
					else:
						delete_id = coupling_row['B'] if mean_a > mean_b else coupling_row['A']

				if del_per_frame:
					for f in coupling_row['frames']:
						delete_ids_per_frame[f].append(delete_id)
				else:
					to_delete_id.append(delete_id)

			ff = 2

					
		# ############## collect the ID to remove based on the category-specific confidence score
		# to_delete_id = list()
		# for track_id, score_list in trk_id_score.items():
		# 	average_score = sum(score_list) / float(len(score_list))
		# 	obj_type = cat_dict[track_id]
		# 	if average_score < thres_dict[obj_type]:
		# 		to_delete_id.append(track_id)

		# ############# remove the ID in the data_0 folder
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
					if del_per_frame:
						if id_tmp not in delete_ids_per_frame[frame]:
							seq_file_save.write(obj.convert_to_trk_output_str(frame) + '\n')
					elif id_tmp not in to_delete_id:
						seq_file_save.write(obj.convert_to_trk_output_str(frame) + '\n')
			seq_file_save.close()

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
					if del_per_frame:
						if obj.id not in delete_ids_per_frame[int(frame_index)]:
							frame_file_save.write(obj.convert_to_det_str() + '\n')
					elif obj.id not in to_delete_id:
						frame_file_save.write(obj.convert_to_det_str() + '\n')

				frame_file_save.close()

if __name__ == '__main__':
	
	# get config
	args = parse_args()
	result_sha = args.result_sha
	num_hypo = args.num_hypo
	det_name = result_sha.split('_')[0]
	subfolder_all = f'{det_name}_{args.split}_H{num_hypo}'
	del_per_frame = args.per_frame

	# # get threshold for filtering
	# thres_dict = get_threshold(args.dataset, det_name)

	# get directories
	file_path = os.path.dirname(os.path.realpath(__file__))
	root_dir = os.path.join(file_path, '../../results', args.dataset)
	data_dir = os.path.join(root_dir, result_sha, subfolder_all)
	save_dir = os.path.join(root_dir, f'{result_sha}_{args.title}', subfolder_all)
	mkdir_if_missing(save_dir)

	run_nms(data_dir, save_dir, num_hypo, del_per_frame)