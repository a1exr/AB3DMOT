import numpy as np
import copy
from scripts.nuScenes.nusc_tracking.track_utils import greedy_assignment
import copy 
import importlib
import sys 

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
  'car':4,
  'truck':4,
  'bus':5.5,
  'trailer':3,
  'pedestrian':1,
  'motorcycle':13,
  'bicycle':3,  
}



class PubTracker(object):
  def __init__(self,  hungarian=False, max_age=0, th=0.0):
    self.hungarian = hungarian
    self.max_age = max_age
    self.th = th

    print("Use hungarian: {}".format(hungarian))

    self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR

    self.reset()
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, detections, time_lag):
    # if len(detections) == 0:
    #   self.tracks = []
    #   return []
    # else:
    temp = []
    for det in detections:
      # filter out classes not evaluated for tracking 
      if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
        continue 

	    # Detection thresholding
      if det['detection_score'] < self.th:
        continue 

      det['ct'] = np.array(det['translation'][:2])
      # det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
      det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
      temp.append(det)
    detections = temp

    temp = []
    for trk in self.tracks:
      trk['tracking'] = np.array(trk['velocity'][:2]) * -1 * time_lag
      trk['ct'] += trk['tracking'].astype(np.float32)
      temp.append(trk)
    self.tracks = temp
    # if len(detections) == 0:
    #   self.tracks = []
    #   return []

    N = len(detections)
    M = len(self.tracks)
    item_cat = np.array([item['label_preds'] for item in detections], np.int32) # N
    track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M
    max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in detections], np.float32)

    # # N X 2 
    # if 'tracking' in detections[0]:
    #   dets = np.array(
    #   [ det['ct'] + det['tracking'].astype(np.float32)
    #    for det in detections], np.float32)
    # else:
    dets = np.array([det['ct'] for det in detections], np.float32)  # N X 2
    tracks = np.array([pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    if len(tracks) > 0:  # i.e. NOT LAST FRAME
      dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
      dist = np.sqrt(dist) # absolute distance in meter

      invalid = ((dist > max_diff.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

      dist = dist  + invalid * 1e18
      # if self.hungarian:
      #   dist[dist > 1e18] = 1e18
      #   matched_indices = linear_assignment(copy.deepcopy(dist))
      # else:
      matched_indices = greedy_assignment(copy.deepcopy(dist))
    else:  # LAST FRAME
      # assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2)

    unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
    unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]
    
    if self.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    output = []
    for m in matches:
      tracked_det = detections[m[0]]
      tracked_det['tracking_id'] = self.tracks[m[1]]['tracking_id']      
      tracked_det['age'] = 1
      tracked_det['active'] = self.tracks[m[1]]['active'] + 1
      output.append(tracked_det)

    for i in unmatched_dets:
      new_det = detections[i]
      self.id_count += 1
      new_det['tracking_id'] = self.id_count
      new_det['age'] = 1
      new_det['active'] =  1
      output.append(new_det)

    # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
    # the object in current frame 
    for i in unmatched_tracks:
      missing_trk = self.tracks[i]
      if missing_trk['age'] < self.max_age:
        missing_trk['age'] += 1
        missing_trk['active'] = 0
        ct = missing_trk['ct']
        # # movement in the last second
        # if 'tracking' in missing_trk:
        #     offset = missing_trk['tracking'] * -1 # move forward 
        #     missing_trk['ct'] = ct + offset 
        output.append(missing_trk)

    self.tracks = output
    return output


    # # if len(detections) == 0:
    # #   self.tracks = []
    # #   return []
    # # else:
    # temp = []
    # for det in detections:
    #   # filter out classes not evaluated for tracking 
    #   if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
    #     continue 

	  #   # Detection thresholding
    #   if det['detection_score'] < self.th:
    #     continue 

    #   det['ct'] = np.array(det['translation'][:2])
    #   det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
    #   det['categories'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
    #   temp.append(det)

    # detections = temp
    # if len(detections) == 0:
    #   self.tracks = []
    #   return []

    # N = len(detections)
    # M = len(self.tracks)
    # item_cat = np.array([item['categories'] for item in detections], np.int32) # N
    # track_cat = np.array([track['categories'] for track in self.tracks], np.int32) # M
    # max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in detections], np.float32)

    # # N X 2 
    # # if 'tracking' in detections[0]:
    # # preds = np.array([det['ct'] + det['tracking'].astype(np.float32) for det in detections], np.float32)
    # # else:
    # dets = np.array([det['ct'] for det in detections], np.float32) 
    # tracks = np.array([pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    # if len(tracks) > 0:  # i.e. NOT LAST FRAME
    #   dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
    #   dist = np.sqrt(dist) # absolute distance in meter

    #   invalid = ((dist > max_diff.reshape(N, 1)) + \
    #   (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

    #   dist = dist  + invalid * 1e18
    #   # if self.hungarian:
    #   #   dist[dist > 1e18] = 1e18
    #   #   matched_indices = linear_assignment(copy.deepcopy(dist))
    #   # else:
    #   matched_indices = greedy_assignment(copy.deepcopy(dist))
    # else:  # LAST FRAME
    #   matched_indices = np.array([], np.int32).reshape(-1, 2)

    # unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
    # unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]
    
    # if self.hungarian:
    #   matches = []
    #   for m in matched_indices:
    #     if dist[m[0], m[1]] > 1e16:
    #       unmatched_dets.append(m[0])
    #     else:
    #       matches.append(m)
    #   matches = np.array(matches).reshape(-1, 2)
    # else:
    #   matches = matched_indices

    # output = []
    # tracks = []
    # for m in matches:
    #   tracked_det = detections[m[0]]
    #   tracked_det['tracking_id'] = self.tracks[m[1]]['tracking_id']      
    #   tracked_det['age'] = 1
    #   tracked_det['active'] = self.tracks[m[1]]['active'] + 1
    #   pred = copy.deepcopy(tracked_det)
    #   pred['ct'] += pred['tracking']
    #   output.append(tracked_det)
    #   tracks.append(pred)

    # for i in unmatched_dets:
    #   new_det = detections[i]
    #   self.id_count += 1
    #   new_det['tracking_id'] = self.id_count
    #   new_det['age'] = 1
    #   new_det['active'] =  1
    #   pred = copy.deepcopy(new_det)
    #   pred['ct'] += pred['tracking']
    #   output.append(new_det)
    #   tracks.append(pred)

    # # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
    # # the object in current frame 
    # for i in unmatched_tracks:
    #   missing_trk = self.tracks[i]
    #   if missing_trk['age'] < self.max_age:
    #     missing_trk['age'] += 1
    #     missing_trk['active'] = 0
    #     pred = copy.deepcopy(missing_trk)
    #     pred['ct'] += np.array(pred['velocity'][:2]) * -1 * time_lag
    #     output.append(missing_trk)
    #     tracks.append(pred)
    #     # # movement in the last second
    #     # if 'tracking' in missing_trk:
    #     #     # ct = missing_trk['ct']
    #     #     # offset = missing_trk['tracking'] * -1 # move forward 
    #     #     missing_trk['ct'] += missing_trk['tracking'] * -1 # move forward 
    #     # output.append(missing_trk)

    # self.tracks = output
    # return output