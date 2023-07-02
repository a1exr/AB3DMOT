import numpy as np
import copy
from scripts.nuScenes.nusc_tracking.track_utils import greedy_assignment, iou_matching
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

NUSCENES_STATIC_ATTRIBUTES = [
    'without_rider',
    'parked',
    # 'stopped',
    'sitting_lying_down',
    # 'standing'
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
  def __init__(self,  hungarian=False, max_age=0, th=0.0, velocity_error=NUSCENE_CLS_VELOCITY_ERROR):
    self.hungarian = hungarian
    self.max_age = max_age
    self.th = th

    print("Use hungarian: {}".format(hungarian))

    self.NUSCENE_CLS_VELOCITY_ERROR = velocity_error

    self.reset()
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, detections, time_lag, matching_coefs):
    if len(detections) == 0:
      self.tracks = []
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
      det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
      det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
      temp.append(det)

    detections = temp
    if len(detections) == 0:
      self.tracks = []
      return []

    N = len(detections)
    M = len(self.tracks)

    # N X 2 
    if 'tracking' in detections[0]:
      dets = np.array(
      [ det['ct'] + det['tracking'].astype(np.float32)
       for det in detections], np.float32)
    else:
      dets = np.array(
        [det['ct'] for det in detections], np.float32) 

    item_cat = np.array([item['label_preds'] for item in detections], np.int32) # N
    track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M

    max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in detections], np.float32)

    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    if len(tracks) > 0:  # NOT FIRST FRAME
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
      dist = np.sqrt(dist) # absolute distance in meter

      invalid = ((dist > max_diff.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

      dist = dist  + invalid * 1e18
      # if self.hungarian:
      #   dist[dist > 1e18] = 1e18
      #   matched_indices = linear_assignment(copy.deepcopy(dist))
      # else:
      # matched_indices = greedy_assignment(copy.deepcopy(dist))
      matched_indices = iou_matching(copy.deepcopy(dist), detections, self.tracks, matching_coefs)
      
    else:  # first few frame
      # assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2)

    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]

    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
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

    ret = []
    for m in matches:
      track = detections[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']      
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)

    for i in unmatched_dets:
      track = detections[i]
      self.id_count += 1
      track['tracking_id'] = self.id_count
      track['age'] = 1
      track['active'] =  1
      ret.append(track)

    # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
    # the object in current frame 
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.max_age:
        # track['active'] = 0
        # ct = track['ct']
        is_static_track = len([attr for attr in NUSCENES_STATIC_ATTRIBUTES if attr in track['attribute_name']]) > 0
        track['active'] = 1 if (is_static_track and track['age']==1 and track['detection_score']>0.3) else 0
          
        # movement in the last second
        if 'tracking' in track:
            offset = track['tracking'] * -1 # move forward 
            track['ct'] += offset

        track['age'] += 1
        ret.append(track)

    self.tracks = ret
    return ret