from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from scripts.nuScenes.nusc_tracking.CP_tracker_backward import PubTracker as Tracker
from nuscenes import NuScenes
import json 
import time
from AB3DMOT_libs.nuScenes_split import get_split

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument("--forward", help="forward track_results_nusc.json file")
    parser.add_argument("--backward", help="backward track_results_nusc.json file")
    # parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='val')
    # parser.add_argument("--max_age", type=int, default=3)
    # parser.add_argument("--TH", type=float, default=0.0)      # TODO: optimize this value

    args = parser.parse_args()

    return args


def main(args):
    forward_res = os.path.join(args.work_dir, args.forward)
    backward_res = os.path.join(args.work_dir, args.backward)

    with open(os.path.join(forward_res, f'track_results_nusc.json'), 'rb') as f:
        forward_tracks=json.load(f)['results']
    with open(os.path.join(forward_res, 'frames_meta.json'), 'rb') as f:
        forward_frames=json.load(f)['frames']
    with open(os.path.join(backward_res, f'track_results_nusc.json'), 'rb') as f:
        backward_tracks=json.load(f)['results']
    with open(os.path.join(backward_res, 'frames_meta.json'), 'rb') as f:
        backward_frames=json.load(f)['frames']

    res_dir = os.path.join(args.work_dir, 'forward_backward_' + args.forward)
    os.makedirs(res_dir, exist_ok=True)


    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(forward_frames)

    print("Begin merging\n")
    start = time.time()
    for i in range(size):
        token = forward_frames[i]['token']

        f_trks = forward_tracks[token]
        b_trks = backward_tracks[token]

        # outputs = tracker.step_centertrack(f_trks, time_lag)
        annos = []

        for forward_trk in f_trks:
            forward_trk_loc = np.array(forward_trk['translation'])
            for backward_trk in b_trks:
                backward_trk_loc = np.array(backward_trk['translation'])
                dist = np.sum(forward_trk_loc - backward_trk_loc)
                if (dist < 0.1) and (forward_trk['tracking_score'] == backward_trk['tracking_score']):
                    b_trks.remove(backward_trk)
                    break 
            nusc_anno = {
                "sample_token": token,
                "translation": forward_trk['translation'],
                "size": forward_trk['size'],
                "rotation": forward_trk['rotation'],
                "velocity": forward_trk['velocity'],
                "tracking_id": str(forward_trk['tracking_id']),
                "tracking_name": forward_trk['tracking_name'],
                "tracking_score": forward_trk['tracking_score'],
            }
            annos.append(nusc_anno)
        
        for backward_trk in b_trks:
            new_id = int(backward_trk['tracking_id']) + 10000
            nusc_anno = {
                "sample_token": token,
                "translation": backward_trk['translation'],
                "size": backward_trk['size'],
                "rotation": backward_trk['rotation'],
                "velocity": backward_trk['velocity'],
                "tracking_id": str(new_id),
                "tracking_name": backward_trk['tracking_name'],
                "tracking_score": backward_trk['tracking_score'],
            }
            annos.append(nusc_anno)

        nusc_annos["results"].update({token: annos})
    
    end = time.time()

    second = (end-start) 

    speed=size / second
    print("The speed is {} FPS".format(speed))

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    with open(os.path.join(res_dir, f'track_results_nusc.json'), "w") as f:
        json.dump(nusc_annos, f)
    return speed

def eval_tracking(args):
    # args = parse_args()
    centerpoint_res_dir = os.path.join(args.work_dir, 'forward_backward_' + args.forward)
    eval_track_folder = os.path.join(centerpoint_res_dir, 'eval_track')
    os.makedirs(eval_track_folder, exist_ok=True)

    eval(os.path.join(centerpoint_res_dir, f'track_results_nusc.json'),
        args.version,
        eval_track_folder,
        args.root
    )

def eval(res_path, eval_set="val", output_dir=None, root_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval 
    from nuscenes.eval.common.config import config_factory as track_configs

    nusc_version = "v1.0-trainval"
    if eval_set=='mini_val': nusc_version = "v1.0-mini"
    elif eval_set=='test': nusc_version = "v1.0-test"

    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version=nusc_version,
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()


def test_time():
    speeds = []
    for i in range(3):
        speeds.append(main())

    print("Speed is {} FPS".format( max(speeds)  ))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    # test_time()
    eval_tracking(args)