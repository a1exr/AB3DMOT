from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import pandas as pd
import time
import optuna
import argparse
import copy
from scripts.nuScenes.nusc_tracking.pub_tracker import PubTracker as Tracker
from nuscenes import NuScenes
from AB3DMOT_libs.nuScenes_split import get_split

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from")
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='val')
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--TH", type=float, default=0.0)      # TODO: optimize this value

    args = parser.parse_args()

    return args


def save_first_frame(args):
    # args = parse_args()
    nusc_version = "v1.0-trainval"
    if args.version=='mini_val': nusc_version = "v1.0-mini"
    elif args.version=='test': nusc_version = "v1.0-test"

    nusc = NuScenes(version=nusc_version, dataroot=args.root, verbose=True)
    if args.version == 'val':
        scenes = get_split()[1]     # val
    elif args.version == 'test':
        scenes = get_split()[2]     # test
    elif args.version == 'custom_val':
        scenes = get_split()[5]     # custom_val
    else:
        raise ValueError("unknown")

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name'] 
        if scene_name not in scenes:
            continue 

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp 

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True 
        else:
            frame['first'] = False 
        frames.append(frame)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)

def load_dets(args):
    with open(args.checkpoint, 'rb') as f:
        detections=json.load(f)['results']

    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames']
    
    return detections, frames

def objective(trial, args, detections, frames):
    # --- coarse optuna params ---
    # max_age = trial.suggest_int('max_age', 1, 5)
    # det_th = trial.suggest_float('det_th', 0.0, 05)

    # velocity_error = {
    #     'car': trial.suggest_float('car', 2.0, 7.0),
    #     'truck': trial.suggest_float('truck', 2.0, 7.0),
    #     'bus': trial.suggest_float('bus', 3.0, 8.0),
    #     'trailer': trial.suggest_float('trailer', 1.0, 6.0),
    #     'pedestrian': trial.suggest_float('pedestrian', 0.5, 5.0),
    #     'motorcycle': trial.suggest_float('motorcycle', 4.0, 20.0),
    #     'bicycle': trial.suggest_float('bicycle', 1.0, 6.0)  
    # }
    
    # matching_coefs = {
    #     'size': trial.suggest_float('size', 0.0, 1.0),
    #     'dist': trial.suggest_float('dist', 0.0, 1.0),
    #     'confidence': trial.suggest_float('confidence', 0.0, 1.0),
    #     'iou': trial.suggest_float('iou', 0.0, 1.0)
    # }

    # # --- fine optuna params ---
    # max_age = trial.suggest_int('max_age', 4, 4)
    # det_th = trial.suggest_float('det_th', 0.00445, 0.0045)

    # velocity_error = {
    #     'car': trial.suggest_float('car', 2.5, 4.0),
    #     'truck': trial.suggest_float('truck', 2.0, 3.5),
    #     'bus': trial.suggest_float('bus', 2.0, 4.5),
    #     'trailer': trial.suggest_float('trailer', 5.0, 7.0),
    #     'pedestrian': trial.suggest_float('pedestrian', 1.0, 2.5),
    #     'motorcycle': trial.suggest_float('motorcycle', 3.0, 5.0),
    #     'bicycle': trial.suggest_float('bicycle', 1.0, 2.5)  
    # }
    
    # matching_coefs = {
    #     'size': trial.suggest_float('size', 0.45, 0.65),
    #     'dist': trial.suggest_float('dist', 0.45, 0.65),
    #     'confidence': trial.suggest_float('confidence', 0.45, 0.65),
    #     'iou': trial.suggest_float('iou', 0.0, 0.2)
    # }

    # --- test new optuna params ---
    tracker_params = {
        'max_age': trial.suggest_int('max_age', 3, 5),
        'det_th': trial.suggest_float('det_th', 0.0044, 0.0045),
        'hungarian': trial.suggest_categorical('hungarian', [True, False])
    }

    velocity_error = {
        'car': trial.suggest_float('car', 2.5, 4.0),
        'truck': trial.suggest_float('truck', 2.0, 3.5),
        'bus': trial.suggest_float('bus', 2.0, 3.5),
        'trailer': trial.suggest_float('trailer', 5.0, 6.5),
        'pedestrian': trial.suggest_float('pedestrian', 1.0, 2.0),
        'motorcycle': trial.suggest_float('motorcycle', 3.0, 4.5),
        'bicycle': trial.suggest_float('bicycle', 1.0, 2.5)  
    }
    
    matching_coefs = {
        'size': trial.suggest_float('size', 0.4, 0.6),
        'dist': trial.suggest_float('dist', 0.45, 0.7),
        'confidence': trial.suggest_float('confidence', 0.35, 0.7),
        'iou': trial.suggest_float('iou', 0.0, 0.2)
    }

    static_score = trial.suggest_float('static_score', 0.1, 0.5)

    main(args, detections, frames, tracker_params, velocity_error, matching_coefs, static_score)
    metrics = eval_tracking(args)
    # ---------------------------------------

    amota = metrics['amota']
    # amotp = metrics['amotp']
    # recall = metrics['recall']
    # motar = metrics['motar']
    # mota = metrics['mota']
    # motp = metrics['motp']

    # return amota, amotp, recall, motar, mota, motp
    return amota


def main(args, detections, frames, tracker_params, velocity_error, matching_coefs, static_score):
    tracker = Tracker(max_age=tracker_params['max_age'], hungarian=tracker_params['hungarian'], th=tracker_params['det_th'], velocity_error=velocity_error)

    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(frames)

    print("Begin Tracking\n")
    start = time.time()
    for i in range(size):
        # print(i)
        token = frames[i]['token']

        # reset tracking after one video sequence
        if frames[i]['first']:
            # use this for sanity check to ensure your token order is correct
            # print("reset ", i)
            tracker.reset()
            last_time_stamp = frames[i]['timestamp']

        time_lag = (frames[i]['timestamp'] - last_time_stamp) 
        last_time_stamp = frames[i]['timestamp']

        dets = detections[token]

        outputs = tracker.step_centertrack(dets, time_lag, matching_coefs, static_score)
        annos = []

        for item in outputs:
            if item['active'] == 0:
                continue 
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
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

    # res_dir = os.path.join(args.work_dir)
    # if not os.path.exists(res_dir):
    #     os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, f'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)
    # return speed


def optuna_study(args, detections, frames):
    # study = optuna.create_study(directions=["maximize", "minimize", "maximize", "maximize", "maximize", "minimize"],
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=10))

    study.optimize(lambda trial: objective(trial, args, detections, frames), n_trials=30, show_progress_bar=False)

    centerpoint_res_dir = os.path.join(args.work_dir, 'eval_track')
    os.makedirs(centerpoint_res_dir, exist_ok=True)
    results = study.trials_dataframe()
    df = pd.DataFrame(results)
    # df.rename(columns={"values_0": "amota", "values_1": "amotp", "values_2": "recall", "values_3": "motar", "values_4": "mota", "values_5": "motp"}, inplace=True)
    df.rename(columns={"value": "amota"}, inplace=True)
    optimization_results_path = os.path.join(centerpoint_res_dir, 'optimization_results.csv')
    df.to_csv(optimization_results_path)
    print(results)

    print("Best trials:")
    trial = study.best_trials[0]

    print(f"Best try:")
    print(f"  Value: {trial.values}")
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Print the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value

    print('Best parameters:', best_params)
    print('Best objective value:', best_value)

def eval_tracking(args):
    # args = parse_args()
    centerpoint_res_dir = os.path.join(args.work_dir, f'eval_track')
    os.makedirs(centerpoint_res_dir, exist_ok=True)

    metrics_summary = eval(os.path.join(args.work_dir, f'tracking_result.json'),
        args.version,
        centerpoint_res_dir,
        args.root
    )

    return metrics_summary

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
    return metrics_summary


def test_time():
    speeds = []
    for i in range(3):
        speeds.append(optuna_study())

    print("Speed is {} FPS".format( max(speeds)  ))

if __name__ == '__main__':
    args = parse_args()
    save_first_frame(args)
    detections, frames = load_dets(args)
    optuna_study(args, detections, frames)
    # test_time()
    