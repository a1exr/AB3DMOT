import os
import sys
import argparse
import traceback
import pandas as pd

import numpy as np
import math

from functools import partial
import threading
import optuna
from optuna.trial import TrialState
# from utils.sh_utils import get_ip_address

from AB3DMOT_libs.utils import Config
from xinshuo_miscellaneous import get_timestring, print_log
from xinshuo_io import mkdir_if_missing
from main import main_per_cat
from scripts.nuScenes.export_kitti import KittiConverter
from scripts.nuScenes.evaluate import eval_api
from scripts.post_processing.combine_trk_cat import combine_trk_cat

# BASE_PATH = media_server.MEDIA_SERVER_URL
# global edge, video_data, model_id, operate_on_class_ids
# global app
global num_trials, analytics_fps_data, enricher_fps_data, n_trials, recall_data, precision_data, n_fragmantations
# media_server = MediaServer()

PORT = 4001
num_trials = 0
n_trials = 50    #5000
# n_jobs = 4
analytics_fps_data = list()
enricher_fps_data = list()
recall_data = list()
precision_data = list()
n_fragmantations = list()
# app = dash.Dash("detector sweeper app")
# app.layout = html.Div([])

TARGET_INFERENCE_FPS = 30
USE_KALMAN_ACCELERATION_MODEL = 0


def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--dataset', type=str, default='nuScenes', help='KITTI, nuScenes')
    parser.add_argument('--split', type=str, default='', help='train / val / test / mini_train / mini_val / custom_val')
    parser.add_argument('--det_name', type=str, default='', help='BEVFormer / centerpoint / megvili')
    parser.add_argument('--results_title', type=str, default='', help='tracker version title')
    # parser.add_argument('--result_name', type=str, default='', help='nuscenes result file name')
    parser.add_argument('--result_path', type=str, default='', help='track_results_nusc')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    # parser.add_argument('--output_dir', type=str, default='', help='eval_track folder path')

    args = parser.parse_args()
    return args


def objective(trial, args, cfg, kitti_export):
    # global edge, video_data, analytics_fps_data, enricher_fps_data, recall_data, precision_data, model_id, operate_on_class_ids, n_fragmantations

    # --- 3D Multi-Object Tracking @ main.py ---
    # args = parse_args()
    # # load config files
    # config_path = './configs/%s.yml' % args.dataset
    # cfg, settings_show = Config(config_path)
    # cfg['optuna'] = True

    # # overwrite split and detection method
    # if args.split != '': cfg.split = args.split
    # if args.det_name != '': cfg.det_name = args.det_name
    # cfg.save_root = os.path.join(cfg.save_root, args.results_title)

    # print configs
    time_str = get_timestring()
    log = os.path.join(cfg.save_root, 'log/log_%s_%s_%s.txt' % (time_str, cfg.dataset, cfg.split))
    mkdir_if_missing(log); log = open(log, 'w')
    for idx, data in enumerate(settings_show):
        print_log(data, log, display=False)

    kf_configs = {
        'Rx': trial.suggest_float("Rx", 0.0, 1.0),
        'Rz': trial.suggest_float("Rz", 0.5, 2.0),
        'Rtheta': trial.suggest_float("Rtheta", 0.0, 0.33),
        'Rl': trial.suggest_float("Rl", 0.05, 2.0),
        'Rw': trial.suggest_float("Rw", 0.0, 0.25),

        'Px': trial.suggest_int("Px", 2, 15),
        'Pz': trial.suggest_int("Pz", 5, 20),
        'Ptheta': trial.suggest_float("Ptheta", 0.05, 0.5),
        'Pl': trial.suggest_float("Pl", 0.0, 3.5),
        'Pw': trial.suggest_float("Pw", 0.0, 2.0),
        'Pdx': trial.suggest_float("Pdx", 1.5, 4.0),
        'Pdz': trial.suggest_float("Pdz", 2.0, 6.0),
        'Pdtheta': trial.suggest_float("Pdtheta", 0.0, 0.2),
        'Pdl': trial.suggest_float("Pdl", 0.1, 3.0),
        'Pdw': trial.suggest_float("Pdw", 0.1, 1.5),

        'Qdx': trial.suggest_float("Qdx", 0.1, 0.5),
        'Qdz': trial.suggest_float("Qdz", 0.1, 0.5),
        'Qdtheta': trial.suggest_float("Qdtheta", 0.0, 0.2),
        'Qdl': trial.suggest_float("Qdl", 0.1, 0.5),
        'Qdw': trial.suggest_float("Qdw", 0.1, 0.5),
     
        'update_R_method': trial.suggest_categorical("update_R_method", ['dist', 'confidence', 'mean']),
    }
    cfg['kf_configs'] = kf_configs

    # global ID counter used for all categories, not start from 1 for each category to prevent different 
    # categories of objects have the same ID. This allows visualization of all object categories together
    # without ID conflicting, Also use 1 (not 0) as start because MOT benchmark requires positive ID
    ID_start = 1
    # job_num = trial.number						

    # # run tracking for each category
    # for cat in cfg.cat_list:
    #     ID_start = main_per_cat(cfg, cat, log, ID_start)
    cat = cfg.cat_list[0]
    # ID_start = main_per_cat(cfg, cat, log, ID_start, job_num)
    ID_start = main_per_cat(cfg, cat, log, ID_start)
    
    # # combine results for every category
    # print_log('\ncombining results......', log=log)
    # combine_trk_cat(cfg.split, cfg.dataset, args.results_title, cfg.det_name, 'H%d' % cfg.num_hypo, cfg.num_hypo)
    print_log('\nDone!', log=log)
    log.close()

    # --- 3D MOT Evaluation @ export_kitti.py ---
    # result_name = f'{args.det_name}_{cat}_{args.split}_H1'
    # kitti_export = KittiConverter(results_title=args.results_title, result_name=result_name, split=args.split)
    # kitti_export.job_num = job_num
    kitti_export.result_name = f'{args.det_name}_{cfg.cat_list[0]}_{args.split}'
    # kitti_export.kitti_trk_result2nuscenes(job_num)
    kitti_export.kitti_trk_result2nuscenes()

    # --- 3D MOT Evaluation @ evaluate.py ---
    args.eval_set = args.split
    # metrics = eval_api(args, job_num)
    metrics = eval_api(args)
    # ---------------------------------------

    cat = cat.lower()
    amota = metrics.label_metrics['amota'][cat]
    amotp = metrics.label_metrics['amotp'][cat]
    recall = metrics.label_metrics['recall'][cat]
    motar = metrics.label_metrics['motar'][cat]
    mota = metrics.label_metrics['mota'][cat]
    motp = metrics.label_metrics['motp'][cat]

    return amota, amotp, recall, motar, mota, motp


if __name__ == "__main__":
    # # Start webserver for optuna results
    # local_ip = get_ip_address()
    # partial_run = partial(app.run_server, host=local_ip, port=PORT, debug=True, use_reloader=False)
    # t = threading.Thread(target=partial_run)
    # t.start()

    # set_globals_for_test()
    study_name = 'fullTrackerPipe'

    args = parse_args()
    # load config files
    config_path = './configs/%s.yml' % args.dataset
    cfg, settings_show = Config(config_path)
    cfg['optuna_params'] = False
    cfg['optuna_kf'] = True

    # overwrite split and detection method
    if args.split != '': cfg.split = args.split
    if args.det_name != '': cfg.det_name = args.det_name
    cfg.save_root = os.path.join(cfg.save_root, args.results_title)

    result_name = f'{args.det_name}_{cfg.cat_list[0]}_{args.split}_H1'
    kitti_export = KittiConverter(results_title=args.results_title, result_name=result_name, split=args.split)

    sampler = optuna.samplers.TPESampler(seed=10)
    # pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto',
    #                                         reduction_factor=10)  # max_resource should be number of epochs
    # study = optuna.create_study(study_name=study_name,
    #                             directions=["maximize", "minimize", "maximize", "maximize", "maximize", "minimize"],
    #                             sampler=sampler,
    #                             # storage=f'sqlite:///{study_name}.db',
    #                             load_if_exists=True,
    #                             pruner=pruner)
    study = optuna.create_study(study_name=study_name,
                                directions=["maximize", "minimize", "maximize", "maximize", "maximize", "minimize"],
                                sampler=sampler)

    # study.optimize(objective, n_trials=n_trials, callbacks=[update_graphs], show_progress_bar=False)
    # study.optimize(lambda trial: objective(trial, args, cfg, kitti_export), n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
    study.optimize(lambda trial: objective(trial, args, cfg, kitti_export), n_trials=n_trials, show_progress_bar=False)
    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    output_dir = os.path.join(args.result_path, 'eval_track')
    results = study.trials_dataframe()
    df = pd.DataFrame(results)
    df.rename(columns={"values_0": "amota", "values_1": "amotp", "values_2": "recall", "values_3": "motar", "values_4": "mota", "values_5": "motp"}, inplace=True)
    optimization_results_path = os.path.join(output_dir, 'optimization_results.csv')
    df.to_csv(optimization_results_path)
    print(results)

    print("Best trials:")
    trial = study.best_trials[0]

    print(f"Best try:")
    print(f"  Value: {trial.values}")
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

