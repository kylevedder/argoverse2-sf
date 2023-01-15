import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import util
from typing import Union, Dict, List
from pathlib import Path
import argparse
from tqdm import tqdm


CATEGORY_MAP = {"ANIMAL":0, "ARTICULATED_BUS":1, "BICYCLE":2, "BICYCLIST":3, "BOLLARD":4,
                "BOX_TRUCK":5, "BUS":6, "CONSTRUCTION_BARREL":7, "CONSTRUCTION_CONE":8, "DOG":9,
                "LARGE_VEHICLE":10, "MESSAGE_BOARD_TRAILER":11, "MOBILE_PEDESTRIAN_CROSSING_SIGN":12,
                "MOTORCYCLE":13, "MOTORCYCLIST":14, "OFFICIAL_SIGNALER":15, "PEDESTRIAN":16,
                "RAILED_VEHICLE":17, "REGULAR_VEHICLE":18, "SCHOOL_BUS":19, "SIGN":20,
                "STOP_SIGN":21, "STROLLER":22, "TRAFFIC_LIGHT_TRAILER":23, "TRUCK":24,
                "TRUCK_CAB":25, "VEHICULAR_TRAILER":26, "WHEELCHAIR":27, "WHEELED_DEVICE":28,
                "WHEELED_RIDER":29, "NONE": -1}

BACKGROUND_CATEGORIES = ['BOLLARD', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 
                         'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'SIGN', 'STOP_SIGN', 'NONE']
PEDESTRIAN_CATEGORIES = ['PEDESTRIAN', 'STROLLER', 'WHEELCHAIR', 'OFFICIAL_SIGNALER']
SMALL_VEHICLE_CATEGORIES = ['BICYCLE', 'BICYCLIST', 'MOTORCYCLE', 'MOTORCYCLIST',
                            'WHEELED_DEVICE', 'WHEELED_RIDER']
VEHICLE_CATEGORIES = ['ARTICULATED_BUS', 'BOX_TRUCK', 'BUS', 'LARGE_VEHICLE', 'RAILED_VEHICLE',
                      'REGULAR_VEHICLE', 'SCHOOL_BUS', 'TRUCK', 'TRUCK_CAB',
                      'VEHICULAR_TRAILER', 'TRAFFIC_LIGHT_TRAILER', 'MESSAGE_BOARD_TRAILER']
ANIMAL_CATEGORIES = ['ANIMAL', 'DOG']

NO_CLASSES = {'All': [k for k in range(-1, 30)]}
FOREGROUND_BACKGROUND = {'Background': [CATEGORY_MAP[k] for k in BACKGROUND_CATEGORIES],
                         'Foreground': [CATEGORY_MAP[k] for k in (PEDESTRIAN_CATEGORIES +
                                                                  SMALL_VEHICLE_CATEGORIES +
                                                                  VEHICLE_CATEGORIES +
                                                                  ANIMAL_CATEGORIES)]}
PED_CYC_VEH_ANI = {'Background': [CATEGORY_MAP[k] for k in BACKGROUND_CATEGORIES],
                   'Pedestrian': [CATEGORY_MAP[k] for k in PEDESTRIAN_CATEGORIES],
                   'Small Vehicle': [CATEGORY_MAP[k] for k in SMALL_VEHICLE_CATEGORIES],
                   'Vehicle': [CATEGORY_MAP[k] for k in VEHICLE_CATEGORIES],
                   'Animal': [CATEGORY_MAP[k] for k in ANIMAL_CATEGORIES]}

Array = Union[np.ndarray, torch.Tensor]

def epe(pred, gt):
    return torch.sqrt(torch.sum((pred - gt) ** 2, -1))

def accuracy(pred, gt, threshold):
    l2_norm = torch.sqrt(torch.sum((pred - gt) ** 2, -1))
    gt_norm = torch.sqrt(torch.sum(gt * gt, -1))
    relative_err = l2_norm / (gt_norm + 1e-20)
    error_lt_5 = (l2_norm < threshold).bool()
    relative_err_lt_5 = (relative_err < threshold).bool()
    return  (error_lt_5 | relative_err_lt_5).float()


def accuracy_strict(pred, gt):
    return accuracy(pred, gt, 0.05)


def accuracy_relax(pred, gt):
    return accuracy(pred, gt, 0.10)


def outliers(pred, gt):
    l2_norm = torch.sqrt(torch.sum((pred - gt) ** 2, -1))
    gt_norm = torch.sqrt(torch.sum(gt * gt, -1))
    relative_err = l2_norm / (gt_norm + 1e-20)

    l2_norm_gt_3 = (l2_norm > 0.3).bool()
    relative_err_gt_10 = (relative_err > 0.1).bool()
    return (l2_norm_gt_3 | relative_err_gt_10).float()


def angle_error(pred, gt):
    unit_label = gt / gt.norm(dim=-1, keepdim=True)
    unit_pred = pred / pred.norm(dim=-1, keepdim=True)
    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(-1).clamp(min=-1+eps, max=1-eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    return torch.acos(dot_product)


def coutn(pred, gt):
    return torch.ones(len(pred))


METRICS = {'EPE': epe, 'Accuracy Strict': accuracy_strict, 'Accuracy Relax': accuracy_relax,
           'Outliers': outliers, 'Angle Error': angle_error}



                   

def metrics(inpt: Dict[str, Array], pred: Array, object_classes: Dict[str, List[int]],
            inpt_name: str, dynamic_threshold: float = 0.05):
    
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    d: Dict[str, torch.Tensor] = util.numpy_to_torch(inpt)

    valid = (d['valid_0'] == 1)
    pred = pred[valid]
    gt = d['flow_0_1'][valid]
    pc = d['pcl_0'][valid]

    dynamic_mask = util.dynamism(pc, gt, d['ego_motion']) > dynamic_threshold
    
    classes = d['classes_0'][valid]

    results = []
    for cls, class_idxs in object_classes.items():
        class_mask = classes == class_idxs[0]
        for i in class_idxs[1:]:
            class_mask = class_mask | (classes == i)

        class_dynamic = class_mask & dynamic_mask
        class_static = class_mask & ~dynamic_mask

        dcnt = class_dynamic.sum().item()
        scnt = class_static.sum().item()

        gt_class_dynamic = gt[class_dynamic]
        pred_class_dynamic = pred[class_dynamic]
        
        gt_class_static = gt[class_static]
        pred_class_static = pred[class_static]

        dynamic_results = [inpt_name, cls, 'Dynamic', dcnt]
        if dcnt > 0:
             dynamic_results += [METRICS[m](pred_class_dynamic, gt_class_dynamic).mean().cpu().item()
                                 for m in METRICS]
        else:
            dynamic_results += [np.nan for m in METRICS]
        static_results = [inpt_name, cls, 'Static', scnt]
        if scnt > 0:
             static_results += [METRICS[m](pred_class_static, gt_class_static).mean().cpu().item()
                                for m in METRICS]
        else:
            static_results += [np.nan for m in METRICS]

        results.append(dynamic_results)
        results.append(static_results)

    return results

def compute_results_dataframe(gt_dir: Path, results_dir: Path, object_classes):
    result_files = list(results_dir.glob('*.npy'))
    gt_files = [gt_dir / rf.with_suffix('.npz').name for rf in result_files]

    results = []
    for result_file, gt_file in tqdm(list(zip(result_files, gt_files))):
        if not gt_file.exists():
            raise(ValueError(f'Result file {result_file.name} has no corresponding ground truth'))

        pred = np.load(result_file)
        inpt = dict(np.load(gt_file))

        results += metrics(inpt, pred, object_classes, result_file.stem)

    return pd.DataFrame(results, columns=['Example', 'Class', 'Motion', 'Count'] + list(METRICS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='eval',
                                     description='Evaluate scene flow predictions.'
                                     'Predictions should npy files in a single directory'
                                     'with the names formatted as <log_id>_<timestamp>.npy')
    parser.add_argument('--gt-dir', type=str, help='path/to/ground/truth')
    parser.add_argument('--pred-dir', type=str, help='path/to/ground/truth')
    parser.add_argument('--output_file', type=str, help='path/to/result/output_file.parquet',
                        default='results.parquet')
    parser.add_argument('--breakdown', choices=['none', 'fgbg', 'object'], default='fgbg',
                        help='What class types to break the mestrics across')

    args = parser.parse_args()

    object_classes = {'none': NO_CLASSES,
                      'fgbg': FOREGROUND_BACKGROUND,
                      'object': PED_CYC_VEH_ANI}[args.breakdown]

    df = compute_results_dataframe(Path(args.gt_dir),
                                   Path(args.pred_dir),
                                   object_classes)

    df.to_parquet(args.output_file)

    
