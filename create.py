import os
os.environ["OMP_NUM_THREADS"] = "1"
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.structures.sweep import Sweep
from av2.structures.cuboid import CuboidList, Cuboid
from av2.utils.io import read_feather
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3

import os
import multiprocessing
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool, current_process
from typing import Optional, Tuple, Dict, Union
from tqdm import tqdm
import numpy as np




CATEGORY_MAP = {"ANIMAL":0, "ARTICULATED_BUS":1, "BICYCLE":2, "BICYCLIST":3, "BOLLARD":4,
                "BOX_TRUCK":5, "BUS":6, "CONSTRUCTION_BARREL":7, "CONSTRUCTION_CONE":8, "DOG":9,
                "LARGE_VEHICLE":10, "MESSAGE_BOARD_TRAILER":11, "MOBILE_PEDESTRIAN_CROSSING_SIGN":12,
                "MOTORCYCLE":13, "MOTORCYCLIST":14, "OFFICIAL_SIGNALER":15, "PEDESTRIAN":16,
                "RAILED_VEHICLE":17, "REGULAR_VEHICLE":18, "SCHOOL_BUS":19, "SIGN":20,
                "STOP_SIGN":21, "STROLLER":22, "TRAFFIC_LIGHT_TRAILER":23, "TRUCK":24,
                "TRUCK_CAB":25, "VEHICULAR_TRAILER":26, "WHEELCHAIR":27, "WHEELED_DEVICE":28,
                "WHEELED_RIDER":29}


def get_ids_and_cuboids_at_lidar_timestamp(dataset: AV2SensorDataLoader,
                                           log_id: str,
                                           lidar_timestamp_ns: int) -> Dict[str, Cuboid]:
    """Load the sweep annotations at the provided timestamp with unique ids.
        Args:
            log_id: Log unique id.
            lidar_timestamp_ns: Nanosecond timestamp.
        Returns:
            dict mapping ids to cuboids
        """
    annotations_feather_path = dataset._data_dir / log_id / "annotations.feather"

    # Load annotations from disk.
    # NOTE: This file contains annotations for the ENTIRE sequence.
    # The sweep annotations are selected below.
    cuboid_list = CuboidList.from_feather(annotations_feather_path)


    raw_data = read_feather(annotations_feather_path)
    ids = raw_data.track_uuid.to_numpy()

    cuboids_and_ids = dict(filter(lambda x: x[1].timestamp_ns == lidar_timestamp_ns,
                                  zip(ids, cuboid_list.cuboids)))

    return cuboids_and_ids

def compute_sceneflow(dataset: AV2SensorDataLoader, log_id: str,
                      timestamps: Tuple[int, int]) -> Dict[str, Union[np.ndarray, SE3]]:
    """Compute sceneflow between the sweeps at the given timestamps.
        Args:
          dataset: Sensor dataset.
          log_id: unique id.
          timestamps: the timestamps of the lidar sweeps to compute flow between
        Returns:
          Dictionary with fields:
            pcl_0: Nx3 array containing the points at time 0
            pcl_1: Mx3 array containing the points at time 1
            flow_0_1: Nx3 array containing flow from timestamp 0 to 1
            flow_1_0: Mx3 array containing flow from timestamp 1 to 0
            valid_0: Nx1 array indicating if the returned flow from 0 to 1 is valid (1 for valid, 0 otherwise)
            valid_1: Mx1 array indicating if the returned flow from 1 to 0 is valid (1 for valid, 0 otherwise)
            classes_0: Nx1 array containing the class ids for each point in sweep 0
            classes_1: Nx1 array containing the class ids for each point in sweep 0
            pose_0: SE3 pose at time 0
            pose_1: SE3 pose at time 1
            ego_motion: SE3 motion from sweep 0 to sweep 1
    """
    def compute_flow(sweeps, cuboids, poses):
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        
        flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz) -  sweeps[0].xyz
        # Convert to float32s
        flow = flow.astype(np.float32)
        
        valid = np.ones(len(sweeps[0].xyz), dtype=np.bool_)
        classes = -np.ones(len(sweeps[0].xyz), dtype=np.int8)
        
        
        for id in cuboids[0]:
            c0 = cuboids[0][id]
            c0.length_m += 0.2 # the bounding boxes are a little too tight and some points are missed
            c0.width_m += 0.2
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
            classes[obj_mask] = CATEGORY_MAP[c0.category]
        
            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                flow[obj_mask] = obj_flow.astype(np.float32)
            else:
                valid[obj_mask] = 0
        return flow, classes, valid, ego1_SE3_ego0
    
    sweeps = [Sweep.from_feather(dataset.get_lidar_fpath(log_id, ts)) for ts in timestamps]
    cuboids = [get_ids_and_cuboids_at_lidar_timestamp(dataset, log_id, ts) for ts in timestamps]
    poses = [dataset.get_city_SE3_ego(log_id, ts) for ts in timestamps]

    flow_0_1, classes_0, valid_0, ego_motion = compute_flow(sweeps, cuboids, poses)
    flow_1_0, classes_1, valid_1, _ = compute_flow([sweeps[1], sweeps[0]],
                                                  [cuboids[1], cuboids[0]],
                                                  [poses[1], poses[0]])


    return {'pcl_0': sweeps[0].xyz, 'pcl_1' :sweeps[1].xyz,
            'flow_0_1': flow_0_1, 'flow_1_0': flow_1_0,
            'valid_0': valid_0, 'valid_1': valid_1,
            'classes_0': classes_0, 'classes_1': classes_1,
            'pose_0': poses[0], 'pose_1': poses[1],
            'ego_motion': ego_motion}

def process_log(dataset: AV2SensorDataLoader, log_id: str, output_dir: Path, skip_pcs : bool, skip_reverse_flow : bool, n: Optional[int] = None) :
    """Outputs sceneflow and auxillary information for each pair of pointclouds in the
       dataset. Output files have the format <output_dir>/<log_id>_<sweep_1_timestamp>.npz
        Args:
          dataset: Sensor dataset to process.
          log_id: Log unique id.
          output_dir: Output_directory.
          n: the position to use for the progres bar
        Returns:
          None
    """
    log_map_dirpath = dataset._data_dir / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)
    timestamps = dataset.get_ordered_log_lidar_timestamps(log_id)

    if n is not None:
        iter_bar = tqdm(zip(timestamps, timestamps[1:]), leave=False,
                         total=len(timestamps) - 1, position=n,
                         desc=f'Log {log_id}')
    else:
        iter_bar = zip(timestamps, timestamps[1:])
    
    for ts0, ts1 in iter_bar:
        flow = compute_sceneflow(dataset, log_id, (ts0, ts1))
        pcl_city_0 = flow['pose_0'].transform_point_cloud(flow['pcl_0'])
        pcl_city_1 = flow['pose_1'].transform_point_cloud(flow['pcl_1'])

        is_ground_0 = avm.get_ground_points_boolean(pcl_city_0)
        is_ground_1 = avm.get_ground_points_boolean(pcl_city_1)

        output = {k:flow[k] for k in ['pcl_0', 'pcl_1',
                                      'flow_0_1', 'flow_1_0',
                                      'valid_0', 'valid_1',
                                      'classes_0', 'classes_1']}
        output['is_ground_0'] = is_ground_0.astype(np.bool_)
        output['is_ground_1'] = is_ground_1.astype(np.bool_)
        output['ego_motion'] = flow['ego_motion'].transform_matrix.astype(np.float32)
        output['pcl_0'] = output['pcl_0'].astype(np.float32)
        output['pcl_1'] = output['pcl_1'].astype(np.float32)

        if skip_pcs:
            output.pop('pcl_0')
            output.pop('pcl_1')
        if skip_reverse_flow:
            output.pop('flow_1_0')

        np.savez(output_dir / f'{log_id}_{ts0}.npz', **output)

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def process_logs(data_dir: Path, output_dir: Path, nproc: int, skip_pcs: bool, skip_reverse_flow: bool):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
       Args:
         data_dir: Argoverse 2.0 directory
         output_dir: Output directory.
    """
    
    if not data_dir.exists():
        print(f'{data_dir} not found')
        return
    
    split_output_dir = output_dir
    split_output_dir.mkdir(exist_ok=True, parents=True)
    
    dataset = AV2SensorDataLoader(data_dir=data_dir, labels_dir=data_dir)
    logs = dataset.get_log_ids()
    args = sorted([(dataset, log, split_output_dir, skip_pcs, skip_reverse_flow) for log in logs])
    
    print(f'Using {nproc} processes')
    if nproc <= 1:
        for x in tqdm(args):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(logs)))

if __name__ == '__main__':
    parser = ArgumentParser(prog='create',
                            description='Create a LiDAR sceneflow dataset from Argoveser 2.0 Sensor')
    parser.add_argument('--argo_dir', type=str, help='The top level directory contating the input dataset')
    parser.add_argument('--output_dir', type=str, help='The location to output the sceneflow files to')
    parser.add_argument('--nproc', type=int, default=(multiprocessing.cpu_count() - 1))
    parser.add_argument('--skip_pcs', action='store_true')
    parser.add_argument('--skip_reverse_flow', action='store_true')


    args = parser.parse_args()
    data_root = Path(args.argo_dir)
    output_dir = Path(args.output_dir)

    process_logs(data_root, output_dir, args.nproc, args.skip_pcs, args.skip_reverse_flow)
