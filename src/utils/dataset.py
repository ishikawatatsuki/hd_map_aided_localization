import pykitti
import numpy as np
import geopandas as gpd
from scipy.spatial.transform import Rotation
from src.utils.transformation_utils import CoordTransformer
from collections import namedtuple


KITTI_SEQUENCE_TO_DRIVE = {
    "01": "0042",
    "02": "0034",
    "03": "0067",
    "04": "0016",
    "05": "0018",
    "06": "0020",
    "07": "0027",
    "08": "0028",
    "09": "0033",
    "10": "0034",
}

KITTI_SEQUENCE_TO_DATE = {
    "01": "2011_10_03",
    "02": "2011_10_03",
    "03": "2011_09_26",
    "04": "2011_09_30",
    "05": "2011_09_30",
    "06": "2011_09_30",
    "07": "2011_09_30",
    "08": "2011_09_30",
    "09": "2011_09_30",
    "10": "2011_09_30",
}

KittiData = namedtuple("KittiData", [
    "timestamps",
    "target_timestamps",
    "input_indices",
    "target_indices",
    "acc",
    "gyro",
    "imu_windows",
    "lla_position",
    "positions",
    "velocities",
    "orientations",
    "initial_lla",
    "initial_position",
    "initial_velocity",
    "initial_orientation",
    "edges",
    "transformer"
])

# Initialize EKF
def get_kitti_data(seq: str, root_dir: str = "./data/KITTI", is_sync: bool = True) -> KittiData:


    edges_map = gpd.read_parquet(f'{root_dir}/raw/karlsruhe_edges.parquet')

    if is_sync:
        root_dir = f"{root_dir}/sync"
    else:
        root_dir = f"{root_dir}/unsync"
        
    date = KITTI_SEQUENCE_TO_DATE[seq]
    drive = KITTI_SEQUENCE_TO_DRIVE[seq]

    dataset_type = "sync" if is_sync else "extract"

    dataset = pykitti.raw(root_dir, date, drive, dataset=dataset_type)
    transformer = CoordTransformer()

    ts_list = []
    acc_list = []
    gyro_list = []
    lla_list = []
    position_list = []
    velocity_list = []
    orientation_list = []

    for i, oxts in enumerate(dataset.oxts):
        packet = oxts.packet
        a = np.array([packet.ax, packet.ay, packet.az])
        w = np.array([packet.wx, packet.wy, packet.wz])
        orientation = np.array([packet.roll, packet.pitch, packet.yaw])
        velocity = np.array([packet.vf, packet.vl, packet.vu])
        lla = np.array([packet.lon, packet.lat, packet.alt])
        position = transformer.transform(lla).flatten()

        ts_list.append(dataset.timestamps[i])
        acc_list.append(a)
        gyro_list.append(w)
        lla_list.append(lla)
        position_list.append(position)
        velocity_list.append(velocity)
        orientation_list.append(orientation)

    ts_array = np.array(ts_list)
    acc_array = np.array(acc_list)
    gyro_array = np.array(gyro_list)
    lla_array = np.array(lla_list)
    position_array = np.array(position_list)
    velocity_array = np.array(velocity_list)
    orientation_array = np.array(orientation_list)

    position_array_full = position_array.copy()
    velocity_array_full = velocity_array.copy()
    orientation_array_full = orientation_array.copy()
    lla_array_full = lla_array.copy()

    min_lon, min_lat, _ = np.min(lla_array, axis=0)
    max_lon, max_lat, _ = np.max(lla_array, axis=0)
    local_edges = edges_map.cx[min_lon:max_lon, min_lat:max_lat]

    if is_sync:
        # For sync data, we can directly use the 10Hz samples without windowing.
        return KittiData(
            timestamps=ts_array,
            target_timestamps=ts_array,
            input_indices=np.arange(len(ts_array)),
            target_indices=np.arange(len(ts_array)),
            acc=acc_array,
            gyro=gyro_array,
            imu_windows=None,  # No windows needed for sync data
            lla_position=lla_array,
            positions=position_array,
            velocities=velocity_array,
            orientations=np.array(
                [Rotation.from_euler('xyz', euler).as_quat(scalar_first=True) for euler in orientation_array]
            ),
            initial_lla=lla_array_full[0],
            initial_position=position_array_full[0],
            initial_velocity=velocity_array_full[0],
            initial_orientation=orientation_array_full[0],
            edges=local_edges,
            transformer=transformer
        )
    
    # 100Hz -> 10Hz pairing:
    # IMU at time t (indices 9, 19, 29, ...) predicts position at t+1 step (indices 19, 29, 39, ...).
    step = 10
    imu_indices_10hz = np.arange(step - 1, len(ts_array), step)

    if imu_indices_10hz.size < 2:
        raise RuntimeError(
            f"Sequence {seq} is too short to create 10Hz IMU/GT pairs with +1-step GT shift."
        )

    # Input at t
    input_indices = imu_indices_10hz[:-1]
    # Target at t+1 (one 10Hz step ahead)
    target_indices = imu_indices_10hz[1:]

    ts_input = ts_array[input_indices]
    ts_target = ts_array[target_indices]

    # 10Hz IMU values at time t (used as explicit "current" samples if needed)
    acc_array = acc_array[input_indices]
    gyro_array = gyro_array[input_indices]

    # 100Hz IMU windows: for each t, collect [t-9, ..., t] (shape: N x 10 x 6)
    raw_imu = np.concatenate([np.array(acc_list), np.array(gyro_list)], axis=1)
    window_starts = input_indices - (step - 1)
    window_indices = window_starts[:, None] + np.arange(step)[None, :]
    imu_windows = raw_imu[window_indices]

    lla_array = lla_array[target_indices]
    position_array = position_array[target_indices]
    velocity_array = velocity_array[target_indices]
    orientation_quat_array = np.array(
        [Rotation.from_euler('xyz', euler).as_quat(scalar_first=True) for euler in orientation_array]
    )[target_indices]

    return KittiData(
        timestamps=ts_input,
        target_timestamps=ts_target,
        input_indices=input_indices,
        target_indices=target_indices,
        acc=acc_array,
        gyro=gyro_array,
        imu_windows=imu_windows,
        lla_position=lla_array,
        positions=position_array,
        velocities=velocity_array,
        orientations=orientation_quat_array,
        initial_lla=lla_array_full[0],
        initial_position=position_array_full[0],
        initial_velocity=velocity_array_full[0],
        initial_orientation=orientation_array_full[0],
        edges=local_edges,
        transformer=transformer
    )