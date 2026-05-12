from dataclasses import dataclass
from typing import List, Optional, Callable, Union
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils.filter_utils import F_numpy, G_numpy, H_pos_heading_numpy, H_yaw_numpy, H_quat_numpy, H_pos_numpy, H_vel_numpy, H_lu_vel_body_numpy
from enum import Enum, auto

def inject_additive(x, dx):
    x_new = x + dx
    x_new[6:10] /= np.linalg.norm(x_new[6:10])
    return x_new

def residual_linear(z, zhat, x):
    return z - zhat

def residual_velocity_constraint(z, zhat, x):
    rot_bw = Rotation.from_quat(x[6:10], scalar_first=True)
    v_body = rot_bw.inv().apply(x[3:6])
    return z - v_body[1:3] # leftward and upward velocity components in body frame

def residual_heading(z, zhat, x):
    r = z - zhat
    r[0] = np.arctan2(np.sin(r[0]), np.cos(r[0]))
    return r

def residual_quat(z, zhat, x):
    q_meas = z.copy()
    q_pred = zhat.copy()
    if np.dot(q_meas, q_pred) < 0.0:
        q_meas = -q_meas
    return q_meas - q_pred

def measurement_heading(x):
    q = np.asarray(x[6:10], dtype=float)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=float)
    yaw = Rotation.from_quat(q_xyzw).as_euler('XYZ')[2]
    return np.array([yaw], dtype=float)

def measurement_quat(x):
    return np.asarray(x[6:10], dtype=float)

def inject_heading_yaw_only(x, dx):
    x_new = np.asarray(x, dtype=float).copy()
    yaw_pred = measurement_heading(x_new)[0]
    H_yaw = H_yaw_numpy(x_new.tolist())
    yaw_delta = float((H_yaw @ np.asarray(dx, dtype=float).reshape(-1, 1)).squeeze())
    yaw_upd = np.arctan2(np.sin(yaw_pred + yaw_delta), np.cos(yaw_pred + yaw_delta))

    q_wxyz = x_new[6:10]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
    euler_zyx = Rotation.from_quat(q_xyzw).as_euler('ZYX')
    euler_zyx[0] = yaw_upd
    q_new_xyzw = Rotation.from_euler('ZYX', euler_zyx).as_quat()
    q_new_wxyz = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]], dtype=float)
    q_new_wxyz /= np.linalg.norm(q_new_wxyz)
    x_new[6:10] = q_new_wxyz
    return x_new


class FusionData(Enum):
    LINEAR_ACCELERATION = auto()
    ANGULAR_VELOCITY = auto()
    POSITION = auto()
    LINEAR_VELOCITY = auto()
    ORIENTATION = auto()
    MAGNETIC_FIELD = auto()
    DISPLACEMENT = auto()
    HEADING_ANGLE = auto()
    CONTROL_SIGNAL = auto()
    VELOCITY_CONSTRAINT = auto()

    FORWARD_VELOCITY = auto()
    LEFTWARD_VELOCITY = auto()
    UPWARD_VELOCITY = auto()

    @staticmethod
    def get_enum_name_list():
        return [
            s.lower() for s in list(FusionData.__members__.keys())
        ]

    @classmethod
    def get_type(cls, s: str):
        s = s.lower()
        try:
            index = FusionData.get_enum_name_list().index(s)
            return cls(index + 1)
        except:
            return None


@dataclass
class MeasurementBlock:
    z: np.ndarray
    R: np.ndarray
    H: Callable[[np.ndarray], np.ndarray]
    residual: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    inject: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    h: Optional[Callable[[np.ndarray], np.ndarray]] = None
    enabled: bool = True
    mask: Optional[np.ndarray] = None  # for batch updates with multiple clones
    measurement_type: Optional[FusionData] = None

    @classmethod
    def from_measurement_type(cls, measurement_type: FusionData, z: np.ndarray, R: np.ndarray, mask: Optional[np.ndarray] = None):
        if measurement_type == FusionData.HEADING_ANGLE:
            return cls(
                z=z,
                R=R,
                H=H_yaw_numpy,
                residual=residual_heading,
                inject=inject_heading_yaw_only,
                h=measurement_heading,
                mask=mask,
                measurement_type=measurement_type,
                enabled=True
            )
        elif measurement_type == FusionData.ORIENTATION:
            return cls(
                z=z,
                R=R,
                H=H_quat_numpy,
                residual=residual_quat,
                inject=inject_additive,
                h=measurement_quat,
                mask=mask,
                measurement_type=measurement_type,
                enabled=True
            )
        elif measurement_type == FusionData.POSITION:
            return cls(
                z=z,
                R=R,
                H=H_pos_numpy,
                residual=residual_linear,
                inject=inject_additive,
                mask=mask,
                measurement_type=measurement_type,
                enabled=True
            )
        elif measurement_type == FusionData.LINEAR_VELOCITY:
            return cls(
                z=z,
                R=R,
                H=H_vel_numpy,
                residual=residual_linear,
                inject=inject_additive,
                mask=mask,
                measurement_type=measurement_type,
                enabled=True
            )
        elif measurement_type == FusionData.VELOCITY_CONSTRAINT:
            return cls(
                z=z,
                R=R,
                H=H_lu_vel_body_numpy,
                residual=residual_velocity_constraint,
                inject=inject_additive,
                mask=mask,
                measurement_type=measurement_type,
                enabled=True
            )
        else:
            raise NotImplementedError(f"Measurement type {measurement_type} not implemented")


@dataclass
class PredictionRecord:
    """A snapshot of the EKF state at a given time, for rollback."""
    u: np.ndarray  # concatenated [acc, gyro]
    dt: float
    F: np.ndarray
    G: np.ndarray
    timestamp: float  # epoch seconds or any monotonic time


@dataclass
class CorrectionRecord:
    """A single correction point with its timestamp."""
    timestamp: float
    block: MeasurementBlock


@dataclass
class RollbackInfo:
    type: str  # 'prediction' or 'correction'
    timestamp: float
    data: Union[PredictionRecord, CorrectionRecord]

BASE_DIM = 16

class EKF:
    def __init__(self):
        self.x = np.zeros(BASE_DIM)
        self.x[6] = 1.0

        self.gyro_bias_noise_std = 5.817764173314432e-05
        self.acc_bias_noise_std = 8.333333333333333e-05
        self.gyro_noise_std = 5.817764173314432e-05
        self.acc_noise_std = 8.333333333333333e-05
        self.gravity = np.array([0, 0, -9.81])

        self.P = np.eye(BASE_DIM) * 0.1
        self.Q = np.diag(
            [self.acc_noise_std**2] * 3 + [self.gyro_noise_std**2] * 3 +
            [self.acc_bias_noise_std**2] * 3 + [self.gyro_bias_noise_std**2] * 3
        )

        self.predictions: List[PredictionRecord] = []
        self.corrections: List[CorrectionRecord] = []
        self.x_snapshot = self.x.copy()
        self.P_snapshot = self.P.copy()

        self.position_consistency_min_dt = 1e-3
        self.position_consistency_min_displacement = 0.25
        self.position_consistency_min_speed = 0.5
        self.position_consistency_velocity_blend = 0.5
        self.position_consistency_velocity_var = 1.0
        self.position_consistency_yaw_var = np.deg2rad(10.0) ** 2

    def init_state(self, pos, vel, quat):
        self.x[0:3] = pos
        self.x[3:6] = vel
        self.x[6:10] = quat

    def omega(self, w):
        wx, wy, wz = w
        return np.array([
            [0.0, -wx, -wy, -wz],
            [wx,   0.0,  wz, -wy],
            [wy,  -wz,  0.0,  wx],
            [wz,   wy, -wx,  0.0],
        ])

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return float(Rotation.from_quat(quat_xyzw).as_euler('XYZ')[2])

    @staticmethod
    def _set_quat_yaw_wxyz(yaw: float, quat_wxyz: np.ndarray) -> np.ndarray:
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        euler_zyx = Rotation.from_quat(quat_xyzw).as_euler('ZYX')  # [yaw, pitch, roll]
        euler_zyx[0] = yaw
        corrected_xyzw = Rotation.from_euler('ZYX', euler_zyx).as_quat()
        q_new = np.array([corrected_xyzw[3], corrected_xyzw[0], corrected_xyzw[1], corrected_xyzw[2]])
        q_new /= np.linalg.norm(q_new)
        return q_new

    def _propagate_state(self, x, acc, gyro, dt):
        pos = x[0:3]
        vel = x[3:6]
        quat = x[6:10]
        acc_bias = x[10:13]
        gyro_bias = x[13:16]

        # new_acc_bias = acc_bias
        # new_gyro_bias = gyro_bias
        new_acc_bias = acc_bias + np.random.normal(
            0, self.acc_bias_noise_std**2, 3
        ) * dt
        new_gyro_bias = gyro_bias + np.random.normal(
            0, self.gyro_bias_noise_std**2, 3
        ) * dt

        acc_unbiased = acc - acc_bias
        gyro_unbiased = gyro - gyro_bias

        omega_mat = self.omega(gyro_unbiased)
        new_quat = (np.eye(4) + 0.5 * dt * omega_mat) @ quat
        new_quat /= np.linalg.norm(new_quat)

        corrected_rot = Rotation.from_quat(new_quat, scalar_first=True)
        acc_world = corrected_rot.apply(acc_unbiased) + self.gravity

        new_vel = vel + acc_world * dt
        new_pos = pos + new_vel * dt

        x_new = x.copy()
        x_new[0:3] = new_pos
        x_new[3:6] = new_vel
        x_new[6:10] = new_quat
        x_new[10:13] = new_acc_bias
        x_new[13:16] = new_gyro_bias
        return x_new, acc_unbiased, gyro_unbiased

    def predict(self, acc, gyro, dt, timestamp: float = 0.0):

        new_state, acc_unbiased, gyro_unbiased = self._propagate_state(self.x, acc, gyro, dt)

        self.x[0:3] = new_state[0:3]
        self.x[3:6] = new_state[3:6]
        self.x[6:10] = new_state[6:10]
        self.x[10:13] = new_state[10:13]
        self.x[13:16] = new_state[13:16]

        F = F_numpy(self.x, np.concatenate([acc_unbiased, gyro_unbiased]), dt)
        G = G_numpy(self.x, np.concatenate([acc_unbiased, gyro_unbiased]), dt)

        self.P = F @ self.P @ F.T + G @ self.Q @ G.T

        self.predictions.append(PredictionRecord(
            u=np.concatenate([acc, gyro]),
            dt=dt,
            F=F.copy(),
            G=G.copy(),
            timestamp=timestamp,
        ))

    def correct(self, blocks: List[MeasurementBlock], timestamp: float = 0.0):
        
        for b in blocks:
            if not b.enabled:
                continue

            z = np.asarray(b.z, dtype=float).reshape(-1)
            R = np.asarray(b.R, dtype=float)
            H = np.asarray(b.H(self.x.tolist()), dtype=float)
            if b.h is None:
                zhat = H @ self.x
            else:
                zhat = np.asarray(b.h(self.x), dtype=float).reshape(-1)

            if H.ndim != 2 or H.shape[1] != self.x.shape[0]:
                raise ValueError(f"H shape mismatch: got {H.shape}, expected (*, {self.x.shape[0]})")
            if z.shape[0] != H.shape[0]:
                raise ValueError(f"z/H mismatch: z={z.shape[0]}, H rows={H.shape[0]}")
            if zhat.shape[0] != z.shape[0]:
                raise ValueError(f"zhat/z mismatch: zhat={zhat.shape[0]}, z={z.shape[0]}")
            if R.shape != (z.shape[0], z.shape[0]):
                raise ValueError(f"R shape mismatch: got {R.shape}, expected {(z.shape[0], z.shape[0])}")

            y = b.residual(z, zhat, self.x)
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)

            if b.mask is not None:
                K = K * np.asarray(b.mask, dtype=float).reshape(-1, 1)
            dx = K @ y

            if b.inject is None:
                self.x = self.x + dx
                self.x[6:10] /= np.linalg.norm(self.x[6:10])
            else:
                self.x = b.inject(self.x, dx)

            I = np.eye(self.x.shape[0])
            IKH = I - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

            self.corrections.append(CorrectionRecord(
                timestamp=timestamp,
                block=MeasurementBlock(
                    z=z.copy(),
                    R=R.copy(),
                    H=b.H,
                    residual=b.residual,
                    inject=b.inject,
                    h=b.h,
                    enabled=b.enabled,
                    mask=None if b.mask is None else np.asarray(b.mask, dtype=bool).copy(),
                    measurement_type=b.measurement_type,
                ),
            ))
    
    def set_snapshot(self):
        self.x_snapshot = self.x.copy()
        self.P_snapshot = self.P.copy()

    def rollback(self, corrections: List[CorrectionRecord]) -> np.ndarray:
        if not corrections:
            return np.empty((0, len(self.x)))

        # --- Restore snapshot ---
        x = self.x_snapshot.copy()
        P = self.P_snapshot.copy()

        # --- Build unified timeline ---
        # Tag each event with type so we can process in order
        events: List[RollbackInfo] = []

        for snap in self.predictions:
            events.append(RollbackInfo(type='prediction', timestamp=snap.timestamp, data=snap))

        for corr in self.corrections:
            events.append(RollbackInfo(type='correction', timestamp=corr.timestamp, data=corr))

        for corr in corrections:
            events.append(RollbackInfo(type='correction', timestamp=corr.timestamp, data=corr))

        # Sort by timestamp — stable sort preserves insertion order
        # for simultaneous events (IMU before correction if same time)
        events.sort(key=lambda e: (e.timestamp, 0 if e.type == 'prediction' else 1))

        intermediate_estimates = []
        timestamp = None
        for event in events:
            if event.type == 'prediction':
                snap = event.data

                # --- Time update ---
                acc, gyro = snap.u[:3], snap.u[3:]
                x, acc_unbiased, gyro_unbiased = self._propagate_state(x, acc, gyro, snap.dt)

                F = F_numpy(x, np.concatenate([acc_unbiased, gyro_unbiased]), snap.dt)
                G = G_numpy(x, np.concatenate([acc_unbiased, gyro_unbiased]), snap.dt)
                P = F @ P @ F.T + G @ self.Q @ G.T

            elif event.type == 'correction':
                corr = event.data
                b = corr.block
                if not b.enabled:
                    continue

                z = np.asarray(b.z, dtype=float).reshape(-1)
                R = np.asarray(b.R, dtype=float)
                H = np.asarray(b.H(x.tolist()), dtype=float)
                if b.h is None:
                    zhat = H @ x
                else:
                    zhat = np.asarray(b.h(x), dtype=float).reshape(-1)

                if H.ndim != 2 or H.shape[1] != x.shape[0]:
                    raise ValueError(f"H shape mismatch: got {H.shape}, expected (*, {x.shape[0]})")
                if z.shape[0] != H.shape[0]:
                    raise ValueError(f"z/H mismatch: z={z.shape[0]}, H rows={H.shape[0]}")
                if zhat.shape[0] != z.shape[0]:
                    raise ValueError(f"zhat/z mismatch: zhat={zhat.shape[0]}, z={z.shape[0]}")
                if R.shape != (z.shape[0], z.shape[0]):
                    raise ValueError(f"R shape mismatch: got {R.shape}, expected {(z.shape[0], z.shape[0])}")

                y = b.residual(z, zhat, x)
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                if b.mask is not None:
                    K = K * np.asarray(b.mask, dtype=float).reshape(-1, 1)
                dx = K @ y

                if b.inject is None:
                    x = x + dx
                    x[6:10] /= np.linalg.norm(x[6:10])
                else:
                    x = b.inject(x, dx)

                I = np.eye(len(x))
                IKH = I - K @ H
                P = IKH @ P @ IKH.T + K @ R @ K.T


            if timestamp is None or np.abs(event.timestamp - timestamp) > 0.05:
                intermediate_estimates.append(x.copy())
                timestamp = event.timestamp

        # --- Update current state ---
        self.x = x.copy()
        self.P = P.copy()
        self.predictions.clear()
        self.corrections.clear()
        self.set_snapshot()

        return (
            np.array(intermediate_estimates)
            if intermediate_estimates
            else np.empty((0, len(self.x)))
        )