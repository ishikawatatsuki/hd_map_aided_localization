import numpy as np
import sympy as sp
from sympy import symbols, Matrix, eye, sqrt, diag

# -----------------------
# State and input symbols
# -----------------------

# state vector (16)
px, py, pz = symbols('p_x p_y p_z')
vx, vy, vz = symbols('v_x v_y v_z')
qw, qx, qy, qz = symbols('q_w q_x q_y q_z')
b_ax, b_ay, b_az = symbols('b_a_x b_a_y b_a_z')
b_wx, b_wy, b_wz = symbols('b_w_x b_w_y b_w_z')

# control input (measured IMU)
ax, ay, az = symbols('a_x a_y a_z')
wx, wy, wz = symbols('w_x w_y w_z')

# delta time
dt = symbols('dt', positive=True)

def R_from_quat(qw, qx, qy, qz):
    # Quaternion is scalar-first [qw, qx, qy, qz].
    # This is the standard DCM mapping body->world for a unit quaternion.
    return Matrix([
        [qw**2 + qx**2 - qy**2 - qz**2, 2*(qx*qy - qw*qz),           2*(qw*qy + qx*qz)],
        [2*(qx*qy + qw*qz),           qw**2 - qx**2 + qy**2 - qz**2, 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),           2*(qw*qx + qy*qz),           qw**2 - qx**2 - qy**2 + qz**2],
    ])

def Omega_from_omega(wx, wy, wz):
    # For qdot = 0.5 * Omega(omega) * q, with q=[qw,qx,qy,qz].
    return Matrix([
        [0,   -wx, -wy, -wz],
        [wx,   0,   wz, -wy],
        [wy,  -wz,  0,   wx],
        [wz,   wy, -wx,  0 ],
    ])

# Nominal covariance
P = eye(16) * 0.1

# Process noise (12): accel noise, gyro noise, accel bias RW, gyro bias RW
n_ax, n_ay, n_az = symbols('n_a_x n_a_y n_a_z')
n_wx, n_wy, n_wz = symbols('n_w_x n_w_y n_w_z')
n_bax, n_bay, n_baz = symbols('n_ba_x n_ba_y n_ba_z')
n_bwx, n_bwy, n_bwz = symbols('n_bw_x n_bw_y n_bw_z')

sigma_a, sigma_w, sigma_ba, sigma_bw = symbols('sigma_a sigma_w sigma_ba sigma_bw', positive=True)
Q = diag(*([sigma_a**2]*3 + [sigma_w**2]*3 + [sigma_ba**2]*3 + [sigma_bw**2]*3))


# -----------------------
# State, input, noise vectors
# -----------------------
p = Matrix([px, py, pz])
v = Matrix([vx, vy, vz])
q = Matrix([qw, qx, qy, qz])
b_a = Matrix([b_ax, b_ay, b_az])
b_w = Matrix([b_wx, b_wy, b_wz])

state = Matrix([px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz])
control_input = Matrix([ax, ay, az, wx, wy, wz])  # measured IMU: accel + gyro

process_noise = Matrix([
    n_ax, n_ay, n_az,  # accel measurement noise
    n_wx, n_wy, n_wz,  # gyro measurement noise
    n_bax, n_bay, n_baz,  # accel bias random walk
    n_bwx, n_bwy, n_bwz,  # gyro bias random walk
])

# -----------------------
# Propagation ingredients
# -----------------------
R_q = R_from_quat(qw, qx, qy, qz)

# Unbiased / noisy IMU signals used by the propagator
a_m = Matrix([ax, ay, az])
w_m = Matrix([wx, wy, wz])

a_used = (a_m + Matrix([n_ax, n_ay, n_az])) - b_a
w_used = (w_m + Matrix([n_wx, n_wy, n_wz])) - b_w

g = Matrix([0, 0, -9.81])

# Helpful: the world-frame linear acceleration
a_world = R_q * a_used + g

# Kinematics discretization (constant acceleration over dt)
p_new = p + v*dt + sp.Rational(1, 2) * a_world * dt**2
v_new = v + a_world * dt

# Quaternion propagation (first-order).
# IMPORTANT: quaternion must be re-normalized numerically after propagation in code.
Omega_used = Omega_from_omega(w_used[0], w_used[1], w_used[2])
q_new = (eye(4) + sp.Rational(1, 2) * dt * Omega_used) * q

# Bias random walks
b_a_new = b_a + Matrix([n_bax, n_bay, n_baz]) * dt
b_w_new = b_w + Matrix([n_bwx, n_bwy, n_bwz]) * dt

fx = Matrix.vstack(p_new, v_new, q_new, b_a_new, b_w_new)
F_jacobian = fx.jacobian(state)
G_jacobian = fx.jacobian(process_noise)

# Evaluate Jacobians at zero process noise (mean propagation)
zero_noise = {
    n_ax: 0.0, n_ay: 0.0, n_az: 0.0,
    n_wx: 0.0, n_wy: 0.0, n_wz: 0.0,
    n_bax: 0.0, n_bay: 0.0, n_baz: 0.0,
    n_bwx: 0.0, n_bwy: 0.0, n_bwz: 0.0,
}

F_expr = sp.Matrix(F_jacobian).subs(zero_noise)
G_expr = sp.Matrix(G_jacobian).subs(zero_noise)

# Argument order: state (16), control (6), dt (1)
_args = [
    px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz,
    ax, ay, az, wx, wy, wz,
    dt,
]

_F_func = sp.lambdify(_args, F_expr, modules="numpy")
_G_func = sp.lambdify(_args, G_expr, modules="numpy")


_args_pos_heading = [
    px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz
]
hx_pos_heading = Matrix([px, py, sp.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))])  # [x, y, heading]
H_pos_heading = hx_pos_heading.jacobian(state)
_H_pos_heading_func = sp.lambdify(_args_pos_heading, H_pos_heading, modules="numpy")

velocity_body = R_q.T * v
hx_lu_vel_body = Matrix([velocity_body[1], velocity_body[2]]) # forward, left, up
H_lu_vel_body = hx_lu_vel_body.jacobian(state)
_H_lu_vel_body_func = sp.lambdify(_args_pos_heading, H_lu_vel_body, modules="numpy")


hx_vel_body = Matrix([velocity_body[0], velocity_body[1], velocity_body[2]])
H_vel_body = hx_vel_body.jacobian(state)
_H_vel_body_func = sp.lambdify(_args_pos_heading, H_vel_body, modules="numpy")

H_yaw = Matrix([sp.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))])  # heading only
H_yaw_jacobian = H_yaw.jacobian(state)
_H_yaw_func = sp.lambdify(_args_pos_heading, H_yaw_jacobian, modules="numpy")


def F_numpy(x_state, u_imu, dt_val):
    """Return F = d f / d x as a NumPy (16x16) array.

    x_state: shape (16,) = [p(3), v(3), q(4 scalar-first), b_a(3), b_w(3)]
    u_imu:   shape (6,)  = [a_x, a_y, a_z, w_x, w_y, w_z]
    """
    x_state = np.asarray(x_state, dtype=float).reshape(-1)
    u_imu = np.asarray(u_imu, dtype=float).reshape(-1)
    if x_state.size != 16:
        raise ValueError(f"x_state must have 16 elements, got {x_state.size}")
    if u_imu.size != 6:
        raise ValueError(f"u_imu must have 6 elements, got {u_imu.size}")
    vals = list(x_state) + list(u_imu) + [float(dt_val)]
    return np.array(_F_func(*vals), dtype=float)

def G_numpy(x_state, u_imu, dt_val):
    """Return G = d f / d w as a NumPy (16x12) array.

    Process-noise ordering matches `process_noise` above:
    [n_a(3), n_w(3), n_ba(3), n_bw(3)].
    """
    x_state = np.asarray(x_state, dtype=float).reshape(-1)
    u_imu = np.asarray(u_imu, dtype=float).reshape(-1)
    if x_state.size != 16:
        raise ValueError(f"x_state must have 16 elements, got {x_state.size}")
    if u_imu.size != 6:
        raise ValueError(f"u_imu must have 6 elements, got {u_imu.size}")
    vals = list(x_state) + list(u_imu) + [float(dt_val)]
    return np.array(_G_func(*vals), dtype=float)

def H_pos_heading_numpy(x_state: list) -> np.ndarray:
    return np.array(_H_pos_heading_func(*x_state), dtype=float)

def H_yaw_numpy(x_state: list) -> np.ndarray:
    return np.array(_H_yaw_func(*x_state), dtype=float)

def H_quat_numpy(x_state: list) -> np.ndarray:
    # Jacobian of quaternion measurement w.r.t. state.
    H = np.zeros((4, 16), dtype=float)
    H[:, 6:10] = np.eye(4)
    return H

# position measured in inertial frame
def H_pos_numpy(x_state: list) -> np.ndarray:
    H = np.zeros((3, 16), dtype=float)
    H[:, 0:3] = np.eye(3)
    return H

# Velocity measured in inertial frame
def H_vel_numpy(x_state: list) -> np.ndarray:
    H = np.zeros((3, 16), dtype=float)
    H[:, 3:6] = np.eye(3)
    return H


# Velocity measured in body frame, and rotated by current orientation estimate
def H_lu_vel_body_numpy(x_state: list) -> np.ndarray:
    return np.array(_H_lu_vel_body_func(*x_state), dtype=float)

def H_vel_body_numpy(x_state: list) -> np.ndarray:
    return np.array(_H_vel_body_func(*x_state), dtype=float)