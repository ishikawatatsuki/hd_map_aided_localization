import numpy as np

class StationaryDetector:
    def __init__(self, acc_threshold=0.6, gyro_threshold=0.05, min_stationary_samples=10):
        self.acc_threshold = acc_threshold
        self.gyro_threshold = gyro_threshold
        self.min_stationary_samples = min_stationary_samples

        self.acc_bias = np.zeros(3)
        self.gyr_bias = np.zeros(3)
        self.g_ref = 9.81
        self.t_prev = None
        self._consec = 0

    def is_stationary(self, acc_f, gyr_f, ts, speed_mps=None):
        """
        acc_f, gyr_f: already low-pass filtered externally — no internal LPF needed.
        """
        if self.t_prev is None:
            self.t_prev = ts
            return False

        gyro_norm = np.linalg.norm(gyr_f)
        acc_dev   = abs(np.linalg.norm(acc_f) - self.g_ref)

        raw_cond = (gyro_norm < self.gyro_threshold) and (acc_dev < self.acc_threshold)
        if speed_mps is not None:
            raw_cond = raw_cond and (speed_mps < 0.5)

        self._consec = self._consec + 1 if raw_cond else 0
        cond = self._consec >= self.min_stationary_samples

        if cond:
            beta = 0.005
            self.acc_bias = (1 - beta) * self.acc_bias + beta * acc_f
            self.gyr_bias = (1 - beta) * self.gyr_bias + beta * gyr_f

        self.t_prev = ts
        return cond
