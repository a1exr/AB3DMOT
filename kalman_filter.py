import numpy as np


class KalmanFilter:
    def __init__(self, noise):
        self.noise = noise

        self.mu_t = np.zeros(4)
        self.sigma_t = self.noise['k'] * np.diag(
            [self.noise['sigma_x_0'] ** 2, self.noise['sigma_y_0'] ** 2, self.noise['sigma_vx_0'] ** 2, self.noise['sigma_vy_0'] ** 2])
        self.mu_t_predict = None
        self.sigma_t_predict = None

        self.u_t = np.zeros(4)
        self.z_t = np.zeros(4)
        self.A_t = np.eye(4)
        self.B_t = np.zeros([4, 4])
        self.R_t = np.zeros([4, 4])
        self.C_t = np.zeros([4, 4])
        self.C_t[0, 0] = 1
        self.C_t[1, 1] = 1
        self.Q_t = np.diag([noise['sigma_x_0'] ** 2, noise['sigma_y_0'] ** 2, noise['sigma_vx_0'] ** 2, noise['sigma_vy_0'] ** 2])

    def update_matrices(self, cur_car_coord, cur_dt):
        self.z_t[:2] = cur_car_coord[:2]

        self.A_t[0, 2] = cur_dt
        self.A_t[1, 3] = cur_dt

        self.R_t[2, 2] = cur_dt * self.noise['sigma_n'] ** 2
        self.R_t[3, 3] = cur_dt * self.noise['sigma_n'] ** 2

    def filter_step(self):
        # ----- Prediction ------
        self.mu_t_predict = self.A_t @ self.mu_t + self.B_t @ self.u_t
        self.sigma_t_predict = self.A_t @ self.sigma_t @ self.A_t.T + self.R_t

        # ----- Kalman Gain -----
        K_t = self.sigma_t_predict @ self.C_t.T @ np.linalg.inv(self.C_t @ self.sigma_t_predict @ self.C_t.T + self.Q_t)

        # ----- Correction ------
        self.mu_t = self.mu_t_predict + K_t @ (self.z_t - self.C_t @ self.mu_t_predict)
        self.sigma_t = self.sigma_t_predict - K_t @ self.C_t @ self.sigma_t_predict

        return self.mu_t, self.sigma_t