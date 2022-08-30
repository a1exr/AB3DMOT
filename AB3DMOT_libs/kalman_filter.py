import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

class Filter(object):
	def __init__(self, bbox3D, info, ID):

		self.initial_pos = bbox3D
		self.time_since_update = 0
		self.id = ID
		self.hits = 1           		# number of total hits including the first detection
		self.info = info        		# other information associated	

# class KF(Filter):
# 	def __init__(self, bbox3D, info, ID):
# 		super().__init__(bbox3D, info, ID)

# 		self.kf = KalmanFilter(dim_x=10, dim_z=7)       
# 		# There is no need to use EKF here as the measurement and state are in the same space with linear relationship

# 		# state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
# 		# constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz 
# 		# while all others (theta, l, w, h, dx, dy, dz) remain the same
# 		self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix, dim_x * dim_x
# 		                      [0,1,0,0,0,0,0,0,1,0],
# 		                      [0,0,1,0,0,0,0,0,0,1],
# 		                      [0,0,0,1,0,0,0,0,0,0],  
# 		                      [0,0,0,0,1,0,0,0,0,0],
# 		                      [0,0,0,0,0,1,0,0,0,0],
# 		                      [0,0,0,0,0,0,1,0,0,0],
# 		                      [0,0,0,0,0,0,0,1,0,0],
# 		                      [0,0,0,0,0,0,0,0,1,0],
# 		                      [0,0,0,0,0,0,0,0,0,1]])     

# 		# measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
# 		self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
# 		                      [0,1,0,0,0,0,0,0,0,0],
# 		                      [0,0,1,0,0,0,0,0,0,0],
# 		                      [0,0,0,1,0,0,0,0,0,0],
# 		                      [0,0,0,0,1,0,0,0,0,0],
# 		                      [0,0,0,0,0,1,0,0,0,0],
# 		                      [0,0,0,0,0,0,1,0,0,0]])

# 		# measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
# 		# self.kf.R[0:,0:] *= 10.   

# 		# initial state uncertainty at time 0
# 		# Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
# 		self.kf.P[7:, 7:] *= 1000. 	
# 		self.kf.P *= 10.

# 		# process uncertainty, make the constant velocity part more certain
# 		self.kf.Q[7:, 7:] *= 0.01

# 		# initialize data
# 		self.kf.x[:7] = self.initial_pos.reshape((7, 1))

# 	def compute_innovation_matrix(self):
# 		""" compute the innovation matrix for association with mahalanobis distance
# 		"""
# 		return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

# 	def get_velocity(self):
# 		# return the object velocity in the state

# 		return self.kf.x[7:]


class KF(Filter):	# 2D BEV: xz
	# 		z
	#		|
	#       |_____ x
	#      /
	#    y/
	def __init__(self, bbox3D, info, ID):
		super().__init__(bbox3D, info, ID)

		self.kf = KalmanFilter(dim_x=10, dim_z=5)       
		# There is no need to use EKF here as the measurement and state are in the same space with linear relationship

		# state x dimension 10: x, z, theta, l, w, dx, dz, dtheta, dl, dw
		# constant velocity model: x' = x + dx, z' = z + dz, theta' = theta + dtheta, l' = l + dl, w' = w + dw 
		# while all others (dx, dy, dtheta, dl, dw) remain the same
		# state transition matrix, dim_x * dim_x
		self.kf.F = np.array([[1,0,0,0,0,1,0,0,0,0],    # x' = x + dx
		                      [0,1,0,0,0,0,1,0,0,0],    # z' = z + dz
		                      [0,0,1,0,0,0,0,1,0,0],  	# theta' = theta + dtheta
		                      [0,0,0,1,0,0,0,0,1,0],	# l' = l + dl
		                      [0,0,0,0,1,0,0,0,0,1],	# w' = w + dw
		                      [0,0,0,0,0,1,0,0,0,0],	# dx
		                      [0,0,0,0,0,0,1,0,0,0],	# dz
		                      [0,0,0,0,0,0,0,1,0,0],	# dtheta
		                      [0,0,0,0,0,0,0,0,1,0],	# dl
		                      [0,0,0,0,0,0,0,0,0,1]])	# dw     

		# measurement function, dim_z * dim_x, the first 5 dimensions of the measurement correspond to the state
		self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
		                      [0,1,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0]])

		# measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
		# self.kf.R[0:,0:] *= 10.   

		# initial state uncertainty at time 0
		# Given a single data, the initial velocity is very uncertain, so give a high uncertainty to start
		self.kf.P = np.array([[6,0,0,0,0,0,0,0,0,0],  	# x
		                      [0,10,0,0,0,0,0,0,0,0],   # z
		                      [0,0,.25,0,0,0,0,0,0,0],  # theta
		                      [0,0,0,1,0,0,0,0,0,0],	# l
		                      [0,0,0,0,1,0,0,0,0,0],	# w
		                      [0,0,0,0,0,3,0,0,0,0],	# dx
		                      [0,0,0,0,0,0,5,0,0,0],	# dz
		                      [0,0,0,0,0,0,0,.1,0,0],	# dtheta
		                      [0,0,0,0,0,0,0,0,.5,0],	# dl
		                      [0,0,0,0,0,0,0,0,0,.5]])	# dw     
		# self.kf.P[5:, 5:] *= 1000. 	
		# self.kf.P *= 10.

		# process uncertainty, make the constant velocity part more certain
		self.kf.Q[5:, 5:] *= 0.01
		# self.kf.Q = self.kf.P.copy()
		# self.kf.Q[:5, :5] = 0
		# self.kf.Q[5:, 5:] *= 0.5

		# initialize data
		x_2d = np.concatenate((self.initial_pos[:1], self.initial_pos[2:6]), axis=0)	# x, z, theta, l, w
		self.kf.x[:5] = x_2d.reshape((5, 1))

	def compute_innovation_matrix(self):
		""" compute the innovation matrix for association with mahalanobis distance
		"""
		return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

	def get_velocity(self):
		# return the object velocity in the state

		return self.kf.x[7:]