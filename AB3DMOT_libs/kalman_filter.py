import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

class Filter(object):
	def __init__(self, bbox3D, info, ID, cat, kf_coeffs):

		self.initial_pos = bbox3D
		self.time_since_update = 0
		self.id = ID
		self.cat = cat
		self.hits = 1           		# number of total hits including the first detection
		self.info = info        		# other information associated	
		self.kf_coeffs = kf_coeffs


class KF(Filter):	# 2D BEV: xz
	# 		z
	#		|
	#       |_____ x
	#      /
	#    y/
	def __init__(self, bbox3D, info, ID, cat, kf_coeffs):
		super().__init__(bbox3D, info, ID, cat, kf_coeffs)

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

		# self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
		# 						[0,1.8,0,0,0],  	# z
		# 						[0,0,0.05,0,0], 	# theta
		# 						[0,0,0,0.2,0],	# l
		# 						[0,0,0,0,0.05]]) 	# w

		# # initial state uncertainty at time 0
		# self.kf.P = np.array([[6,0,0,0,0,0,0,0,0,0],  	# x
		# 						[0,10,0,0,0,0,0,0,0,0],   # z
		# 						[0,0,.2,0,0,0,0,0,0,0],  # theta
		# 						[0,0,0,1,0,0,0,0,0,0],	# l
		# 						[0,0,0,0,.5,0,0,0,0,0],	# w
		# 						[0,0,0,0,0,3,0,0,0,0],	# dx
		# 						[0,0,0,0,0,0,5,0,0,0],	# dz
		# 						[0,0,0,0,0,0,0,.1,0,0],	# dtheta
		# 						[0,0,0,0,0,0,0,0,.5,0],	# dl
		# 						[0,0,0,0,0,0,0,0,0,.25]])	# dw

		# # process uncertainty, make the constant velocity part more certain
		# self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
		# 						[0,0,0,0,0,0,0,0,0,0],    # z
		# 						[0,0,0,0,0,0,0,0,0,0],  	# theta
		# 						[0,0,0,0,0,0,0,0,0,0],	# l
		# 						[0,0,0,0,0,0,0,0,0,0],	# w
		# 						[0,0,0,0,0,.25,0,0,0,0],	# dx
		# 						[0,0,0,0,0,0,.25,0,0,0],	# dz
		# 						[0,0,0,0,0,0,0,.1,0,0],	# dtheta
		# 						[0,0,0,0,0,0,0,0,.25,0],	# dl
		# 						[0,0,0,0,0,0,0,0,0,.25]])	# dw  

# ----------------------------------------------------------------------
		# self.kf.F = np.array([[1,0,0,0,0,0,0,0,0,0],    # x' = x
		#                       [0,1,0,0,0,0,0,0,0,0],    # z' = z
		#                       [0,0,1,0,0,0,0,0,0,0],  	# theta' = theta
		#                       [0,0,0,1,0,0,0,0,0,0],	# l' = l
		#                       [0,0,0,0,1,0,0,0,0,0],	# w' = w
		#                       [0,0,0,0,0,1,0,0,0,0],	# dx
		#                       [0,0,0,0,0,0,1,0,0,0],	# dz
		#                       [0,0,0,0,0,0,0,1,0,0],	# dtheta
		#                       [0,0,0,0,0,0,0,0,1,0],	# dl
		#                       [0,0,0,0,0,0,0,0,0,1]])	# dw     

		# # measurement function, dim_z * dim_x, the first 5 dimensions of the measurement correspond to the state
		# self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
		#                       [0,1,0,0,0,0,0,0,0,0],
		#                       [0,0,1,0,0,0,0,0,0,0],
		#                       [0,0,0,1,0,0,0,0,0,0],
		#                       [0,0,0,0,1,0,0,0,0,0]])

		# self.kf.R = np.array([[1e6,0,0,0,0],  	# x  
		# 						[0,1e6,0,0,0],  	# z
		# 						[0,0,1e6,0,0], 	# theta
		# 						[0,0,0,1e6,0],	# l
		# 						[0,0,0,0,1e6]]) 	# w

		# # initial state uncertainty at time 0
		# self.kf.P = np.array([[0,0,0,0,0,0,0,0,0,0],  	# x
		# 						[0,0,0,0,0,0,0,0,0,0],   # z
		# 						[0,0,0,0,0,0,0,0,0,0],  # theta
		# 						[0,0,0,0,0,0,0,0,0,0],	# l
		# 						[0,0,0,0,0,0,0,0,0,0],	# w
		# 						[0,0,0,0,0,0,0,0,0,0],	# dx
		# 						[0,0,0,0,0,0,0,0,0,0],	# dz
		# 						[0,0,0,0,0,0,0,0,0,0],	# dtheta
		# 						[0,0,0,0,0,0,0,0,0,0],	# dl
		# 						[0,0,0,0,0,0,0,0,0,0]])	# dw

		# # process uncertainty, make the constant velocity part more certain
		# self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
		# 						[0,0,0,0,0,0,0,0,0,0],    # z
		# 						[0,0,0,0,0,0,0,0,0,0],  	# theta
		# 						[0,0,0,0,0,0,0,0,0,0],	# l
		# 						[0,0,0,0,0,0,0,0,0,0],	# w
		# 						[0,0,0,0,0,0,0,0,0,0],	# dx
		# 						[0,0,0,0,0,0,0,0,0,0],	# dz
		# 						[0,0,0,0,0,0,0,0,0,0],	# dtheta
		# 						[0,0,0,0,0,0,0,0,0,0],	# dl
		# 						[0,0,0,0,0,0,0,0,0,0]])	# dw  
# ----------------------------------------------------------------------

		if kf_coeffs is not None:
			# measurement uncertainty
			self.kf.R = np.array([[kf_coeffs['Rx'],0,0,0,0],  	# x  
								  [0,kf_coeffs['Rz'],0,0,0],  	# z
								  [0,0,kf_coeffs['Rtheta'],0,0],	# theta
								  [0,0,0,kf_coeffs['Rl'],0],	# l
								  [0,0,0,0,kf_coeffs['Rw']]]) 	# w

			# initial state uncertainty at time 0
			self.kf.P = np.array([[kf_coeffs['Px'],0,0,0,0,0,0,0,0,0],  	# x
								  [0,kf_coeffs['Pz'],0,0,0,0,0,0,0,0],   # z
								  [0,0,kf_coeffs['Ptheta'],0,0,0,0,0,0,0],  # theta
								  [0,0,0,kf_coeffs['Pl'],0,0,0,0,0,0],	# l
								  [0,0,0,0,kf_coeffs['Pw'],0,0,0,0,0],	# w
								  [0,0,0,0,0,kf_coeffs['Pdx'],0,0,0,0],	# dx
								  [0,0,0,0,0,0,kf_coeffs['Pdz'],0,0,0],	# dz
								  [0,0,0,0,0,0,0,kf_coeffs['Pdtheta'],0,0],	# dtheta
								  [0,0,0,0,0,0,0,0,kf_coeffs['Pdl'],0],	# dl
								  [0,0,0,0,0,0,0,0,0,kf_coeffs['Pdw']]])	# dw

			# process uncertainty, make the constant velocity part more certain
			self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
								  [0,0,0,0,0,0,0,0,0,0],    # z
								  [0,0,0,0,0,0,0,0,0,0],  	# theta
								  [0,0,0,0,0,0,0,0,0,0],	# l
								  [0,0,0,0,0,0,0,0,0,0],	# w
								  [0,0,0,0,0,kf_coeffs['Qdx'],0,0,0,0],	# dx
								  [0,0,0,0,0,0,kf_coeffs['Qdz'],0,0,0],	# dz
								  [0,0,0,0,0,0,0,kf_coeffs['Qdtheta'],0,0],	# dtheta
								  [0,0,0,0,0,0,0,0,kf_coeffs['Qdl'],0],	# dl
								  [0,0,0,0,0,0,0,0,0,kf_coeffs['Qdw']]])	# dw    

		else:
			if cat == 'Car':
				# measurement uncertainty
				self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
									[0,1,0,0,0],  	# z
									[0,0,0.0125,0,0],	# theta
									[0,0,0,0.16,0],	# l
									[0,0,0,0,0.05]]) 	# w
				
				# # measurement uncertainty
				# self.kf.R = np.array([[100,0,0,0,0],  	# x  
				# 					[0,100,0,0,0],  	# z
				# 					[0,0,100,0,0],	# theta
				# 					[0,0,0,100,0],	# l
				# 					[0,0,0,0,100]]) 	# w

				# initial state uncertainty at time 0
				self.kf.P = np.array([[6,0,0,0,0,0,0,0,0,0],  	# x
									[0,10,0,0,0,0,0,0,0,0],   # z
									[0,0,.2,0,0,0,0,0,0,0],  # theta
									[0,0,0,1,0,0,0,0,0,0],	# l
									[0,0,0,0,.5,0,0,0,0,0],	# w
									[0,0,0,0,0,3,0,0,0,0],	# dx
									[0,0,0,0,0,0,5,0,0,0],	# dz
									[0,0,0,0,0,0,0,.1,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,.5,0],	# dl
									[0,0,0,0,0,0,0,0,0,.25]])	# dw

				# process uncertainty, make the constant velocity part more certain
				self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
									[0,0,0,0,0,0,0,0,0,0],    # z
									[0,0,0,0,0,0,0,0,0,0],  	# theta
									[0,0,0,0,0,0,0,0,0,0],	# l
									[0,0,0,0,0,0,0,0,0,0],	# w
									[0,0,0,0,0,.25,0,0,0,0],	# dx
									[0,0,0,0,0,0,.25,0,0,0],	# dz
									[0,0,0,0,0,0,0,.01,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,.25,0],	# dl
									[0,0,0,0,0,0,0,0,0,.25]])	# dw         
				
				# # process uncertainty, make the constant velocity part more certain
				# self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
				# 					[0,0,0,0,0,0,0,0,0,0],    # z
				# 					[0,0,0,0,0,0,0,0,0,0],  	# theta
				# 					[0,0,0,0,0,0,0,0,0,0],	# l
				# 					[0,0,0,0,0,0,0,0,0,0],	# w
				# 					[0,0,0,0,0,0,0,0,0,0],	# dx
				# 					[0,0,0,0,0,0,0,0,0,0],	# dz
				# 					[0,0,0,0,0,0,0,0,0,0],	# dtheta
				# 					[0,0,0,0,0,0,0,0,0,0],	# dl
				# 					[0,0,0,0,0,0,0,0,0,0]])	# dw        

			elif cat == 'Pedestrian':
				# measurement uncertainty
				self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
									[0,1,0,0,0],  	# z
									[0,0,0.25,0,0], 	# theta
									[0,0,0,0.1,0],	# l
									[0,0,0,0,0.05]]) 	# w

				# initial state uncertainty at time 0
				self.kf.P = np.array([[5,0,0,0,0,0,0,0,0,0],  	# x
									[0,10,0,0,0,0,0,0,0,0],   # z
									[0,0,.15,0,0,0,0,0,0,0],  # theta
									[0,0,0,.5,0,0,0,0,0,0],	# l
									[0,0,0,0,.5,0,0,0,0,0],	# w
									[0,0,0,0,0,2,0,0,0,0],	# dx
									[0,0,0,0,0,0,3,0,0,0],	# dz
									[0,0,0,0,0,0,0,.05,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,.25,0],	# dl
									[0,0,0,0,0,0,0,0,0,.25]])	# dw     
			
				# process uncertainty, make the constant velocity part more certain
				self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
									[0,0,0,0,0,0,0,0,0,0],    # z
									[0,0,0,0,0,0,0,0,0,0],  	# theta
									[0,0,0,0,0,0,0,0,0,0],	# l
									[0,0,0,0,0,0,0,0,0,0],	# w
									[0,0,0,0,0,0.3,0,0,0,0],	# dx
									[0,0,0,0,0,0,0.3,0,0,0],	# dz
									[0,0,0,0,0,0,0,0.1,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,0.2,0],	# dl
									[0,0,0,0,0,0,0,0,0,0.2]])	# dw    

			elif cat == 'Bicycle':
				# measurement uncertainty
				self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
									[0,1,0,0,0],  	# z
									[0,0,0.05,0,0], 	# theta
									[0,0,0,0.15,0],	# l
									[0,0,0,0,0.05]]) 	# w

				# initial state uncertainty at time 0
				self.kf.P = np.array([[5,0,0,0,0,0,0,0,0,0],  	# x
									[0,7,0,0,0,0,0,0,0,0],   	# z
									[0,0,.1,0,0,0,0,0,0,0],  	# theta
									[0,0,0,1,0,0,0,0,0,0],	# l
									[0,0,0,0,.5,0,0,0,0,0],	# w
									[0,0,0,0,0,2.5,0,0,0,0],	# dx
									[0,0,0,0,0,0,3.5,0,0,0],	# dz
									[0,0,0,0,0,0,0,.05,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,.5,0],	# dl
									[0,0,0,0,0,0,0,0,0,.25]])	# dw     
			
				# process uncertainty, make the constant velocity part more certain
				self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
									[0,0,0,0,0,0,0,0,0,0],    # z
									[0,0,0,0,0,0,0,0,0,0],  	# theta
									[0,0,0,0,0,0,0,0,0,0],	# l
									[0,0,0,0,0,0,0,0,0,0],	# w
									[0,0,0,0,0,0.25,0,0,0,0],	# dx
									[0,0,0,0,0,0,0.25,0,0,0],	# dz
									[0,0,0,0,0,0,0,0.1,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,0.25,0],	# dl
									[0,0,0,0,0,0,0,0,0,0.25]])# dw    

			elif cat == 'Motorcycle':
				# measurement uncertainty
				self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
									[0,1.2,0,0,0],  	# z
									[0,0,0.015,0,0], 	# theta
									[0,0,0,0.15,0],	# l
									[0,0,0,0,0.035]])	# w

				# initial state uncertainty at time 0
				self.kf.P = np.array([[5,0,0,0,0,0,0,0,0,0],  	# x
									[0,7,0,0,0,0,0,0,0,0],   	# z
									[0,0,.1,0,0,0,0,0,0,0],  	# theta
									[0,0,0,1,0,0,0,0,0,0],	# l
									[0,0,0,0,.5,0,0,0,0,0],	# w
									[0,0,0,0,0,2.5,0,0,0,0],	# dx
									[0,0,0,0,0,0,3.5,0,0,0],	# dz
									[0,0,0,0,0,0,0,.05,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,.5,0],	# dl
									[0,0,0,0,0,0,0,0,0,.25]])	# dw    
			
				# process uncertainty, make the constant velocity part more certain
				self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
									[0,0,0,0,0,0,0,0,0,0],    # z
									[0,0,0,0,0,0,0,0,0,0],  	# theta
									[0,0,0,0,0,0,0,0,0,0],	# l
									[0,0,0,0,0,0,0,0,0,0],	# w
									[0,0,0,0,0,0.25,0,0,0,0],	# dx
									[0,0,0,0,0,0,0.25,0,0,0],	# dz
									[0,0,0,0,0,0,0,0.05,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,0.25,0],	# dl
									[0,0,0,0,0,0,0,0,0,0.25]])# dw    

			elif cat == 'Bus' or cat == 'Trailer' or cat == 'Truck':
				# measurement uncertainty
				self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
									[0,1.5,0,0,0],  	# z
									[0,0,0.015,0,0], 	# theta
									[0,0,0,1.5,0],	# l
									[0,0,0,0,0.1]]) 	# w

				# initial state uncertainty at time 0
				self.kf.P = np.array([[10,0,0,0,0,0,0,0,0,0],  	# x
									[0,15,0,0,0,0,0,0,0,0],   # z
									[0,0,.2,0,0,0,0,0,0,0],  # theta
									[0,0,0,2,0,0,0,0,0,0],	# l
									[0,0,0,0,1,0,0,0,0,0],	# w
									[0,0,0,0,0,3,0,0,0,0],	# dx
									[0,0,0,0,0,0,5,0,0,0],	# dz
									[0,0,0,0,0,0,0,.05,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,2,0],	# dl
									[0,0,0,0,0,0,0,0,0,1]])	# dw     

				# process uncertainty, make the constant velocity part more certain
				self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
									[0,0,0,0,0,0,0,0,0,0],    # z
									[0,0,0,0,0,0,0,0,0,0],  	# theta
									[0,0,0,0,0,0,0,0,0,0],	# l
									[0,0,0,0,0,0,0,0,0,0],	# w
									[0,0,0,0,0,0.25,0,0,0,0],	# dx
									[0,0,0,0,0,0,0.25,0,0,0],	# dz
									[0,0,0,0,0,0,0,0.05,0,0],	# dtheta
									[0,0,0,0,0,0,0,0,0.25,0],	# dl
									[0,0,0,0,0,0,0,0,0,0.25]])# dw    

		# initialize data
		x_2d = np.concatenate((self.initial_pos[:1], self.initial_pos[2:6]), axis=0)	# x, z, theta, l, w
		self.kf.x[:5] = x_2d.reshape((5, 1))

		self.kf.cur_R = self.kf.R.copy()

	def compute_innovation_matrix(self):
		""" compute the innovation matrix for association with mahalanobis distance
		"""
		return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

	def get_velocity(self):
		# return the object velocity in the state

		return self.kf.x[7:]