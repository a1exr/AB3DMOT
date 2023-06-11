import numpy as np
from filterpy.kalman import KalmanFilter
		
class KF(object):	# 2D BEV: xz
	# 		z
	#		|
	#       |_____ x
	#      /
	#    y/
    def __init__(self, det, dt):
        category = det['label_preds']   # TODO: use categories
        # self.noise = noise
        # def __init__(self, bbox3D, info, ID, cat, kf_coeffs):
        # 	super().__init__(bbox3D, info, ID, cat, kf_coeffs)

        self.kf = KalmanFilter(dim_x=4, dim_z=4)
        self.initialize_state_vector(det)

        # state x dimension 4: x, z, vx, vz
        # constant velocity model: x' = x + vx*dt, z' = z + vz*dt 
        # while (vx', vz') remain the same: vx' = vx, vz' = vz
        # state transition matrix, dim_x * dim_x
        self.kf.F = np.array([  [1,0,dt,0],     # x' = x + vx*dt
                                [0,1,0,dt],     # z' = z + vz*dt 
                                [0,0,1,0],	    # vx
                                [0,0,0,1]])	    # vz     

        self.kf.H = np.array([  [1,0,0,0],      
                                [0,1,0,0],
                                [0,0,1,0],
                                [0,0,0,1]])

        # measurement uncertainty
        self.kf.R = np.array([  [0.25,0,0,0],  	# x  
        						[0,1.8,0,0],  	# z
        						[0,0,0.05,0], 	# vx
        						[0,0,0,0.05]]) 	# vz

        # initial state uncertainty at time 0
        self.kf.P = np.array([  [6,0,0,0],      # x
                                [0,10,0,0],     # z
                                [0,0,3,0],	    # vx
                                [0,0,0,5]])	    # vz    

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q = np.array([  [0,0,0,0],          # x
                                [0,0,0,0],          # z
                                [0,0,dt*0.25,0],	# vx
                                [0,0,0,dt*0.25]])	# vz    

        # if kf_coeffs is not None:
        #     # measurement uncertainty
        #     self.kf.R = np.array([[kf_coeffs['Rx'],0,0,0,0],  	# x  
        #                             [0,kf_coeffs['Rz'],0,0,0],  	# z
        #                             [0,0,kf_coeffs['Rtheta'],0,0],	# theta
        #                             [0,0,0,kf_coeffs['Rl'],0],	# l
        #                             [0,0,0,0,kf_coeffs['Rw']]]) 	# w

        #     # initial state uncertainty at time 0
        #     self.kf.P = np.array([[kf_coeffs['Px'],0,0,0,0,0,0,0,0,0],  	# x
        #                             [0,kf_coeffs['Pz'],0,0,0,0,0,0,0,0],   # z
        #                             [0,0,kf_coeffs['Ptheta'],0,0,0,0,0,0,0],  # theta
        #                             [0,0,0,kf_coeffs['Pl'],0,0,0,0,0,0],	# l
        #                             [0,0,0,0,kf_coeffs['Pw'],0,0,0,0,0],	# w
        #                             [0,0,0,0,0,kf_coeffs['Pdx'],0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,kf_coeffs['Pdz'],0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,kf_coeffs['Pdtheta'],0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,kf_coeffs['Pdl'],0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,kf_coeffs['Pdw']]])	# dw

        #     # process uncertainty, make the constant velocity part more certain
        #     self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
        #                             [0,0,0,0,0,0,0,0,0,0],    # z
        #                             [0,0,0,0,0,0,0,0,0,0],  	# theta
        #                             [0,0,0,0,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,0,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,kf_coeffs['Qdx'],0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,kf_coeffs['Qdz'],0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,kf_coeffs['Qdtheta'],0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,kf_coeffs['Qdl'],0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,kf_coeffs['Qdw']]])	# dw    

        # else:
        #     if category == 'Car':
        #         # measurement uncertainty
        #         self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
        #                             [0,1,0,0,0],  	# z
        #                             [0,0,0.0125,0,0],	# theta
        #                             [0,0,0,0.16,0],	# l
        #                             [0,0,0,0,0.05]]) 	# w
                
        #         # # measurement uncertainty
        #         # self.kf.R = np.array([[100,0,0,0,0],  	# x  
        #         # 					[0,100,0,0,0],  	# z
        #         # 					[0,0,100,0,0],	# theta
        #         # 					[0,0,0,100,0],	# l
        #         # 					[0,0,0,0,100]]) 	# w

        #         # initial state uncertainty at time 0
        #         self.kf.P = np.array([[6,0,0,0,0,0,0,0,0,0],  	# x
        #                             [0,10,0,0,0,0,0,0,0,0],   # z
        #                             [0,0,.2,0,0,0,0,0,0,0],  # theta
        #                             [0,0,0,1,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,.5,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,3,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,5,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,.1,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,.5,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,.25]])	# dw

        #         # process uncertainty, make the constant velocity part more certain
        #         self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
        #                             [0,0,0,0,0,0,0,0,0,0],    # z
        #                             [0,0,0,0,0,0,0,0,0,0],  	# theta
        #                             [0,0,0,0,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,0,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,.25,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,.25,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,.01,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,.25,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,.25]])	# dw         
                
        #         # # process uncertainty, make the constant velocity part more certain
        #         # self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
        #         # 					[0,0,0,0,0,0,0,0,0,0],    # z
        #         # 					[0,0,0,0,0,0,0,0,0,0],  	# theta
        #         # 					[0,0,0,0,0,0,0,0,0,0],	# l
        #         # 					[0,0,0,0,0,0,0,0,0,0],	# w
        #         # 					[0,0,0,0,0,0,0,0,0,0],	# dx
        #         # 					[0,0,0,0,0,0,0,0,0,0],	# dz
        #         # 					[0,0,0,0,0,0,0,0,0,0],	# dtheta
        #         # 					[0,0,0,0,0,0,0,0,0,0],	# dl
        #         # 					[0,0,0,0,0,0,0,0,0,0]])	# dw        

        #     elif category == 'Pedestrian':
        #         # measurement uncertainty
        #         self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
        #                             [0,1,0,0,0],  	# z
        #                             [0,0,0.25,0,0], 	# theta
        #                             [0,0,0,0.1,0],	# l
        #                             [0,0,0,0,0.05]]) 	# w

        #         # initial state uncertainty at time 0
        #         self.kf.P = np.array([[5,0,0,0,0,0,0,0,0,0],  	# x
        #                             [0,10,0,0,0,0,0,0,0,0],   # z
        #                             [0,0,.15,0,0,0,0,0,0,0],  # theta
        #                             [0,0,0,.5,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,.5,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,2,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,3,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,.05,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,.25,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,.25]])	# dw     
            
        #         # process uncertainty, make the constant velocity part more certain
        #         self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
        #                             [0,0,0,0,0,0,0,0,0,0],    # z
        #                             [0,0,0,0,0,0,0,0,0,0],  	# theta
        #                             [0,0,0,0,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,0,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,0.3,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,0.3,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,0.1,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,0.2,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,0.2]])	# dw    

        #     elif category == 'Bicycle':
        #         # measurement uncertainty
        #         self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
        #                             [0,1,0,0,0],  	# z
        #                             [0,0,0.05,0,0], 	# theta
        #                             [0,0,0,0.15,0],	# l
        #                             [0,0,0,0,0.05]]) 	# w

        #         # initial state uncertainty at time 0
        #         self.kf.P = np.array([[5,0,0,0,0,0,0,0,0,0],  	# x
        #                             [0,7,0,0,0,0,0,0,0,0],   	# z
        #                             [0,0,.1,0,0,0,0,0,0,0],  	# theta
        #                             [0,0,0,1,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,.5,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,2.5,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,3.5,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,.05,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,.5,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,.25]])	# dw     
            
        #         # process uncertainty, make the constant velocity part more certain
        #         self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
        #                             [0,0,0,0,0,0,0,0,0,0],    # z
        #                             [0,0,0,0,0,0,0,0,0,0],  	# theta
        #                             [0,0,0,0,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,0,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,0.25,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,0.25,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,0.1,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,0.25,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,0.25]])# dw    

        #     elif category == 'Motorcycle':
        #         # measurement uncertainty
        #         self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
        #                             [0,1.2,0,0,0],  	# z
        #                             [0,0,0.015,0,0], 	# theta
        #                             [0,0,0,0.15,0],	# l
        #                             [0,0,0,0,0.035]])	# w

        #         # initial state uncertainty at time 0
        #         self.kf.P = np.array([[5,0,0,0,0,0,0,0,0,0],  	# x
        #                             [0,7,0,0,0,0,0,0,0,0],   	# z
        #                             [0,0,.1,0,0,0,0,0,0,0],  	# theta
        #                             [0,0,0,1,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,.5,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,2.5,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,3.5,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,.05,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,.5,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,.25]])	# dw    
            
        #         # process uncertainty, make the constant velocity part more certain
        #         self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
        #                             [0,0,0,0,0,0,0,0,0,0],    # z
        #                             [0,0,0,0,0,0,0,0,0,0],  	# theta
        #                             [0,0,0,0,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,0,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,0.25,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,0.25,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,0.05,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,0.25,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,0.25]])# dw    

        #     elif category == 'Bus' or category == 'Trailer' or category == 'Truck':
        #         # measurement uncertainty
        #         self.kf.R = np.array([[0.25,0,0,0,0],  	# x  
        #                             [0,1.5,0,0,0],  	# z
        #                             [0,0,0.015,0,0], 	# theta
        #                             [0,0,0,1.5,0],	# l
        #                             [0,0,0,0,0.1]]) 	# w

        #         # initial state uncertainty at time 0
        #         self.kf.P = np.array([[10,0,0,0,0,0,0,0,0,0],  	# x
        #                             [0,15,0,0,0,0,0,0,0,0],   # z
        #                             [0,0,.2,0,0,0,0,0,0,0],  # theta
        #                             [0,0,0,2,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,1,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,3,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,5,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,.05,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,2,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,1]])	# dw     

        #         # process uncertainty, make the constant velocity part more certain
        #         self.kf.Q = np.array([[0,0,0,0,0,0,0,0,0,0],    # x
        #                             [0,0,0,0,0,0,0,0,0,0],    # z
        #                             [0,0,0,0,0,0,0,0,0,0],  	# theta
        #                             [0,0,0,0,0,0,0,0,0,0],	# l
        #                             [0,0,0,0,0,0,0,0,0,0],	# w
        #                             [0,0,0,0,0,0.25,0,0,0,0],	# dx
        #                             [0,0,0,0,0,0,0.25,0,0,0],	# dz
        #                             [0,0,0,0,0,0,0,0.05,0,0],	# dtheta
        #                             [0,0,0,0,0,0,0,0,0.25,0],	# dl
        #                             [0,0,0,0,0,0,0,0,0,0.25]])# dw    


    def filter_predict(self, dt):
        # update F and Q matrices according to current dt 
        self.update_FQ_matrices(dt)
                                
        self.kf.x = self.kf.F @ self.kf.x   # + self.kf.B @ self.kf.u
        self.kf.P = self.kf.F @ self.kf.P @ self.kf.F + self.kf.Q
        self.kf.x_prior = self.kf.x.copy()
        self.kf.P_prior = self.kf.P.copy()
        # return self.kf.x

    def filter_update(self, det):
        self.get_sensor_reading(det)

        # ----- Kalman Gain -----
        self.kf.S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        self.kf.SI = np.linalg.inv(self.kf.S)
        self.kf.K = self.kf.P @ self.kf.H.T @ self.kf.SI

        # ----- Correction ------
        self.kf.y = self.kf.z - self.kf.H @ self.kf.x
        self.kf.x = self.kf.x + self.kf.K @ self.kf.y
        self.kf.P = self.kf.P - self.kf.K @ self.kf.H @ self.kf.P
        self.kf.x_post = self.kf.x.copy()
        self.kf.P_post = self.kf.P.copy()
        # return self.kf.x
    
    def update_FQ_matrices(self, dt):
        self.kf.F = np.array([  [1,0,dt,0],     # x' = x + vx*dt
                                [0,1,0,dt],     # z' = z + vz*dt 
                                [0,0,1,0],	    # vx
                                [0,0,0,1]])	    # vz     

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q = np.array([  [0,0,0,0],          # x
                                [0,0,0,0],          # z
                                [0,0,dt*0.25,0],	# vx
                                [0,0,0,dt*0.25]])	# vz    
        
    def initialize_state_vector(self, det):
        center = det['ct']
        velocity = det['velocity'][:2]

        self.kf.x[0] = center[0]
        self.kf.x[1] = center[1]
        self.kf.x[2] = velocity[0]
        self.kf.x[3] = velocity[1]

    def get_sensor_reading(self, det):
        center = det['ct']
        velocity = det['velocity'][:2]

        self.kf.z[0] = center[0]
        self.kf.z[1] = center[1]
        self.kf.z[2] = velocity[0]
        self.kf.z[3] = velocity[1]