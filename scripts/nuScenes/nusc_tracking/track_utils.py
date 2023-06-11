import numpy as np

def greedy_assignment(dist, main_dim=0):
  secondary_dim = 1 if main_dim==0 else 0
  dist = np.array(dist)
  matched_indices = []
  if dist.shape[secondary_dim] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  
  if main_dim == 0:
    for i in range(dist.shape[0]):
      j = dist[i].argmin()
      if dist[i][j] < 1e16:
        dist[:, j] = 1e18
        matched_indices.append([i, j])
  else:   # main_dim=1
    for j in range(dist.shape[1]):
      i = dist[:, j].argmin()
      if dist[i][j] < 1e16:
        dist[i, :] = 1e18
        matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)