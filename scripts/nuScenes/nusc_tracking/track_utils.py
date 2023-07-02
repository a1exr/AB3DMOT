import numpy as np

def greedy_assignment(dist, main_dim=0):
  secondary_dim = 1 if main_dim==0 else 0
  # dist = np.array(dist)
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

def iou_matching(dist, dets, preds, coefs):
  matched_indices = []
  dists_shape = dist.shape
  if np.min(dists_shape) == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)

  num_of_candidates = min(5, dists_shape[1])
  for i in range(dists_shape[0]):
    det_dists = dist[i]
    pred_cands = np.argpartition(det_dists, num_of_candidates-1)[:num_of_candidates]
    matching_scores = np.zeros(num_of_candidates)
    cur_det = dets[i]
    for jj, cand_indx in enumerate(pred_cands):
      if det_dists[cand_indx] > 1e16:
        continue

      cur_pred = preds[cand_indx]
      size_diff = np.array(cur_det['size']) - np.array(cur_pred['size'])
      size_score = coefs['size']**np.sum(np.abs(size_diff))  # TODO: optimize
      dist_score = coefs['dist']**dist[i, cand_indx]  # TODO: optimize
      conf_score = coefs['confidence']*cur_pred['detection_score']
      iou_score = coefs['iou']*iou_2d(cur_det, cur_pred)

      matching_scores[jj] = iou_score + conf_score + dist_score + size_score
      # matching_scores[jj] = dist_score + size_score

    best_cand_indx = matching_scores.argmax()
    j = pred_cands[best_cand_indx]
    if matching_scores[best_cand_indx] > 0:
      dist[:, j] = 1e18   # TODO: consider hungarian algorithm here
      matched_indices.append([i, j])

  return np.array(matched_indices, np.int32).reshape(-1, 2)

def iou_2d(box1, box2):
    x1, z1, _ = box1['translation']
    w1, l1, _ = box1['size']
    r1 = box1['rotation']
    
    x2, z2, _ = box2['translation']
    w2, l2, _ = box2['size']
    r2 = box2['rotation']

    # Convert rotation vectors to rotation matrices
    rot_mat1 = quaternion_to_rotation_matrix(r1)
    rot_mat2 = quaternion_to_rotation_matrix(r2)

    # Calculate rotated box corners for box1
    corners1 = get_box_corners(x1, z1, w1, l1, rot_mat1)

    # Calculate rotated box corners for box2
    corners2 = get_box_corners(x2, z2, w2, l2, rot_mat2)

    # Calculate the intersection area
    intersection_area = calculate_intersection_area(corners1, corners2)

    # Calculate the union area
    box1_area = w1 * l1
    box2_area = w2 * l2
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    return iou

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    rot_mat = np.zeros((2, 2))
    rot_mat[0, 0] = 1 - 2 * (y**2 + z**2)
    rot_mat[0, 1] = 2 * (x * y - w * z)
    rot_mat[1, 0] = 2 * (x * y + w * z)
    rot_mat[1, 1] = 1 - 2 * (x**2 + z**2)
    return rot_mat

def get_box_corners(x, z, w, l, rot_mat):
    # Calculate box corners in the local coordinate system
    corners = np.array([
        [-w / 2, -l / 2],
        [w / 2, -l / 2],
        [w / 2, l / 2],
        [-w / 2, l / 2]
    ])

    # Rotate the corners
    corners = np.matmul(corners, rot_mat.T)

    # Translate the corners to the global coordinate system
    corners[:, 0] += x
    corners[:, 1] += z

    return corners

def calculate_intersection_area(corners1, corners2):
    min_x = max(np.min(corners1[:, 0]), np.min(corners2[:, 0]))
    max_x = min(np.max(corners1[:, 0]), np.max(corners2[:, 0]))
    min_z = max(np.min(corners1[:, 1]), np.min(corners2[:, 1]))
    max_z = min(np.max(corners1[:, 1]), np.max(corners2[:, 1]))

    intersection_width = max(0, max_x - min_x)
    intersection_height = max(0, max_z - min_z)
    intersection_area = intersection_width * intersection_height

    return intersection_area

# def calculate_iou(box1, box2):
#     # Extract box coordinates, dimensions, and rotation
#     x1, z1, y1 = box1['translation']  # Corrected order (x, z, y)
#     w1, l1, h1 = box1['size']
#     r1 = box1['rotation']
    
#     x2, z2, y2 = box2['translation']  # Corrected order (x, z, y)
#     w2, l2, h2 = box2['size']
#     r2 = box2['rotation']

#     # Convert rotation vectors to rotation matrices
#     rot_mat1 = quaternion_to_rotation_matrix(r1)
#     rot_mat2 = quaternion_to_rotation_matrix(r2)

#     corners1 = get_box_corners(x1, z1, y1, w1, l1, h1, rot_mat1)
#     corners2 = get_box_corners(x2, z2, y2, w2, l2, h2, rot_mat2)

#     intersection = calculate_intersection(corners1, corners2)
#     union = w1 * l1 * h1 + w2 * l2 * h2 - intersection

#     # Calculate the IoU
#     iou = intersection / union
#     return iou

# def quaternion_to_rotation_matrix(q):
#     # Convert quaternion to rotation matrix
#     x, y, z, w = q
#     rot_mat = np.zeros((3, 3))
#     rot_mat[0, 0] = 1 - 2 * (y**2 + z**2)
#     rot_mat[0, 1] = 2 * (x * y - z * w)
#     rot_mat[0, 2] = 2 * (x * z + y * w)
#     rot_mat[1, 0] = 2 * (x * y + z * w)
#     rot_mat[1, 1] = 1 - 2 * (x**2 + z**2)
#     rot_mat[1, 2] = 2 * (y * z - x * w)
#     rot_mat[2, 0] = 2 * (x * z - y * w)
#     rot_mat[2, 1] = 2 * (y * z + x * w)
#     rot_mat[2, 2] = 1 - 2 * (x**2 + y**2)
#     return rot_mat

# def get_box_corners(x, z, y, w, l, h, rot_mat):
#     # Calculate box corners in the global coordinate system
#     corners = np.array([
#         [x - w / 2, z - l / 2, y - h / 2],
#         [x + w / 2, z - l / 2, y - h / 2],
#         [x + w / 2, z + l / 2, y - h / 2],
#         [x - w / 2, z + l / 2, y - h / 2],
#         [x - w / 2, z - l / 2, y + h / 2],
#         [x + w / 2, z - l / 2, y + h / 2],
#         [x + w / 2, z + l / 2, y + h / 2],
#         [x - w / 2, z + l / 2, y + h / 2]
#     ])

#     # Rotate the corners
#     corners = np.matmul(corners, rot_mat.T)
#     return corners

# def calculate_intersection(corners1, corners2):
#     min_coords = np.maximum(np.min(corners1, axis=0), np.min(corners2, axis=0))
#     max_coords = np.minimum(np.max(corners1, axis=0), np.max(corners2, axis=0))
#     intersection_dims = np.maximum(0, max_coords - min_coords)
#     intersection = np.prod(intersection_dims)
#     return intersection
