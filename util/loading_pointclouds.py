import os
import pickle
import numpy as np
import random
from scipy.linalg import expm, norm
import torch
import math


def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries


def get_sets_dict(filename):
    # [key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories


"""
def load_pc_file(filename, dataset_folder):
    # returns Nx3 matrix
    pc = np.fromfile(os.path.join(dataset_folder, filename), dtype=np.float64)

    if(pc.shape[0] != 4096*3):
        print("Error in pointcloud shape")
        return np.array([])

    pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc


def load_pc_files(filenames, dataset_folder):
    pcs = []
    for filename in filenames:
        # print(filename)
        pc = load_pc_file(filename, dataset_folder)
        if pc.shape[0] != 4096:
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs
"""


def load_pc_file(filename, dataset_folder, input_dim=3, num_points=4096):
    # returns Nx13 matrix (3 pose 10 handcraft features)
    dataset_folder = 'benchmark_datasets/'
    pc = np.fromfile(os.path.join(dataset_folder, filename), dtype=np.float64)

    if input_dim == 3:
        if pc.shape[0] != num_points * 3:
            print("Error in pointcloud shape")
            print(pc.shape)
            print(filename)
            return np.zeros([num_points, 3])
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    else:
        if pc.shape[0] != num_points * 13:
            print("Error in pointcloud shape")
            print(pc.shape)
            print(filename)
            return np.zeros([num_points, 13])
        pc = np.reshape(pc, (pc.shape[0] // 13, 13))
        # preprocessing data
        # Normalization
        pc[:, 3:12] = ((pc - pc.min(axis=0)) / (
                pc.max(axis=0) - pc.min(axis=0)))[:, 3:12]
        pc[np.isnan(pc)] = 0.0  # some pcs are NAN
        pc[np.isinf(pc)] = 1.0  # some pcs are INF

    return pc


def load_pc_files(filenames, dataset_folder, input_dim=3):
    pcs = []
    for filename in filenames:
        pc = load_pc_file(filename, dataset_folder, input_dim)
        if pc.shape[0] != 4096:
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def rotate_point_cloud(batch_data):
    r""" Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        # -90 to 90
        rotation_angle = (np.random.uniform() * np.pi) - np.pi / 2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.005, clip=0.05):
    r""" Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def get_query_tuple(anchor_idx, dict_value, num_pos, num_neg, QUERY_DICT,
                    hard_neg=[], other_neg=False, dataset_folder=None,
                    data=None, num_points=4096):
    r"""
    get query tuple for dictionary entry
    return list [query,positives,negatives]
    """
    if num_points < 4096:
        nlist = list(range(4096))

    query = data[anchor_idx]
    if num_points < 4096:
        tidx = np.random.choice(nlist, size=num_points, replace=False)
        query = query[tidx, :]

    # get positive samples by random selection
    pos_files_idx = random.sample(dict_value["positives"], num_pos)
    positives = data[pos_files_idx]
    if num_points < 4096:
        tmp = np.zeros((num_pos, num_points, 3), dtype=np.float32)
        for i in range(num_pos):
            tidx = np.random.choice(nlist, size=num_points, replace=False)
            tmp[i, :, :] = positives[i, tidx, :]
        positives = tmp

    neg_indices = []
    if len(hard_neg) == 0:  # if not hard triplet mining, we random choose negative samples
        neg_indices = random.sample(dict_value["negatives"], num_neg)
    else:  # if we have mine hard triplets, we first add the hard negative sample, then random choose the rest
        neg_indices = neg_indices + hard_neg
        while len(neg_indices) < num_neg:
            idx = random.choice(dict_value["negatives"])
            if not idx in neg_indices:
                neg_indices.append(idx)
    negatives = data[neg_indices]
    if num_points < 4096:
        tmp = np.zeros((num_neg, num_points, 3), dtype=np.float32)
        for i in range(num_neg):
            tidx = np.random.choice(nlist, size=num_points, replace=False)
            tmp[i, :, :] = negatives[i, tidx, :]
        negatives = tmp

    if other_neg is False:
        return [query, positives, negatives]
    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        # select the idx of positives
        neighbors = neighbors + dict_value["positives"]

        # select the idx of negative's positives
        for neg in neg_indices:
            neighbors = neighbors + QUERY_DICT[neg]["positives"]
        # erase the whole possible idx, which is the potential areas
        possible_negs = list(set(QUERY_DICT.keys()) - set(neighbors))
        # random.shuffle(possible_negs)

        if len(possible_negs) == 0:  # if not exist the potential areas, then the other_neg=[]
            return [query, positives, negatives, np.array([])]

        # neg2 = data[possible_negs[0]]  # select only one other neg
        neg2 = data[random.choice(possible_negs)]
        if num_points < 4096:
            tidx = np.random.choice(nlist, size=num_points, replace=False)
            neg2 = neg2[tidx, :]

        return [query, positives, negatives, neg2]  # Nx3, 2xNx3, 18xNx3, Nx3


def get_rotated_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[],
                      other_neg=False, dataset_folder=None):
    query = load_pc_file(dict_value["query"], dataset_folder)  # Nx3
    q_rot = rotate_point_cloud(np.expand_dims(query, axis=0))
    q_rot = np.squeeze(q_rot)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    positives = load_pc_files(pos_files, dataset_folder)
    p_rot = rotate_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if (len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while (len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files, dataset_folder)
    n_rot = rotate_point_cloud(negatives)

    if other_neg is False:
        return [q_rot, p_rot, n_rot]

    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys()) - set(neighbors))
        random.shuffle(possible_negs)

        if (len(possible_negs) == 0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"],
                            dataset_folder)
        n2_rot = rotate_point_cloud(np.expand_dims(neg2, axis=0))
        n2_rot = np.squeeze(n2_rot)

        return [q_rot, p_rot, n_rot, n2_rot]


def get_jittered_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[],
                       other_neg=False, dataset_folder=None):
    query = load_pc_file(dict_value["query"], dataset_folder)  # Nx3
    q_jit = jitter_point_cloud(np.expand_dims(query, axis=0))
    q_jit = np.squeeze(q_jit)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    positives = load_pc_files(pos_files, dataset_folder)
    p_jit = jitter_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if (len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while (len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files, dataset_folder)
    n_jit = jitter_point_cloud(negatives)

    if other_neg is False:
        return [q_jit, p_jit, n_jit]

    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys()) - set(neighbors))
        random.shuffle(possible_negs)

        if (len(possible_negs) == 0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"],
                            dataset_folder)
        n2_jit = jitter_point_cloud(np.expand_dims(neg2, axis=0))
        n2_jit = np.squeeze(n2_jit)

        return [q_jit, p_jit, n_jit, n2_jit]


# -------------------------- augmentation -----------------------------
def RandomTranslation(coords, max_delta=0.05):
    trans = max_delta * np.random.randn(1, 3)
    return coords + torch.from_numpy(trans).float()


def JitterPoints(e, sigma=0.01, clip=None, p=1.):
    r"""
    Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    assert 0 < p <= 1.
    assert sigma > 0.
    sample_shape = (e.shape[0],)
    if p < 1.:
        # Create a mask for points to jitter
        m = torch.distributions.categorical.Categorical(
            probs=torch.tensor([1 - p, p]))
        mask = m.sample(sample_shape=sample_shape)
    else:
        mask = torch.ones(sample_shape, dtype=torch.int64)
    mask = mask == 1
    jitter = sigma * torch.randn_like(e[mask])
    if clip is not None:
        jitter = torch.clamp(jitter, min=-clip, max=clip)
    e[mask] = e[mask] + jitter
    return e


def RemoveRandomPoints(e, r):
    if type(r) is list or type(r) is tuple:
        assert len(r) == 2
        assert 0 <= r[0] <= 1
        assert 0 <= r[1] <= 1
        r_min = float(r[0])
        r_max = float(r[1])
    else:
        assert 0 <= r <= 1
        r_min = None
        r_max = float(r)
    n = len(e)
    if r_min is None:
        r = r_max
    else:
        # Randomly select removal ratio
        r = random.uniform(r_min, r_max)
    mask = np.random.choice(range(n), size=int(n * r),
                            replace=False)  # select elements to remove
    e[mask] = torch.zeros_like(e[mask])
    return e


def get_params(coords, scale, ratio):
    # Find point cloud 3D bounding box
    flattened_coords = coords.view(-1, 3)
    min_coords, _ = torch.min(flattened_coords, dim=0)
    max_coords, _ = torch.max(flattened_coords, dim=0)
    span = max_coords - min_coords
    area = span[0] * span[1]
    erase_area = random.uniform(scale[0], scale[1]) * area
    aspect_ratio = random.uniform(ratio[0], ratio[1])
    h = math.sqrt(erase_area * aspect_ratio)
    w = math.sqrt(erase_area / aspect_ratio)
    x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
    y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)
    return x, y, w, h


def RemoveRandomBlock(coords, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    r"""
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    if random.random() < p:
        x, y, w, h = get_params(coords, scale,
                                ratio)  # Fronto-parallel cuboid to remove
        mask = (x < coords[..., 0]) & (coords[..., 0] < x + w) & (
                y < coords[..., 1]) & (coords[..., 1] < y + h)
        coords[mask] = torch.zeros_like(coords[mask])
    return coords


def TrainTransform(e):
    e = JitterPoints(e, sigma=0.001, clip=0.002)
    e = RemoveRandomPoints(e, r=(0.0, 0.1))
    e = RandomTranslation(e, max_delta=0.01)
    e = RemoveRandomBlock(e, p=0.4)
    return e


def EXP_M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(
        np.float32)


def RandomRotation(coords, axis=None, max_theta=180, max_theta2=15):
    if axis is None:
        axis = np.random.rand(3) - 0.5
    R = EXP_M(axis, (np.pi * max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
    R = torch.from_numpy(R).float()

    if max_theta2 is None:
        coords = coords @ R
    else:
        R_n = EXP_M(np.random.rand(3) - 0.5,
                    (np.pi * max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5))
        R_n = torch.from_numpy(R_n).float()
        coords = coords @ R @ R_n
    return coords


def RandomFlip(coords, p):
    # p = [p_x, p_y, p_z] probability of flipping each axis
    assert len(p) == 3
    assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(
        sum(p))
    p_cum_sum = np.cumsum(p)
    r = random.random()
    if r <= p_cum_sum[0]:
        # Flip the first axis
        coords[..., 0] = -coords[..., 0]
    elif r <= p_cum_sum[1]:
        # Flip the second axis
        coords[..., 1] = -coords[..., 1]
    elif r <= p_cum_sum[2]:
        # Flip the third axis
        coords[..., 2] = -coords[..., 2]
    return coords


def TrainSetTransform(e):
    e = RandomRotation(e, max_theta=5, max_theta2=0, axis=np.array([0, 0, 1]))
    e = RandomFlip(e, [0.25, 0.25, 0.])
    return e
