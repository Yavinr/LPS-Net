import torch
import pickle
import numpy as np
import os
from sklearn.neighbors import KDTree
import random
from torch.backends import cudnn


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        torch.cuda.set_device(i)
        return torch.device(f'cuda:{i}')
    return torch.device(f'cuda:{0}')


def set_gpu_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_any(query_list, string):
    d1 = any(word if word in string else False for word in query_list)
    return d1


l = [
    'gate'
]


def check_A_in_B(A, B):
    not_in_B = []
    for k in A.keys():
        if k not in B:
            not_in_B.append(k)
            continue
        if A[k].size() != B[k].size():
            not_in_B.append(k)
    return not_in_B


def load_para(model, args):
    target_state_dict = model.state_dict()
    # src_state_dict = torch.load('para/NewVLADNet_mySharedMLP_4090.params', map_location='cuda:0')
    src_state_dict = torch.load(args['paras_file'], map_location=args["device"])
    skip_keys = []
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            skip_keys.append(k)
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    skip_keys1 = check_A_in_B(src_state_dict, target_state_dict)
    for k in skip_keys:
        del src_state_dict[k]

    model.load_state_dict(src_state_dict, strict=False)
    for name, value in model.named_parameters():
        if check_any(args["froze"], name) and args["if_froze"]:
            value.requires_grad = False
        else:
            value.requires_grad = True
    total = 0
    for name, value in model.named_parameters():
        single = 1
        for s in list(value.shape):
            single *= s
        total += single
    layer = {}
    for name, value in model.named_parameters():
        single = 1
        for s in list(value.shape):
            single *= s
        percent = 100 * single / total
        layer[name] = percent
    layer = sorted(layer.items(), key=lambda x: x[1], reverse=False)
    for l in layer:
        print(l)
    return model


def calc_model(net):
    total = sum([param.nelement() for param in net.parameters()])
    print(f"Number of total parameter: {total / 1e6:.2f}")


def get_queries_dict(filename):
    """

    :param filename:
    :return: key:{'query':file,
                  'positives':[files],
                  'negatives:[files],
                  'neighbors':[keys]}
    """
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries


def get_sets_dict(filename):
    """
    :param filename:
    :return:
    [key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},
    key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    """

    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories


def load_pc_file(filename, dataset_folder, input_dim=3, num_points=4096):
    # returns Nx13 matrix (3 pose 10 handcraft features)
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


def load_pc_data(args, data_, train=True):
    len_data = len(data_.keys())
    pcs = []
    cnt_error = 0
    for i in range(len_data):
        pc = load_pc_file(data_[i]['query'], args["DATASET_FOLDER"], 3,
                          args['NUM_POINTS'])
        pc = pc.astype(np.float32)
        if pc.shape[0] != args['NUM_POINTS']:
            cnt_error += 1
            print('error data! idx: {}'.format(i))
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def load_pc_data_set(args, data_set):
    pc_set = []
    for i in range(len(data_set)):
        pc = load_pc_data(args, data_set[i], train=False)
        pc_set.append(pc)
    return pc_set


def calculate_RecallRate(query_feature, query_dict, database_feature,
                         database_idx, top_K=25):
    database_Tree = KDTree(database_feature)
    recall = [0] * top_K
    top1_similarity_score = []
    one_percent_threshold = max(int(round(len(database_feature) / 100.0)), 1)
    evaluated_samples_num = 0
    for query_idx in range(len(query_feature)):
        true_neighbors = query_dict[query_idx][database_idx]
        if len(true_neighbors) == 0:
            continue
        evaluated_samples_num += 1
        distances, indices = database_Tree.query(
            np.array([query_feature[query_idx]]), k=top_K)
        for i in range(len(indices[0])):
            if indices[0][i] in true_neighbors:
                if i == 0:
                    similarity = np.dot(query_feature[query_idx],
                                        database_feature[indices[0][i]])
                    top1_similarity_score.append(similarity)
                recall[i] += 1
                break
    recall = np.array(recall)
    one_percent_retrieved = np.sum(recall[0:one_percent_threshold])
    one_percent_recall = (one_percent_retrieved / float(
        evaluated_samples_num)) * 100
    top_K_recall = (recall.cumsum() / float(evaluated_samples_num)) * 100
    return {'top_K_recall': top_K_recall,
            'top1_similarity_score': top1_similarity_score,
            'one_percent_recall': one_percent_recall,
            'evaluated_samples_num': evaluated_samples_num,
            'lost_num': evaluated_samples_num - one_percent_retrieved, }


def get_query_tuple(anchor_idx, dict_value, num_pos, num_neg, QUERY_DICT,
                    hard_neg=None, other_neg=False, dataset_folder=None,
                    data=None, num_points=4096):
    if hard_neg is None:
        hard_neg = []
    if num_points < 4096:
        nlist = list(range(4096))
    query = data[anchor_idx]
    if num_points < 4096:
        tidx = np.random.choice(nlist, size=num_points, replace=False)
        query = query[tidx, :]
    pos_files_idx = random.sample(dict_value["positives"], num_pos)
    positives = data[pos_files_idx]
    if num_points < 4096:
        tmp = np.zeros((num_pos, num_points, 3), dtype=np.float32)
        for i in range(num_pos):
            tidx = np.random.choice(nlist, size=num_points, replace=False)
            tmp[i, :, :] = positives[i, tidx, :]
        positives = tmp
    neg_indices = []
    if len(hard_neg) == 0:
        neg_indices = random.sample(dict_value["negatives"], num_neg)
    else:
        neg_indices = neg_indices + hard_neg
        while len(neg_indices) < num_neg:
            idx = random.choice(dict_value["negatives"])
            if idx not in neg_indices:
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
    else:
        neighbors = []
        neighbors = neighbors + dict_value["positives"]
        for neg in neg_indices:
            neighbors = neighbors + QUERY_DICT[neg]["positives"]
        possible_negs = list(set(QUERY_DICT.keys()) - set(neighbors))
        if len(possible_negs) == 0:
            return [query, positives, negatives, np.array([])]
        neg2 = data[random.choice(possible_negs)]
        if num_points < 4096:
            tidx = np.random.choice(nlist, size=num_points, replace=False)
            neg2 = neg2[tidx, :]
        return [query, positives, negatives, neg2]  # Nx3, 2xNx3, 18xNx3, Nx3


def generate_tuple(args, dataset, queries_idx, batch_size):
    q_tuples = []
    for idx in queries_idx:
        if len(dataset['TRAINING_LATENT_VECTORS']) == 0:
            q_tuples.append(get_query_tuple(
                idx,
                dataset['TRAINING_QUERIES'][idx],
                args["TRAIN_POSITIVES_PER_QUERY"],
                args["TRAIN_NEGATIVES_PER_QUERY"],
                dataset['TRAINING_QUERIES'],
                hard_neg=[],
                other_neg=True,
                dataset_folder=args["DATASET_FOLDER"],
                data=dataset['train_data'],
                num_points=args['NUM_POINTS']))
        elif idx not in dataset['HARD_NEGATIVES']:
            query = dataset['TRAINING_LATENT_VECTORS'][idx]
            random_negs = np.random.choice(
                dataset['TRAINING_QUERIES'][idx]['negatives'],
                args['SAMPLED_NEG'],
                replace=False)
            latent_vecs = dataset['TRAINING_LATENT_VECTORS'][random_negs]
            nbrs = KDTree(latent_vecs)
            distances, indices = nbrs.query(np.array([query]),
                                            k=args['NUM_TO_TAKE'])
            hard_negs = np.squeeze(random_negs[indices[0]]).tolist()
            q_tuples.append(get_query_tuple(
                idx,
                dataset['TRAINING_QUERIES'][idx],
                args["TRAIN_POSITIVES_PER_QUERY"],
                args["TRAIN_NEGATIVES_PER_QUERY"],
                dataset['TRAINING_QUERIES'],
                hard_neg=hard_negs,
                other_neg=True,
                dataset_folder=args["DATASET_FOLDER"],
                data=dataset['train_data'],
                num_points=args['NUM_POINTS']))
    return q_tuples


def tuple_reshape(q_tuples):
    queries = []
    positives = []
    negatives = []
    other_neg = []
    for k in range(len(q_tuples)):
        queries.append(q_tuples[k][0])
        positives.append(q_tuples[k][1])
        negatives.append(q_tuples[k][2])
        other_neg.append(q_tuples[k][3])

    queries = np.array(queries, dtype=np.float32)
    queries = np.expand_dims(queries, axis=1)
    other_neg = np.array(other_neg, dtype=np.float32)
    other_neg = np.expand_dims(other_neg, axis=1)
    positives = np.array(positives, dtype=np.float32)
    negatives = np.array(negatives, dtype=np.float32)
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor),
        1)

    return feed_tensor


def rotate_pc(batch_data, rotation_angle):
    """ from pointnet
    Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)),
                                      rotation_matrix)
    return rotated_data


if __name__ == '__main__':
    print(1)
