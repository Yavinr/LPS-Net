from utils import *
import argparse
import yaml
import importlib
import math


def get_para():
    parser = argparse.ArgumentParser(
        description='LPS-Net: Lightweight Parameter-Shared Network for Point Cloud-Based Place Recognition')
    parser.add_argument('--config', type=str, default='configs/LPS_Net_S.yaml', help='config file')
    parser.add_argument('--G', type=int, default=1, help='GPU')
    parser.add_argument('--bs', type=int, default=20, help='batch_size')
    parser.add_argument('--ds', type=int, default=4, help='dataset num')
    parser.add_argument('--sp', type=int, default=0, help='if use simple dataset,0=False(default),1=True')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    if args.sp:
        folder = 'preprocessor'
    else:
        folder = 'dataldad'
    cfg["eval_database"] = [os.path.join(folder, f) for f in cfg["eval_database"]]
    cfg["eval_query"] = [os.path.join(folder, f) for f in cfg["eval_query"]]
    cfg["device"] = try_gpu(args.G)
    cfg["eval_batch_size"] = args.bs
    cfg["dataset_num"] = args.ds
    return cfg


def load_one_dataset(args):
    dataset = {}
    dataset['DATABASE_SETS'] = get_sets_dict(
        args["EVAL_DATABASE_FILE"])
    dataset['QUERY_SETS'] = get_sets_dict(args["EVAL_QUERY_FILE"])
    dataset['eval_database_set'] = load_pc_data_set(
        args, dataset['DATABASE_SETS'])
    dataset['eval_query_set'] = load_pc_data_set(
        args, dataset['QUERY_SETS'])
    return dataset


def get_model(args):
    Model = importlib.import_module(args["model"])
    net = Model.Network(args)
    net = net.to(args["device"])
    print('model is running on', next(net.parameters()).device)
    return net


def eval(model, dataset, args):
    result = {
        'top_K_recall': np.zeros(25),
        'top1_similarity_score': [],
        'one_percent_recall': 0.0,
        'lost_num': 0.0,
        'evaluated_samples_num': 0,
        'evaluated_folders_num': 0,
        'database_feature': [],
        'query_feature': [],
    }

    def get_features_eval(args, model, pc_dataset,
                          batch_size=args['eval_batch_size'], if_done=True):
        q_output = np.zeros((1, args['opt_descriptor_dim']))
        iter_num = len(pc_dataset)
        step_num = iter_num // batch_size + (iter_num % batch_size != 0)
        for step_idx in range(step_num):
            pc = pc_dataset[
                 step_idx * batch_size:(step_idx + 1) * batch_size if
                 (step_idx + 1) * batch_size <= iter_num else iter_num]
            # arg = math.radians((2 * random.random() - 1) * 30)
            # pc = rotate_pc(pc, arg)
            feed_tensor = torch.from_numpy(pc).float().to(args["device"])
            # if if_done:
            #     idx = np.random.choice(np.arange(0, 4096, dtype=int), size=(4000), replace=False)
            #     feed_tensor = feed_tensor[:, idx]
            model.eval()
            with torch.no_grad():
                out = model(feed_tensor, mode='gate')
            q_output = np.concatenate(
                (q_output, out.detach().cpu().numpy()), axis=0)
        return q_output[1:]

    print('')
    for i in range(len(dataset['eval_database_set'])):
        print(f"\r calculate eval_database_set "
              f"{i + 1:02d}/{len(dataset['eval_database_set'])}", end='')
        result['database_feature'].append(
            get_features_eval(args, model, dataset['eval_database_set'][i]))
    print('')
    for i in range(len(dataset['eval_query_set'])):
        print(f"\r calculate eval_query_set "
              f"{i + 1:02d}/{len(dataset['eval_query_set'])}", end='')
        result['query_feature'].append(
            get_features_eval(args, model, dataset['eval_query_set'][i], if_done=True))
    result['evaluated_folders_num'] = len(result['query_feature']) * (
            len(result['database_feature']) - 1)
    print('')
    cnt = 0
    for query_idx in range(len(result['query_feature'])):
        for database_idx in range(len(result['database_feature'])):
            if database_idx == query_idx:
                continue
            cnt += 1
            print(f"\r calculate RecallRate {cnt:03d}/{result['evaluated_folders_num']}", end='')
            Ans = calculate_RecallRate(
                query_feature=result['query_feature'][query_idx],
                query_dict=dataset['QUERY_SETS'][query_idx],
                database_feature=result['database_feature'][database_idx],
                database_idx=database_idx)
            for item in Ans:
                result[item] += Ans[item]
    print('')
    result['top_K_recall'] /= result['evaluated_folders_num']
    result['top1_similarity_score'] = np.mean(
        result['top1_similarity_score'])
    result['one_percent_recall'] = (1 - result['lost_num'] / result[
        'evaluated_samples_num']) * 100
    return result


def main():
    args = get_para()
    set_gpu_seed(args["MANUAL_SEED"])
    model = load_para(get_model(args), args)
    calc_model(model)
    results = []
    for i, pickle in enumerate(zip(args['eval_database'], args['eval_query'])):
        print(i, pickle[0])
        args["EVAL_DATABASE_FILE"] = pickle[0]
        args["EVAL_QUERY_FILE"] = pickle[1]
        dataset = load_one_dataset(args)
        res = eval(model, dataset, args)
        results.append(res)
        print('one_percent_recall:', res['one_percent_recall'], '\ntop_K_recall:\n', res['top_K_recall'])
        print('\n', '-' * 100, '\n')
    return results


if __name__ == '__main__':
    main()
