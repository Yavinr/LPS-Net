from models.import_model import *
import datetime
import argparse
import importlib
import yaml
from utils import *


def get_para():
    parser = argparse.ArgumentParser(
        description='LPS-Net: Lightweight Parameter-Shared Network for Point Cloud-Based Place Recognition')
    parser.add_argument('--config', type=str, default='configs/LPS_Net_L.yaml', help='config file')
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--bs', type=int, default=None, help='batch_size')
    parser.add_argument('--epc', type=int, default=None, help='epoch,default=None')
    parser.add_argument('--ds', type=str, default='bl', help='dataset,bl=baseline,rf=refine,sp=simple')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    if args.ds == 'sp':
        folder = 'preprocessor'
        cfg["SAMPLED_NEG"] = 40
    elif args.ds == 'rf':
        folder = 'dataldad'
        cfg["TRAIN_FILE"] = 'training_queries_refine_v1.pickle'
    else:
        folder = 'dataldad'
    cfg["EVAL_DATABASE_FILE"] = os.path.join(folder, cfg["EVAL_DATABASE_FILE"])
    cfg["EVAL_QUERY_FILE"] = os.path.join(folder, cfg["EVAL_QUERY_FILE"])
    cfg["TRAIN_FILE"] = os.path.join(folder, cfg["TRAIN_FILE"])
    cfg["TEST_FILE"] = os.path.join(folder, cfg["TEST_FILE"])
    cfg["yaml"] = args.config[8:]
    cfg["batch_size"] = args.bs
    cfg["device"] = try_gpu(args.GPU)
    cfg["GPU_idx"] = args.GPU
    cfg["ds"] = args.ds
    if args.epc is not None:
        cfg["MAX_EPOCH"] = args.epc
    return cfg


def load_dataset(args):
    dataset = {}
    eligible_queries_idx = []
    dataset['TRAINING_QUERIES'] = get_queries_dict(args["TRAIN_FILE"])
    for idx in range(len(dataset['TRAINING_QUERIES'])):
        if len(dataset['TRAINING_QUERIES'][idx]["positives"]) >= args["TRAIN_POSITIVES_PER_QUERY"]:
            eligible_queries_idx.append(idx)
    dataset['train_file_idxs'] = np.array(eligible_queries_idx)
    dataset['DATABASE_SETS'] = get_sets_dict(args["EVAL_DATABASE_FILE"])
    dataset['QUERY_SETS'] = get_sets_dict(args["EVAL_QUERY_FILE"])
    dataset['train_data'] = load_pc_data(args, dataset['TRAINING_QUERIES'])
    dataset['eval_database_set'] = load_pc_data_set(args, dataset['DATABASE_SETS'])
    dataset['eval_query_set'] = load_pc_data_set(args, dataset['QUERY_SETS'])
    dataset['TRAINING_LATENT_VECTORS'] = []
    dataset['HARD_NEGATIVES'] = []
    return dataset


def get_model(args):
    Model = importlib.import_module(args["model"])
    net = Model.Network(args)
    net = net.to(args["device"])
    print('model is running on', next(net.parameters()).device)
    return net


def train_one_step(model, optimizer, dataset, args, loss_function, epoch,
                   batch_size, step_idx, iter_num, step_num):
    queries_idx = dataset['train_file_idxs'][
                  step_idx * batch_size:(step_idx + 1) * batch_size if
                  (step_idx + 1) * batch_size <= iter_num else iter_num]
    q_tuples = generate_tuple(
        args, dataset, queries_idx, batch_size)
    feed_tensor = tuple_reshape(q_tuples)
    model.train()
    optimizer.zero_grad()
    feed_tensor = feed_tensor.view((-1, args["NUM_POINTS"], 3)).to(args["device"]).requires_grad_(True)
    output = model(feed_tensor, mode=args['train_opt_mode'])
    if args['train_opt_mode'] == 'gate':
        output = output
    elif args['train_opt_mode'] == 'punish':
        output, punish = output
    output = output.view(len(q_tuples), -1, args["opt_descriptor_dim"])
    output_queries, output_positives, output_negatives, output_other_neg = \
        torch.split(output, [1, args["TRAIN_POSITIVES_PER_QUERY"],
                             args["TRAIN_NEGATIVES_PER_QUERY"], 1], dim=1)
    loss, _ = loss_function(output_queries, output_positives,
                            output_negatives, output_other_neg,
                            args["MARGIN_1"], args["MARGIN_2"],
                            use_min=args["TRIPLET_USE_BEST_POSITIVES"],
                            lazy=args["LOSS_LAZY"],
                            ignore_zero_loss=args["LOSS_IGNORE_ZERO_BATCH"])
    if args['train_opt_mode'] == 'gate':
        pass
    else:
        loss += punish.sum() * args['VLAD_punish']

    if loss > 1e-10:
        loss.backward(retain_graph=True)
        optimizer.step()
    print(f"\repoch: {epoch:02d}, step: {step_idx + 1:04d}/{step_num}, "
          f"loss: {loss:.3f}  ", end='')


def train_one_epoch(model, optimizer, dataset, args, loss_function, epoch):
    def get_features(args_, model_, pc_dataset_, batch_size_):
        q_output = np.zeros((1, args_['opt_descriptor_dim']))
        iter_num_ = len(pc_dataset_)
        step_num_ = iter_num_ // batch_size_ + (iter_num_ % batch_size_ != 0)
        for step_idx_ in range(step_num_):
            pc_ = pc_dataset_[
                  step_idx_ * batch_size_:(step_idx_ + 1) * batch_size_ if
                  (step_idx_ + 1) * batch_size_ <= iter_num_ else iter_num_]
            feed_tensor = \
                torch.from_numpy(pc_).float().to(args_["device"])
            model_.eval()
            with torch.no_grad():
                out = model_(feed_tensor, mode=args_['eval_opt_mode'])
            q_output = np.concatenate(
                (q_output, out.detach().cpu().numpy()), axis=0)
            print(
                f"\rcalculate latent vectors: {step_idx_ + 1:02d}/{step_num_}",
                end='')
        return q_output[1:]

    batch_size = args['BATCH_NUM_QUERIES']
    iter_num = len(dataset['train_file_idxs'])
    step_num = iter_num // batch_size + (iter_num % batch_size != 0)
    eval_step = args['eval_step'] // batch_size
    np.random.shuffle(dataset['train_file_idxs'])
    dataset['query_idx'] = 0
    print(f"epoch {epoch}", '-' * 5, args["device"], '-' * 5)
    if args["batch_size"] is not None:
        step_num = args["batch_size"]
    for step_idx in range(step_num):
        train_one_step(model, optimizer, dataset, args, loss_function, epoch,
                       batch_size, step_idx, iter_num, step_num)
        if (epoch > 5 and step_idx % eval_step == 0) or False:
            dataset['TRAINING_LATENT_VECTORS'] = get_features(
                args, model, dataset['train_data'],
                batch_size_=args['eval_batch_size'])


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

    def get_features_eval(args_, model_, pc_dataset,
                          batch_size=args['eval_batch_size']):
        q_output = np.zeros((1, args_['opt_descriptor_dim']))
        iter_num = len(pc_dataset)
        step_num = iter_num // batch_size + (iter_num % batch_size != 0)
        for step_idx in range(step_num):
            pc = pc_dataset[
                 step_idx * batch_size:(step_idx + 1) * batch_size if
                 (step_idx + 1) * batch_size <= iter_num else iter_num]
            feed_tensor = \
                torch.from_numpy(pc).float().to(args_["device"])
            model_.eval()
            with torch.no_grad():
                out = model_(feed_tensor, mode=args_['eval_opt_mode'])
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
            get_features_eval(args, model, dataset['eval_query_set'][i]))

    result['evaluated_folders_num'] = len(result['query_feature']) * (
            len(result['database_feature']) - 1)
    print('')
    cnt = 0
    for query_idx in range(len(result['query_feature'])):
        for database_idx in range(len(result['database_feature'])):
            if database_idx == query_idx:
                continue
            cnt += 1
            print(f"\r calculate RecallRate "
                  f"{cnt:03d}/{result['evaluated_folders_num']}",
                  end='')
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
    result['one_percent_recall'] /= result['evaluated_folders_num']
    print(result['top_K_recall'][:4])
    return result


def main():
    args = get_para()
    set_gpu_seed(args["MANUAL_SEED"])
    loss_fun = get_loss(args)
    model = get_model(args)
    optimizer, lr_scheduler = get_optimizer(args, model)
    epoch_offset = 0
    calc_model(model)
    dataset = load_dataset(args)
    eval(model, dataset, args)
    for e in range(args["MAX_EPOCH"] - epoch_offset):
        epoch = e + epoch_offset
        train_one_epoch(
            model, optimizer, dataset, args, loss_fun, epoch)
        eval(model, dataset, args)
        torch.save(model.state_dict(), f"log/epoch_{e}.params")
        if lr_scheduler is not None:
            lr_scheduler.step()


if __name__ == '__main__':
    main()
    print(f"finished at {str(datetime.datetime.now())}")
