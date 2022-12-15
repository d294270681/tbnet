"""TB-Net evaluation."""

import os
import argparse
import moxing as mox
import math

from mindspore import context, Model, load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype

from src import tbnet, config, metrics, dataset


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Train TBNet.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='steam',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default='test.csv',
        help="the csv datafile inside the dataset folder (e.g. test.csv)"
    )

    parser.add_argument(
        '--checkpoint_id',
        type=int,
        required=True,
        help="use which checkpoint(.ckpt) file to eval"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--device_target',
        type=str,
        required=False,
        default='Ascend',
        choices=['GPU', 'Ascend'],
        help="run code on GPU or Ascend NPU"
    )

    parser.add_argument(
        '--data_url',
        type=str,
        default="./Data",
        help='path where the dataset is saved'
    )

    parser.add_argument(
        '--ckpt_url',
        help='model to save/load',
        default='./ckpt_url'
    )

    parser.add_argument(
        '--result_url',
        help='result folder to save/load',
        default='./result'
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='graph',
        choices=['graph', 'pynative'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    return parser.parse_args()


def eval_tbnet():
    """Evaluation process."""
    args = get_args()
    home = os.path.dirname(os.path.realpath(__file__))
    obs_data_url = args.data_url
    args.data_url = home
    if not os.path.exists(args.data_url):
        os.mkdir(args.data_url)
    try:
        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url,
                                                      args.data_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(
            obs_data_url, args.data_url) + str(e))

    os.system("python "+home+"/preprocess_dataset.py --device_target "+args.device_target)

    obs_ckpt_url = args.ckpt_url
    args.ckpt_url = home + '/checkpoints/tbnet_epoch' + str(args.checkpoint_id) + '.ckpt'
    try:
        mox.file.copy(obs_ckpt_url, args.ckpt_url)
        print("Successfully Download {} to {}".format(obs_ckpt_url,
                                                      args.ckpt_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(
            obs_ckpt_url, args.ckpt_url) + str(e))

    obs_result_url = args.result_url
    args.result_url = '/home/work/user-job-dir/result/'
    if not os.path.exists(args.result_url):
        os.mkdir(args.result_url)

    config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    test_csv_path = os.path.join(home, 'data', args.dataset, args.csv)
    ckpt_path = os.path.join(home, 'checkpoints')

    context.set_context(device_id=args.device_id)
    if args.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    print(f"creating dataset from {test_csv_path}...")
    net_config = config.TBNetConfig(config_path)
    if args.device_target == 'Ascend':
        net_config.per_item_paths = math.ceil(net_config.per_item_paths / 16) * 16
        net_config.embedding_dim = math.ceil(net_config.embedding_dim / 16) * 16
    eval_ds = dataset.create(test_csv_path, net_config.per_item_paths, train=True).batch(net_config.batch_size)

    print(f"creating TBNet from checkpoint {args.checkpoint_id} for evaluation...")
    network = tbnet.TBNet(net_config)
    if args.device_target == 'Ascend':
        network.to_float(mstype.float16)
    param_dict = load_checkpoint(os.path.join(ckpt_path, f'tbnet_epoch{args.checkpoint_id}.ckpt'))
    load_param_into_net(network, param_dict)

    loss_net = tbnet.NetWithLossClass(network, net_config)
    train_net = tbnet.TrainStepWrap(loss_net, net_config.lr)
    train_net.set_train()
    eval_net = tbnet.PredictWithSigmoid(network)
    model = Model(network=train_net, eval_network=eval_net, metrics={'auc': metrics.AUC(), 'acc': metrics.ACC()})
    model.build(valid_dataset=eval_ds, epoch=1)

    print("evaluating...")
    e_out = model.eval(eval_ds)
    print(f'Test AUC:{e_out ["auc"]} ACC:{e_out ["acc"]}')
    filename = 'result.txt'
    file_path = os.path.join(args.result_url, filename)
    with open(file_path, 'a+') as file:
        file.write(f'Test AUC:{e_out["auc"]} ACC:{e_out["acc"]}')

    try:
        mox.file.copy_parallel(args.result_url, obs_result_url)
        print("Successfully Upload {} to {}".format(args.result_url, obs_result_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(args.result_url, obs_result_url) + str(e))

if __name__ == '__main__':
    eval_tbnet()
