import os
import argparse

try:
    import moxing as mox
except:
    print('not use moxing')


def prepare_data_on_modelarts(args):
    if args.pretrained_weights:
        _, weights_name = os.path.split(args.pretrained_weights)
        mox.file.copy(args.pretrained_weights, os.path.join(args.local_data_root, 'model/'+weights_name))
        args.pretrained_weights = os.path.join(args.local_data_root, 'model/'+weights_name)
    if not (args.data_url.startswith('s3://') or args.data_url.startswith('obs://')):
        args.data_local = args.data_url
    else:
        os.mkdir(os.path.join(args.local_data_root, 'datasets'))
        args.data_local = os.path.join(args.local_data_root, 'datasets/')
        if not os.path.exists(args.data_local):
            os.system('ls /cache')
            os.mkdir(args.data_local)
            #data_dir = os.path.join(args.local_data_root, 'datasets')
            mox.file.copy_parallel(args.data_url, args.data_local)

        else:
            print('args.data_local: %s is already exist, skip copy' % args.data_local)

    if not (args.train_url.startswith('s3://') or args.train_url.startswith('obs://')):
        args.train_local = args.train_url
    else:
        args.train_local = os.path.join(args.local_data_root, 'log/')
        if not os.path.exists(args.train_local):
            os.mkdir(args.train_local)

    return args


def gen_model_dir(log_dir, args):
    if args.train_url.startswith('s3://') or args.train_url.startswith('obs://'):
        mox.file.copy_parallel(log_dir, args.train_url)

    current_dir = os.path.dirname(__file__)

    print('gen_model_dir success, model dir is at', os.path.join(args.train_url, 'model'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", default='', type=str, help="if specified starts from checkpoint model")
    parser.add_argument('--local_data_root', default='/cache/', type=str,
                        help='a directory used for transfer data between local path and OBS path')
    parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
    parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
    parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
    parser.add_argument('--train_local', default='', type=str, help='the training output results on local')

    parser.add_argument('--init_method', default='', type=str, help='the training output results on local')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')

    opt = parser.parse_args()
    print(opt)
    opt = prepare_data_on_modelarts(opt)

    #############################
    # put your train script here
    # your dataset will be put in /cache/datasets
    # your pretrain models will be put in /cache/model
    # your trained models will be put in /cache/log
    #############################

    gen_model_dir(opt.train_local, opt)