#coding=utf-8

class DefaultConfig(object):
    env = 'default'  #visdom环境
    model = 'AlexNet'

    train_data_root = './data/train/'
    test_data_root = './data/test/'
    load_model_path = 'checkpoints/model.pth'

    batch_size = 128
    num_workers = 4
    print_freq = 20
    
    debug_file = './debug'
    result_file = 'result.csv'

    max_epoch = 0
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4
