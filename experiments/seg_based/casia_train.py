"""
Created on May 26, 2020

@author: yhe

"""
import argparse
import copy
import logging
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from doc_analysis.evaluation.casia_evaluator import CASIA_Evaluator
from cnn.models.myphocnet import PHOCNet
from cnn.models.residual_net import resnet18, resnet34, resnet50
from seg_based.datasets.casia import CASIADataset
from utils.save_load import my_torch_save, my_torch_load


# ########################
#   helper methods
# ########################

def learning_rate_step_parser(lrs_string):
    """
    Parser learning rate string
    """
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for
            elem in lrs_string.split(',')]


def imshow_single(character_tensor):
    character_numpy = character_tensor.numpy()
    character_numpy = character_numpy.reshape(character_numpy.shape[1:])
    plt.imshow(character_numpy, cmap='Greys')
    plt.show()


def imshow_many(character_tensors):
    for character_tensor in character_tensors:
        imshow_single(character_tensor)


def local_mode():
    return os.getcwd().startswith('/Users')


def set_args_in_local_mode():
    data_root_dir = '/Users/mellome1992/Documents/LocalRepository/phocnet_kws/src/gnt_utils/dataset'
    model_dir = '/Users/mellome1992/Documents/LocalRepository/phocnet_kws/src/gnt_utils/models'
    gpu_id = None
    train_split = [1]
    test_split = [2]

    return data_root_dir, model_dir, gpu_id, train_split, test_split


def check_validation(test_set):
    qry_size = test_set.get_qry_size()
    print("[casia_train]<<<<<<<<<<<< checking validation of qry_list")
    print('length of qry_list: %d' % qry_size)
    print("[casia_train]>>>>>>>>>>>")
    return qry_size > 0


'''
def config_resnet(res, output_size):
    children_list = list(res.children())
    children_list[-1].out_features = 4096
    fc6 = torch.nn.Linear(4096, 4096)
    fc7 = torch.nn.Linear(4096, output_size)
    children_list.append(fc6)
    children_list.append(fc7)
    res = nn.Sequential(*children_list)
    return res
'''


def config_resnet(res, output_size):
    res.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7))
    res.fc3 = nn.Linear(list(res.children())[-1].in_features, output_size)
    return res


###########################################

def train(local_mode=True):
    """
    Method for Training CASIA_PHOCNet
    """
    pt_collector = ['/data/yhe/models/hwdb1_0_4096_phoc',  # 0
                    '/data/yhe/models/hwdb1_0_4096_res50',  # 1
                    '/data/yhe/models/hwdb1_0_4096_res18',  # 2
                    '/data/yhe/models/hwdb1_1_4096_res18',  # 3
                    '/data/yhe/models/hwdb1_1_4096_res50',  # 4
                    '/data/yhe/models/hwdb1_1_4096_phoc',  # 5
                    ]
    logger = logging.getLogger('CASIA_PHOCNet-Experiment::train')
    logger.info('--- Running CASIA_PHOCNet Training ---')

    model_dir = '/data/yhe/models'
    gpu_id = '2'

    # ########################
    #   config a simple way
    # ########################

    '''
    only for HWDB1.0
    '''
    data_root_dir, train_split, test_split = '/data/yhe/hwdb1_0', [1, 2, 3], [4]

    '''
    only for HWDB1.1
    '''
    # data_root_dir = '/vol/corpora/document-image-analysis/casia' # complete ds for casia
    # data_root_dir, train_split, test_split = '/data/yhe/partial_DS/more', [5, 6], [7]
    # data_root_dir, train_split, test_split = '/data/yhe/partial_DS/less', [5, 6], [7]
    # TODO yhe: choose a trained model for testing
    pt_collector = pt_collector[5]

    if local_mode:
        data_root_dir, model_dir, gpu_id, train_split, test_split = set_args_in_local_mode()

    # argument parsing
    parser = argparse.ArgumentParser()

    # -misc
    parser.add_argument('--model_dir', '-mdir', action='store', type=str,
                        default=model_dir,
                        help='The path of the where to save trained models' + \
                             'Default: ' + model_dir)
    parser.add_argument('--experiment_id', '-exp_id', action='store', type=int, default=1978,
                        help='The Experiment ID. Default: Based on Model directory')
    parser.add_argument('--gpu_id', '-gpu', action='store',
                        type=int, default=gpu_id,
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')

    # - dataset arguments
    parser.add_argument('--dataset', '-ds', default='casia',
                        help='The dataset to be trained on')
    parser.add_argument('--train_split', '-trains', action='store', type=int, default=train_split,
                        help='The split of the train dataset. Default: [5, 6]')
    parser.add_argument('--test_split', '-tests', action='store', type=int, default=test_split,
                        help='The split of the test dataset. Default: [7]')
    parser.add_argument('--local_mode', '-lm', action='store', type=bool, default=False,
                        help='The mode of the experiment. Default: local mode (False)')

    parser.add_argument('--augmentation', '-aug', choices=['none', 'balanced', 'unbalanced'], default='none',
                        help='Data augmentation type')

    # - train arguments
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--loss_type', '-lt', choices=['BCE', 'cosine'], default='cosine',
                        help='The Type of loss function')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')

    parser.add_argument('--display', action='store', type=int, default=10,
                        help='The number of iterations after which to display the loss values. Default: 10')
    parser.add_argument('--test_interval', action='store', type=int, default=10,
                        help='The number of iterations after which to periodically evaluate the CASIA_PHOCNet. '
                             'Default: 10')
    parser.add_argument('--iter_size', '-is', action='store', type=int, default=40,
                        help='The iteration size after which the gradient is computed. Default: 500')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=150,
                        help='The batch size after which the gradient is computed. Default: 150')
    parser.add_argument('--run_kws', '-kws', action='store', type=bool, default=False,
                        help='Running kws now? Default: False')
    parser.add_argument('--net', action='store', type=str, default='phoc',
                        help='Choose a net for training and testing later. Default: phoc')

    args = parser.parse_args()

    ###############################################################
    # prepare partitions and code map
    train_set = CASIADataset(casia_root_dir=data_root_dir,
                             train_split=args.train_split,
                             test_split=args.test_split,
                             local_mode=local_mode)

    train_set.mainLoader(partition='train')
    test_set = copy.copy(train_set)
    test_set.mainLoader(partition='test')

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=8)
    test_loader = DataLoader(test_set,
                             batch_size=1, shuffle=True,
                             num_workers=8)

    # train_loader_iter = _BaseDataLoaderIter(loader=train_loader)

    # load CNN
    logger.info('Preparing PHOCNet...')

    _, class_ids = train_set.lexicon()
    output_size = len(class_ids)
    print(">>>>>>>> output_size %d " % output_size)

    print(">>>>>>>> qry_size ")
    print(test_set.get_qry_size())
    print("<<<<<<<< ")

    if args.net == 'phoc':
        cnn = PHOCNet(output_size,
                      input_channels=1,
                      gpp_type='gpp',
                      pooling_levels=([1], [5]))
        cnn.init_weights()

    if args.net == 'res18':
        cnn = config_resnet(resnet18(), output_size)

    if args.net == 'res34':
        cnn = config_resnet(resnet34(), output_size)

    if args.net == 'res50':
        cnn = config_resnet(resnet50(), output_size)

    print('[Train]>>>>>>>> cnn structure <<<<<<<')
    print(cnn)
    print('================================================================')

    # move CNN to GPU
    if args.gpu_id is not None:
        cnn.cuda(args.gpu_id)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    # TODO yhe: show ont_hot
    one_hot = None
    epochs = 30
    for epoch_idx in range(epochs):

        # check model is already trained
        if args.run_kws:
            break
        print('>>>>>>> epoch_idx ', epoch_idx)

        # evaluating during training process˝
        if (epoch_idx + 1) % args.test_interval == 0:

            # TODO yhe: show ont_hot
            print('>>>>>>> one_hot ')
            print(one_hot)

            logger.info('Evaluating net after %d epochs', epoch_idx + 1)
            print('Evaluating net after %d epochs' % (epoch_idx + 1))
            cnn.eval()

            evaluator = CASIA_Evaluator(cnn, test_loader, test_set, args.gpu_id)
            accuracy, mAP_qbe = evaluator.eval_kws()
            logger.info('accuracy: %3.2f    QbE mAP: %3.2f', accuracy, mAP_qbe)
            print('accuracy: %3.2f     QbE mAP: %3.2f' % (accuracy, mAP_qbe))
            cnn.train()

        for i, train_data in enumerate(train_loader, 0):
            inputs, labels, class_ids, _ = train_data

            if args.gpu_id is not None:
                inputs = inputs.cuda(args.gpu_id)
                class_ids = class_ids.cuda(args.gpu_id)

            # torch.Size([150,1,32,32]) . 150 is batch_size
            inputs = torch.autograd.Variable(inputs)
            class_ids = torch.autograd.Variable(class_ids)

            # forward + backward + optimize
            outputs = cnn(inputs)
            loss_val = criterion(outputs, class_ids)
            # TODO yhe: show ont_hot
            one_hot = outputs

            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()


        # print loss
        if (epoch_idx + 1) % args.display == 0:
            print('Epoch %d, Class Id %d, Loss %f \n' % (epoch_idx + 1, i, loss_val.item()))
            logger.info('Epoch %*d: %f', epoch_idx + 1, loss_val.item())

    # save network
    if not local_mode and not args.run_kws:
        file_name = ('%s_model_%d_%s' % (args.net,
                                         output_size,
                                         time.strftime("%Y-%m-%d_%H:%M", time.localtime())))
        torch.save(cnn.state_dict(), os.path.join(args.model_dir, file_name))
        print('>>>>>>> Finished Training')
        print('>>>>>>> Model saved successfully')

    # running kws on trained model
    if args.run_kws and check_validation(test_set):
        logger.info('Loading trained net')
        print('Loading trained net')

        my_torch_load(cnn, pt_collector)
        print('Loading trained phoc model')
        '''
        if args.net is 'phoc':
            my_torch_load(cnn, pt_collector)
            print('Loading trained phoc model')
        else:
            cnn.load_state_dict(torch.load(pt_collector))
            print('Loading trained res model')
        '''
        cnn.eval()

        evaluator = CASIA_Evaluator(cnn, test_loader, test_set, args.gpu_id)
        accuracy, mAP_qbe = evaluator.eval_kws(qry_is_random=False)
        print('accuracy: %3.2f     QbE mAP: %3.2f' % (accuracy, mAP_qbe))


if __name__ == '__main__':
    # calculate run time
    start = time.time()
    train(local_mode())
    # print(resnet18())
    # print(config_resnet(resnet18(), 3926))
    print('>>>>>> Run time <<<<<<: ', (time.time() - start) / 60)
