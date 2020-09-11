"""
Created on June 5, 2020

@author: yhe
"""

import logging

import numpy as np
import torch.autograd
import tqdm
from torch import nn

from doc_analysis.evaluation.retrieval import map_from_query_test_feature_matrices, complete_map_from_qry_test_list


class CASIA_Evaluator(object):
    """
    Class for evaluating CASIA
    """

    def __init__(self, cnn, test_loader, gpu_id, qry_loader=None):
        self.logger = logging.getLogger('CASIA-Evaluation::eval')
        self.dataset_loader = test_loader
        self.qry_loader = qry_loader
        self.cnn = cnn
        self.gpu_id = gpu_id

        # move CNN to GPU
        if self.gpu_id is not None:
            cnn.cuda(self.gpu_id)

    '''
    def _compute_net_outputs(self, data_loader):

        # initialize Data structures
        class_ids = np.zeros(len(data_loader), dtype=np.int32)

        _, ids = data_loader.dataset.lexicon()
        class_size = len(ids)
        output_size = class_size

        output_size = list(self.cnn.children())[-1].out_features
        print('[casia_evaluator] output_size: ', output_size)

        outputs = np.zeros((len(data_loader), output_size), dtype=np.float32)
        qry_ids = []
        correct = 0
        total = 0

        # save RAM to accelerate the running speed
        with torch.no_grad():
            for sample_idx, (character, label, class_id, is_query) in enumerate(tqdm.tqdm(data_loader)):

                if self.gpu_id is not None:
                    character = character.cuda(self.gpu_id)
                    class_id = class_id.cuda(self.gpu_id)

                character = torch.autograd.Variable(character)
                class_id = torch.autograd.Variable(class_id)

                output = torch.softmax(self.cnn(character), dim=1)
                # output = self.cnn(character)

                outputs[sample_idx] = output.data.cpu().numpy().flatten()
                class_ids[sample_idx] = class_id.cpu().numpy()

                # prediction and calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += class_id.size(0)
                correct += (predicted == class_id).sum().item()

                if is_query[0].item() == 1:
                    qry_ids.append(sample_idx)

            qry_outputs = outputs[qry_ids][:]
            qry_class_ids = class_ids[qry_ids]
            print('[qry_outputs.shape]>>>>>>>>>>> ', qry_outputs.shape)

        accuracy = 100 * correct / total
        return accuracy, class_ids, outputs, qry_outputs, qry_class_ids
    '''

    def _compute_net_outputs_with_modified_cnn(self, data_loader):

        # initialize Data structures
        class_ids = np.zeros(len(data_loader), dtype=np.int32)
        _, ids = data_loader.dataset.lexicon()

        qry_ids = []
        correct = 0
        total = 0

        tmp_cnn = self.cnn
        tmp_cnn = self.modi_model(tmp_cnn)

        output_size = list(tmp_cnn.children())[-1].out_features
        outputs = np.zeros((len(data_loader), output_size), dtype=np.float32)
        print('[casia_evaluator] output_size: ', output_size)

        # save RAM to accelerate the running speed
        with torch.no_grad():
            for sample_idx, (character, label, class_id, is_query) in enumerate(tqdm.tqdm(data_loader)):

                if self.gpu_id is not None:
                    character = character.cuda(self.gpu_id)
                    class_id = class_id.cuda(self.gpu_id)

                character = torch.autograd.Variable(character)
                class_id = torch.autograd.Variable(class_id)

                output = torch.softmax(self.cnn(character), dim=1)

                # calculate classification accuracy
                _, predicted = torch.max(output.data, 1)
                total += class_id.size(0)
                correct += (predicted == class_id).sum().item()

                # calculate for mAP
                output_for_mAP = torch.softmax(tmp_cnn(character), dim=1)
                outputs[sample_idx] = output_for_mAP.data.cpu().numpy().flatten()
                class_ids[sample_idx] = class_id.cpu().numpy()

                if is_query[0].item() == 1:
                    qry_ids.append(sample_idx)

            qry_outputs = outputs[qry_ids][:]
            qry_class_ids = class_ids[qry_ids]

        accuracy = 100 * correct / total
        return accuracy, class_ids, outputs, qry_outputs, qry_class_ids

    def eval_qbe(self):
        self.logger.info('---Running QbE Evaluation---')
        self.cnn.eval()

        self.logger.info('Computing net output:')
        accuracy, class_ids, outputs, qry_outputs, qry_class_ids = self._compute_net_outputs_with_modified_cnn(
            self.dataset_loader)

        # compute net outputs for qry images (if not part of test set)
        if self.qry_loader is not None:
            accuracy, _, _, qry_outputs, qry_class_ids = self._compute_net_outputs_with_modified_cnn(
                self.qry_loader)

        # run word spotting
        self.logger.info('Computing mAP...')
        # map_from_query_test_feature_matrices
        # complete_map_from_qry_test_list
        mAP, _ = complete_map_from_qry_test_list(query_features=qry_outputs,
                                                 test_features=outputs,
                                                 query_labels=qry_class_ids,
                                                 test_labels=class_ids,
                                                 metric='cosine',
                                                 drop_first=False)
        return accuracy, mAP * 100

    def eval_kws(self):
        self.logger.info('---Running CASIA Evaluation---')
        self.cnn.eval()
        self.logger.info('Running QbE evaluation...')
        accuracy, mAP_qbe = self.eval_qbe()

        return accuracy, mAP_qbe

    def modi_model(self, model):

        # remove the last fc layer
        print('>>>>>>>> experimental changed by fc configuration')
        children_list = list(model.children())
        # del children_list[-1]

        # remove the last fc layer
        model = nn.Sequential(*(children_list[:-1]))

        # remove the last two fc layers
        # cnn = nn.Sequential(*(children_list[:-2]))

        # fc6 with different number of neurons
        # neurons: 1024, 2048, 4096

        '''
        print('>>>>>>>> number of neurons changed on fc6')
        children_list = children_list[:-2]  # last layer -> fc5
        exp_fc = torch.nn.Linear(4096, 2048)
        children_list.append(exp_fc)    # add exp. fc layer -> fc6
        cnn = nn.Sequential(*children_list)
        '''

        '''
        # fc5 with different number of neurons
        # neurons: 1024, 2048, 4096
        print('>>>>>>>> number of neurons changed on fc5')
        children_list = children_list[:-2]  # last layer -> fc5
        children_list[-1].out_features = 2048
        cnn = nn.Sequential(*children_list)
        '''
        print('>>>>>>>> cnn structure after modification <<<<<<<')
        print(model)
        print('================================================================')

        return model


if __name__ == '__main__':
    pass
