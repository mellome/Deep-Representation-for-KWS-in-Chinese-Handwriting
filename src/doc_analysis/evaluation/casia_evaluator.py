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
from cnn.models.modi_phocnet import Modi_PHOCNet
from cnn.models.modi_resnet import Modi_ResNet, resnet18, resnet50
from torch.utils.data import DataLoader


class CASIA_Evaluator(object):
    """
    Class for evaluating CASIA
    """

    def __init__(self, cnn, test_loader, test_set, gpu_id, qry_loader=None):
        self.logger = logging.getLogger('CASIA-Evaluation::eval')
        self.dataset_loader = test_loader
        self.qry_loader = qry_loader
        self.cnn = cnn
        self.gpu_id = gpu_id
        self.query_list, self.query_class_id_list = test_set.generate_random_qry_list()
        self.output_size = list(self.cnn.children())[-1].out_features

        if len(self.query_list) != len(self.query_class_id_list):
            raise ValueError('Length mismatch')

        # move CNN to GPU
        if self.gpu_id is not None:
            cnn.cuda(self.gpu_id)

    def _compute_net_outputs_with_modified_cnn(self, data_loader, qry_is_random):

        tmp_cnn = self.cnn

        # TODO yhe: create a modified model and import entire weights from the old one
        # tmp_cnn = self.modi_model(self.cnn, modified_model=self.create_modi_resnet(net='res50'))
        # tmp_cnn = self.modi_model(self.cnn, modified_model=self.create_modi_phocnet())

        output_size = list(tmp_cnn.children())[-1].out_features
        # move modified CNN to GPU
        if self.gpu_id is not None:
            tmp_cnn.cuda(self.gpu_id)

        # initialize Data structures
        outputs = np.zeros((len(data_loader), output_size), dtype=np.float32)
        class_ids = np.zeros(len(data_loader), dtype=np.int32)
        qry_outputs = np.zeros((len(self.query_list), output_size), dtype=np.float32)
        qry_class_ids = np.zeros((len(self.query_class_id_list)), dtype=np.int32)
        qry_ids = []
        correct = 0
        total = 0

        print('[Evaluator] output_size of outputs: ', output_size)
        self.show_model_orig(self.cnn)
        self.show_model_modified(tmp_cnn)

        if qry_is_random:
            # accuracy in qry by random will not be calculated
            accuracy = 0
            with torch.no_grad():
                # translate qry_char to qry_feature_vector
                for sample_idx, (qry_char, qry_class_id) in enumerate(
                    tqdm.tqdm(zip(self.query_list,
                                  self.query_class_id_list))):

                    if self.gpu_id is not None:
                        qry_char = qry_char.cuda(self.gpu_id)
                        # qry_class_id = qry_class_id.cuda(self.gpu_id)

                    qry_char = torch.autograd.Variable(qry_char)
                    # qry_class_id = torch.autograd.Variable(qry_class_id)

                    # calculate for mAP
                    output_for_mAP = torch.softmax(tmp_cnn(qry_char), dim=1)
                    qry_outputs[sample_idx] = output_for_mAP.data.cpu().numpy().flatten()
                    # qry_class_ids[sample_idx] = qry_class_id.cpu().numpy()
                    qry_class_ids[sample_idx] = qry_class_id

                # translate test_char to test_feature_vector
                for sample_idx, (character, _, class_id, _) in enumerate(tqdm.tqdm(data_loader)):

                    if self.gpu_id is not None:
                        character = character.cuda(self.gpu_id)
                        class_id = class_id.cuda(self.gpu_id)

                    character = torch.autograd.Variable(character)
                    class_id = torch.autograd.Variable(class_id)

                    # calculate for mAP
                    output_for_mAP = torch.softmax(tmp_cnn(character), dim=1)
                    outputs[sample_idx] = output_for_mAP.data.cpu().numpy().flatten()
                    class_ids[sample_idx] = class_id.cpu().numpy()

                    if sample_idx == 1:
                        print('[Evaluator]>>>>>> output from modified cnn')
                        print(output_for_mAP)

                print('>>>>>>>> test_outputs')
                print(outputs.shape)
                print(outputs)
                print('>>>>>>>> test_class_ids')
                print(class_ids)
                print('>>>>>>>> qry_outputs')
                print(qry_outputs.shape)
                print(qry_outputs)
                print('>>>>>>>> qry_class_ids')
                print(qry_class_ids)

        else:
            # save RAM to accelerate the running speed
            with torch.no_grad():
                for sample_idx, (character, label, class_id, is_query) in enumerate(tqdm.tqdm(data_loader)):
                    '''
                    print('character')
                    print(character.numpy().shape)
                    '''
                    if self.gpu_id is not None:
                        character = character.cuda(self.gpu_id)
                        class_id = class_id.cuda(self.gpu_id)

                    character = torch.autograd.Variable(character)
                    class_id = torch.autograd.Variable(class_id)
                    '''
                    print('cnn before softmax')
                    print(self.cnn)
                    '''
                    output = torch.softmax(self.cnn(character), dim=1)
                    output_for_mAP = torch.softmax(tmp_cnn(character), dim=1)

                    # calculate classification accuracy
                    _, predicted = torch.max(output.data, 1)
                    total += class_id.size(0)
                    correct += (predicted == class_id).sum().item()

                    '''
                    print('output')
                    print(output.data.cpu().numpy().shape)
                    print('output_for_mAP')
                    print(output_for_mAP.data.cpu().numpy().shape)
                    print('outputs')
                    print(len(outputs))
                    '''
                    # calculate for mAP
                    outputs[sample_idx] = output_for_mAP.data.cpu().numpy().flatten()
                    class_ids[sample_idx] = class_id.cpu().numpy()

                    if is_query[0].item() == 1:
                        qry_ids.append(sample_idx)

                    if sample_idx == 1:
                        print('[Evaluator]>>>>>> output from cnn')
                        print(output)
                        print('[Evaluator]>>>>>> output from modified cnn')
                        print(output_for_mAP)

                qry_outputs = outputs[qry_ids][:]
                qry_class_ids = class_ids[qry_ids]
            accuracy = 100 * correct / total
        return accuracy, class_ids, outputs, qry_outputs, qry_class_ids

    def eval_qbe(self, qry_is_random):
        self.logger.info('---Running QbE Evaluation---')
        self.cnn.eval()

        self.logger.info('Computing net output:')
        accuracy, class_ids, outputs, qry_outputs, qry_class_ids = self._compute_net_outputs_with_modified_cnn(
            self.dataset_loader,
            qry_is_random)

        # compute net outputs for qry images (if not part of test set)
        if self.qry_loader is not None:
            accuracy, _, _, qry_outputs, qry_class_ids = self._compute_net_outputs_with_modified_cnn(
                self.qry_loader,
                qry_is_random)

        # run word spotting
        self.logger.info('Computing mAP...')
        # map_from_query_test_feature_matrices
        # complete_map_from_qry_test_list
        # TODO yhe: test completely or partially
        mAP, _ = complete_map_from_qry_test_list(query_features=qry_outputs,
                                                 test_features=outputs,
                                                 query_labels=qry_class_ids,
                                                 test_labels=class_ids,
                                                 metric='cosine',
                                                 drop_first=False)
        return accuracy, mAP * 100

    def eval_kws(self, qry_is_random=False):
        self.logger.info('---Running CASIA Evaluation---')
        self.cnn.eval()
        self.logger.info('Running QbE evaluation...')
        accuracy, mAP_qbe = self.eval_qbe(qry_is_random)

        return accuracy, mAP_qbe

    # ########################
    #   helper methods
    # ########################

    def show_model_orig(self, model):
        print('[Evaluator]>>>>>>>> cnn structure <<<<<<<')
        print(model)
        print('================================================================')

    def show_model_modified(self, modified_model):
        print('[Evaluator]>>>>>>>> cnn structure after modification <<<<<<<')
        print(modified_model)
        print('================================================================')

    '''
    create modified models
    '''

    def create_modi_phocnet(self):
        modified_model = Modi_PHOCNet(self.output_size,
                                      input_channels=1,
                                      gpp_type='gpp',
                                      pooling_levels=([1], [5]))
        modified_model.init_weights()
        return modified_model

    def create_modi_resnet(self, net='res18'):
        res = resnet18()
        if net is 'res50':
            res = resnet50()

        res.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7))
        try:
            if res.fc3 is not None:
                res.fc3 = nn.Linear(list(res.children())[-1].in_features, self.output_size)
        except AttributeError as ae:
            print(ae)

        return res

    def modi_model(self, model, modified_model):

        print('>>>>>>>> experimental changed by fc configuration')

        # retrieve the weights
        model_dict = model.state_dict()
        modified_model_dict = modified_model.state_dict()

        '''
        # rename entire keys of modified_model_dict
        for key_of_md, key_of_mmd in zip(list(model_dict.keys()), list(modified_model_dict.keys())):
            modified_model_dict[key_of_md] = modified_model_dict.pop(key_of_mmd)
        print('[Evaluator]>>>>>>> state_dict of modified cnn after rename')
        print(modified_model_dict.keys())
        '''
        for value_of_md, key_of_mmd in zip(list(model_dict.values()), list(modified_model_dict.keys())):
            modified_model_dict[key_of_mmd] = value_of_md
        # print('[Evaluator]>>>>>>> state_dict of modified cnn after rename')
        # print(modified_model_dict)

        # ensure each value of mmd same as each from md
        for value_of_md, value_of_mmd in zip(list(model_dict.values()), list(modified_model_dict.values())):
            if value_of_md is not value_of_mmd:
                raise ValueError('values are not same')

        # remove the unrelated weights from modi_model_dict
        # modified_model_dict = {k: v for k, v in modified_model_dict.items() if k in model_dict}

        # update the latest weights
        # model_dict.update(modified_model_dict)
        # reload the modi_model
        modified_model.load_state_dict(modified_model_dict)
        modified_model.eval()

        return modified_model


if __name__ == '__main__':
    pass
