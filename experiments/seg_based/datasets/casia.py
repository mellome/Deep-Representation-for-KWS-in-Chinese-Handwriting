"""
Created on Mai 20, 2020

@author: yhe
"""
import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sklearn.decomposition import PCA

from doc_analysis.transformations.homography_augmentation import HomographyAugmentation
from seg_based.datasets.dataset_loader import DatasetLoader


# ########################
#   helper methods
# ########################

def imshow(character_tensor):
    character_numpy = character_tensor.numpy()
    character_numpy = character_numpy.reshape(character_numpy.shape[1:])
    plt.imshow(character_numpy, cmap='Greys')
    plt.show()


def conv_to_tensor(char_list):
    ten_char_list = []
    for char in char_list:
        nor_char = 1 - char.get_character() / 255.0
        nor_char = nor_char.reshape((1,) + nor_char.shape)
        ten_char_list.append(torch.from_numpy(nor_char.astype(np.float32)))
    return ten_char_list


class CASIADataset(Dataset):
    """
    PyTorch dataset class for the segmentation-based CASIA dataset
    """

    def __init__(self,
                 casia_root_dir='/vol/corpora/document-image-analysis/casia',
                 train_split=[1],
                 test_split=[2],
                 local_mode=True
                 ):

        if local_mode:
            casia_root_dir = '/Users/mellome1992/Documents/LocalRepository/phocnet/src/gnt_utils/dataset'

        # load the dataset
        character_list, code_map = DatasetLoader.load_casia(casia_root_dir)
        self.code_map = code_map

        # Gnt dataset(s) with split_id loads as train list
        self.train_list = [character for character in character_list
                           if int(character.get_folder_id()) in train_split]
        self.test_list = [character for character in character_list
                          if int(character.get_folder_id()) in test_split]

        # compute a mapping from class string to class id
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([character.get_label() for character in self.train_list + self.test_list])

    def __len__(self):
        """
        returns length of the dataset.
        """
        return len(self.character_list)

    def __getitem__(self, index):
        character = self.character_list[index].get_character()
        character = 1 - character / 255.0
        label = self.character_list[index].get_label()

        class_id = self.label_encoder.transform([label])[0]
        is_query = self.query_list[index]

        # prepare data for torch
        character = character.reshape((1,) + character.shape)
        character_tensor = torch.from_numpy(character.astype(np.float32))

        # class_id = self.code_map[label]
        return character_tensor, label, class_id, is_query

    def mainLoader(self, partition=None, transforms=HomographyAugmentation()):
        """
        Initializes Dataloader for desired partition
        """
        self.transforms = transforms

        if partition not in [None, 'train', 'test']:
            raise ValueError('partition must be one of None, train or test')

        if partition is not None:
            if partition == 'train':
                self.character_list = self.train_list
            else:
                self.character_list = self.test_list
        else:
            # use the entire dataset
            self.character_list = self.train_list + self.test_list

        # weights for sampling
        if partition == 'train':
            train_labels = [character.get_label() for character in self.train_list]
            unique_character_labels, counts = np.unique(train_labels,
                                                        return_counts=True)
            ref_count_labels = {ucharacter: count for ucharacter, count
                                in zip(unique_character_labels, counts)}

        if partition == 'test':

            test_labels = [character.get_label() for character in self.test_list]

            unique_character_labels, counts = np.unique(test_labels,
                                                        return_counts=True)

            qry_character_labels = unique_character_labels[np.where(counts > 1)[0]]
            query_list = np.zeros(len(self.test_list), np.int8)
            qry_ids = [i for i in range(len(self.test_list))
                       if test_labels[i] in qry_character_labels]
            query_list[qry_ids] = 1

            self.query_list = query_list

        else:
            self.query_list = np.zeros(len(self.character_list), np.int8)


        # create queries
        if partition == 'query':

            test_labels = [character.get_label() for character in self.test_list]
            unique_character_labels, counts = np.unique(test_labels,
                                                        return_counts=True)
            print("[casia]>>>>>>>>>>>>> unique_character_labels")
            print(unique_character_labels)
            print('length of unique_character_labels: %d' % len(unique_character_labels))
            print("[casia]<<<<<<<<<<<<")

            fre_character_labels = unique_character_labels[np.where(counts > 1)[0]]
            print("[casia]>>>>>>>>>>>>> fre_character_labels")
            print(fre_character_labels)
            print('length of fre_character_labels: %d' % len(fre_character_labels))
            print("[casia]<<<<<<<<<<<<")

            tmp_fre_character_labels = fre_character_labels
            qry_list = []
            for char in self.test_list:
                # ensure that characters with different labels in qry_list
                if char.get_label() in tmp_fre_character_labels:
                    qry_list.append(char)
                    tmp_fre_character_labels = np.delete(tmp_fre_character_labels,
                                                         np.argwhere(fre_character_labels == char.get_label())
                                                         )

            print("[casia]>>>>>>>>>>>>> qry_list")
            print('length of qry_list: %d' % len(qry_list))
            print("[casia]<<<<<<<<<<<<")
            self.query_list = qry_list

    def generate_random_qry_list(self):
        test_labels = [character.get_label() for character in self.test_list]
        unique_character_labels, counts = np.unique(test_labels,
                                                    return_counts=True)
        print("[casia]>>>>>>>>>>>>> unique_character_labels")
        print('length of unique_character_labels: %d' % len(unique_character_labels))
        print(unique_character_labels)

        fre_character_labels = unique_character_labels[np.where(counts > 1)[0]]
        print("[casia]>>>>>>>>>>>>> fre_character_labels")
        print('length of fre_character_labels: %d' % len(fre_character_labels))
        print(fre_character_labels)

        tmp_fre_character_labels = fre_character_labels
        qry_list = []
        qry_class_id_list = []
        for char in self.test_list:
            char_label = char.get_label()

            # ensure that characters with different labels in qry_list
            if char_label in tmp_fre_character_labels:
                qry_list.append(char)
                qry_class_id_list.append(self.label_encoder.transform([char_label])[0])
                tmp_fre_character_labels = np.delete(tmp_fre_character_labels,
                                                     np.argwhere(fre_character_labels == char.get_label())
                                                     )
        self.random_query_list = qry_list
        self.random_query_class_id_list = qry_class_id_list
        print("[casia]>>>>>>>>>>>>> qry_list")
        print('length of qry_list: %d' % len(self.random_query_list))
        print("[casia]>>>>>>>>>>>>> query_class_id_list")
        print('length of query_class_id_list: %d' % len(self.random_query_class_id_list))

        # prepare data for torch
        self.random_query_list = conv_to_tensor(self.random_query_list)
        print(type(self.random_query_list))

        return self.random_query_list, self.random_query_class_id_list

    def display_character(self, character_tensor):
        character_numpy = character_tensor.numpy()
        plt.imshow(character_numpy, cmap='Greys')
        plt.show()

    def get_class_id(self, character_obj):
        return self.code_map[character_obj.get_label()]

    def get_class_ids_list(self):
        class_ids_list = [self.get_class_id(character_obj) for character_obj
                          in self.character_list]
        self.class_ids_list = list(set(class_ids_list))
        return self.class_ids_list

    def get_train_ids_list(self):
        train_ids_list = [self.get_class_id(train_character) for train_character
                          in self.train_list]
        return train_ids_list

    def get_train_ids_size(self):
        return len(set(self.get_train_ids_list()))

    def get_train_labels_size(self):
        return len(set([train_character.get_label() for train_character in self.train_list]))

    def get_test_ids_list(self):
        test_ids_list = [self.get_class_id(test_character) for test_character
                         in self.test_list]
        return test_ids_list

    def get_test_ids_size(self):
        return len(set(self.get_test_ids_list()))

    def lexicon(self):
        """
        returns the closed lexicon (train + test)
        """
        unique_characters = np.unique(self.label_encoder.classes_)

        class_ids = self.label_encoder.transform(unique_characters)

        return unique_characters, class_ids

    def get_code_map(self):
        return self.code_map

    def get_qry_size(self):
        return np.sum(list(map(lambda x: x == 1, self.query_list)))
        # return len(self.query_list)


if __name__ == '__main__':
    ds_train = CASIADataset(local_mode=True)
    ds_test = copy.copy(ds_train)
    ds_train.mainLoader(partition='train')
    ds_test.mainLoader(partition='test')

    train_labels = ds_train.get_train_labels_size()
    train_ids = ds_train.get_train_ids_size()
    assert ds_train.get_train_labels_size() == ds_train.get_train_ids_size()

    qry_list = ds_train.generate_random_qry_list()
    print('end')

    train_tensor, t_label, train_class_id, train_query = ds_train.__getitem__(5)
    print(train_tensor.shape)
    print(train_tensor)
    imshow(train_tensor)
    '''
    test_tensor, test_label, test_class_id, test_query = ds_test.__getitem__(5)
    print(test_label)
    print(test_class_id)
    imshow(test_tensor)
    '''
