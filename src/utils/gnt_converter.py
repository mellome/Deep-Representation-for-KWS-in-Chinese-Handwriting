"""
Created on May 26, 2020

@author: yhe

"""
import os
import struct
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image
from skimage.transform import resize


def save_to_npy(images, labels, dst_file):
    np.save(dst_file + 'casia_characters.npy', images)
    np.save(dst_file + 'casia_labels.npy', labels)


def imshow(character_tensor):
    character_numpy = character_tensor.numpy()
    character_numpy = character_numpy.reshape(character_numpy.shape[1:])
    plt.imshow(character_numpy, cmap='Greys')
    plt.show()


# extract all characters from the given gnt file
def read_Gnt(data_file, image_size, total_bytes, code_map):
    decoded_bytes = 0
    image_list = []
    label_list = []
    new_label = len(code_map)
    while decoded_bytes != total_bytes:
        data_length, = struct.unpack('<I', data_file.read(4))
        tag_code, = struct.unpack('>H', data_file.read(2))
        image_width, = struct.unpack('<H', data_file.read(2))
        image_height, = struct.unpack('<H', data_file.read(2))
        arc_length = image_width
        if image_width < image_height:
            arc_length = image_height
        temp_image = 255 * np.ones((arc_length, arc_length, 1), np.uint8)
        row_begin = (arc_length - image_height) // 2
        col_begin = (arc_length - image_width) // 2
        for row in range(row_begin, image_height + row_begin):
            for col in range(col_begin, image_width + col_begin):
                temp_image[row, col], = struct.unpack('B', data_file.read(1))
        decoded_bytes += data_length
        result_image = cv2.resize(temp_image, (image_size, image_size))  # type(result_image) : ndarray
        if tag_code not in code_map:
            code_map[tag_code] = new_label
            new_label += 1
        image_list.append(result_image)
        label_list.append(tag_code)
        # label_list.append(code_map[tag_code])
    return image_list, label_list, code_map


# extract all characters from the given gnt file
def read_gnt(data_file, code_map):
    image_list = []
    label_list = []
    new_label = len(code_map)
    while True:
        packed_length = data_file.read(4)
        if packed_length == b'': break
        length = struct.unpack("<I", packed_length)[0]
        # raw_label = struct.unpack(">cc", data_file.read(2))
        tag_code = struct.unpack('>H', data_file.read(2))[0]
        width = struct.unpack("<H", data_file.read(2))[0]
        height = struct.unpack("<H", data_file.read(2))[0]
        photo_bytes = struct.unpack("{}B".format(height * width), data_file.read(height * width))

        # Create an array of bytes for the image, match it to the proper dimensions, and turn it into an image.
        image = np.float32(np.array(photo_bytes)).reshape(height, width)

        try:
            image = cv2.resize(image, (32, 32))
            if tag_code not in code_map:
                code_map[tag_code] = new_label
                new_label += 1
            image_list.append(image)
            label_list.append(tag_code)

        except Exception as e:
            print("========= Wrong Image: " + data_file.name)
            print(str(e))

    return image_list, label_list


# extract all characters from the given zip file
def load_casia_zip(gnt_folder_dir, dst_file, map_file):
    code_map = {}
    if os.path.exists(map_file):
        with open(map_file, 'r') as fp:
            for line in fp.readlines():
                if len(line) == 0:
                    continue;
                code, label = line.split()
                code_map[int(code)] = int(label)
        fp.close()

    images = []
    labels = []

    for root_path, dir_list, _ in os.walk(gnt_folder_dir):  # folder, folder, folder...

        for next_dir_name in dir_list:  # folder
            gnt_folder_dir = os.path.join(root_path, next_dir_name)

            for file_name in os.listdir(gnt_folder_dir):  # <name>.gnt, <name>.gnt, <name>.gnt...

                if file_name.endswith('.zip'):
                    zip_file_dir = gnt_folder_dir + '/' + file_name
                    unzipped_file = zipfile.ZipFile(zip_file_dir, 'r')
                    unzipped_file_list = unzipped_file.namelist()
                    for gnt_file_name in unzipped_file_list:
                        print("processing %s ..." % gnt_file_name)
                        total_bytes = unzipped_file.getinfo(gnt_file_name).file_size
                        gnt_file = unzipped_file.open(gnt_file_name)
                        image_list, label_list, code_map = read_Gnt(gnt_file, 64, total_bytes,
                                                                    code_map)  # image_size=64
                        images += image_list
                        labels += label_list
                        break  # only for test

                if file_name.endswith('.gnt'):
                    gnt_file_dir = os.path.join(gnt_folder_dir, file_name)
                    with open(gnt_file_dir, 'rb') as f:
                        print("processing %s ..." % file_name)
                        total_bytes = os.path.getsize(os.path.join(gnt_folder_dir, file_name))
                        image_list, label_list, code_map = read_Gnt(f, 64, total_bytes, code_map)
                        images += image_list
                        labels += label_list

    # save_to_npy(images, labels, dst_file)
    return images, labels, code_map


# extract all characters from the given gnt file
def load_gnt(gnt_path, gnt_id, code_map):
    if code_map is None:
        code_map = {}

    with open(gnt_path, 'rb') as f:
        print("processing %s ..." % gnt_id)
        image_list, label_list = read_gnt(f, code_map)
        return image_list, label_list


def normalize_bitmap(bitmap):
    # pad the bitmap to make it squared
    pad_size = abs(bitmap.shape[0] - bitmap.shape[1]) // 2
    if bitmap.shape[0] < bitmap.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    bitmap = np.lib.pad(bitmap, pad_dims, mode='constant', constant_values=255)

    # rescale and add empty border
    bitmap = resize(bitmap, (64 - 4 * 2, 64 - 4 * 2))
    bitmap = np.lib.pad(bitmap, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert bitmap.shape == (64, 64)

    return bitmap


def preprocess_bitmap(bitmap):
    # contrast stretching
    p2, p98 = np.percentile(bitmap, (2, 98))
    assert abs(p2 - p98) > 10
    bitmap = skimage.exposure.rescale_intensity(bitmap, in_range=(p2, p98))

    return bitmap


def tagcode_to_unicode(tagcode):
    return struct.pack('>H', tagcode).decode('gb2312')


def unicode_to_tagcode(tagcode_unicode):
    return struct.unpack('>H', tagcode_unicode.encode('gb2312'))[0]


if __name__ == '__main__':

    def save_character_img(images):
        for i in range(0, len(images)):
            img = Image.fromarray(images[i])
            dir_name = png_dir + '/' + '%0.5d' % labels[i]
            # if not os.path.exists(dir_name):
            #    os.mkdir(dir_name)
            img.convert('RGB').save(dir_name + '.png')
            break


    # local path
    zip_dir = '/Users/mellome1992/Documents/LocalRepository/phocnet_kws/src/gnt_utils/dataset'
    dst_dir = '/Users/mellome1992/Documents/LocalRepository/phocnet_kws/experiments/seg_based/character_set'
    png_dir = '/Users/mellome1992/Documents/LocalRepository/phocnet_kws/src/gnt_utils/dataset/train_sets'

    images, labels, code_map = load_casia_zip(zip_dir, dst_dir, 'code_map')

    for imgs in images:
        plt.imshow(imgs)
        plt.show()
        break

    '''
    print('length of images %d' % len(images))
    print('length of labels %d' % len(labels))
    print('length of code_map %d' % len(code_map))
    print(code_map)

    for imgs in images:
        plt.imshow(imgs)
        plt.show()
        #break
    '''
