# # Emotion-FAN.pytorch
#  ICIP 2019: Frame Attention Networks for Facial Expression Recognition in Videos  [pdf](https://arxiv.org/pdf/1907.00193.pdf)
 
#  [Debin Meng](michaeldbmeng19@outlook.com), [Xiaojiang Peng](https://pengxj.github.io/), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), etc.

# ## Citation
# If you are using pieces of the posted code, please cite the above paper. thanks.:
# ```
# @inproceedings{meng2019frame,
#   title={frame attention networks for facial expression recognition in videos},
#   author={Meng, Debin and Peng, Xiaojiang and Wang, Kai and Qiao, Yu},
#   booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
#   pages={3866--3870},
#   year={2019},
#   organization={IEEE},
#   url={https://github.com/Open-Debin/Emotion-FAN}
# }
# ```

#coding=utf-8
import pdb
import os, sys, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

## data generator for ck_plus
class TenFold_VideoDataset(data.Dataset):
    def __init__(self, video_root='', video_list='', rectify_label=None, transform=None, fold=1, run_type='train'):
        self.imgs_first, self.index = load_imgs_tenfold_totalframe(video_root, video_list, rectify_label, fold, run_type)

        self.transform = transform
        self.video_root = video_root

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]

        img_first = Image.open(path_first).convert('RGB')

        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)

class TenFold_TripleImageDataset(data.Dataset):
    def __init__(self, video_root='', video_list='', rectify_label=None, transform=None, fold=1, run_type='train'):

        self.imgs_first, self.imgs_second, self.imgs_third, self.index = load_imgs_tsn_tenfold(video_root,video_list,rectify_label, fold, run_type)

        self.transform = transform
        self.video_root = video_root

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]

        img_first = Image.open(path_first).convert("RGB")

        if self.transform is not None:
            img_first = self.transform(img_first)

        path_second, target_second = self.imgs_second[index]
        
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)

        return img_first, img_second, img_third, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)


def load_imgs_tenfold_totalframe(video_root, video_list, rectify_label, fold, run_type):

    #video root : root path which is ./data/face/ck_face
    #video_list : List of the videos which is ./data/txt/CK+_10-fold_sample_IDascendorder_step10.txt
    #rectify_label : It is a dictionary which is used for encoding emotions from text to int and from int to text
    # fold : It gives a fold which we have to keep as validation data and remaining folds are used as training data
    # run_type : shows training or testing

    imgs_first = list()
    new_imf = list()

    ''' Make ten-fold list '''
    with open(video_list, 'r') as imf:

        #reading the lines from the .txt file and storing it into 'imf'
        imf = imf.readlines()

    #for training
    if run_type == 'train':

        fold_ = list(range(1, 11))  #This creates a list [1,2,3,4,5,6,7,8,9,10]
        fold_.remove(fold)  # [1,2,3,4,5,6,7,8,9, 10] -> [2,3,4,5,6,7,8,9,10] : removes the test fold number from the list

        #iterating over all training folds
        for i in fold_:
            fold_str = str(i) + '-fold'  # 1-fold

            for index, item in enumerate(imf):  # 0, '1-fold\t31\n' in {[0, '1-fold\t31\n'], [1, 'S037/006 Happy\n'], ...}

                if fold_str in item:  # 1-fold in '1-fold\t31\n'

                    #here "index" is for starting line of each fold, it shows the starting line number
                    #We are collecting the videos' paths and emotions which are there in the particular fold and appending them to new_imf

                    for j in range(index + 1, index + int(item.split()[1]) + 1):  # (0 + 1, 0 + 31 + 1 )
                        new_imf.append(imf[j])  # imf[2] = 'S042/006 Happy\n'

    #The same process happens in test also but here we are getting testing fold's data instead of all training folds'.
    if run_type == 'test':
        fold_ = fold    #testing fold
        fold_str = str(fold_) + '-fold'

        for index, item in enumerate(imf):
            if fold_str in item:

                for j in range(index + 1, index + int(item.split()[1]) + 1):
                    new_imf.append(imf[j])

    index = []

    #We have created a list 'new_imf' which has video path and it's label.
    for id, line in enumerate(new_imf):

        video_label = line.strip().split()

        video_name = video_label[0]  # name of video
        try:
            label = rectify_label[video_label[1]]  # label of video
        except:
            pdb.set_trace()

        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video


        ###  for sampling triple imgs in the single video_path  ####
        img_lists = os.listdir(video_path)
        img_lists.sort()  # sort files by ascending
        
        img_lists = img_lists[ - int(round(len(img_lists))) : ]

        img_count = len(img_lists)  # number of frames in video

        for frame in img_lists:
            imgs_first.append((os.path.join(video_path, frame), label))

        ###  return video frame index  #####
        index.append(np.ones(img_count) * id)

    index = np.concatenate(index, axis=0)

    #returning the frames and index according to each frame. 
    return imgs_first, index

def load_imgs_tsn_tenfold(video_root, video_list, rectify_label, fold, run_type):

    # video root : root path which is ./data/face/ck_face
    # video_list : List of the videos which is ./data/txt/CK+_10-fold_sample_IDascendorder_step10.txt
    # rectify_label : Itis a dictionary which is used for encoding emotions from text to int and from int to text
    # fold : It gives a fold which we have to keep as validation data and remaining folds are used as training data
    # run_type : shows training or testing

    imgs_first = list()
    imgs_second = list()
    imgs_third = list()
    new_imf = list()

    ''' Make ten-fold list '''
    with open(video_list, 'r') as imf:
        imf = imf.readlines()

    if run_type == 'train':
        fold_ = list(range(1, 11))

        #we are removing the test fold from the list
        fold_.remove(fold)  # [1,2,3,4,5,6,7,8,9,10] -> [2,3,4,5,6,7,8,9,10]

        #we are iterating over the train folds
        for i in fold_:
            fold_str = str(i) + '-fold'  # 1-fold

            for index, item in enumerate(imf):  # 0, '1-fold\t31\n' in {[0, '1-fold\t31\n'], [1, 'S037/006 Happy\n'], ...}
                
                #if the line is fold's strating line
                if fold_str in item:  # 1-fold in '1-fold\t31\n'

                    #then we are appending all image names and labels whcih are in that fold into new_imf
                    for j in range(index + 1, index + int(item.split()[1]) + 1):  # (0 + 1, 0 + 31 + 1 )
                        new_imf.append(imf[j])  # imf[2] = 'S042/006 Happy\n'

    if run_type == 'test':
        fold_ = fold

        #here we are adding all test fold's images to new_imf

        fold_str = str(fold_) + '-fold'
        for index, item in enumerate(imf):
            if fold_str in item:
                for j in range(index + 1, index + int(item.split()[1]) + 1):
                    new_imf.append(imf[j])


    ''' Make triple-image list '''
    index = []
    for id, line in enumerate(new_imf):


        video_label = line.strip().split()  
        video_name = video_label[0]  # name of video
        label = rectify_label[video_label[1]]  # label of video
        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video


        ###  for sampling triple imgs in the single video_path  ####
        img_lists = os.listdir(video_path)
        img_lists.sort()  # sort files by ascending

        img_lists = img_lists[ - int(round(len(img_lists))):]

        img_count = len(img_lists)  # number of frames in video

        num_per_part = int(img_count) // 5

        if int(img_count) > 5:

            #loop runs for nummber of frmaes times
            for i in range(img_count):
                # pdb.set_trace()
            
                random_select_first = random.randint(0, num_per_part)
                random_select_second = random.randint(num_per_part, 2 * num_per_part)
                random_select_third = random.randint(2 * num_per_part, 3 * num_per_part)

                img_path_first = os.path.join(video_path, img_lists[random_select_first])
                img_path_second = os.path.join(video_path, img_lists[random_select_second])
                img_path_third = os.path.join(video_path, img_lists[random_select_third])

                imgs_first.append((img_path_first, label))
                imgs_second.append((img_path_second, label))
                imgs_third.append((img_path_third, label))

        else:
            for j in range(len(img_lists)):
                img_path_first = os.path.join(video_path, img_lists[j])
                img_path_second = os.path.join(video_path, random.choice(img_lists))
                img_path_third = os.path.join(video_path, random.choice(img_lists))

                imgs_first.append((img_path_first, label))
                imgs_second.append((img_path_second, label))
                imgs_third.append((img_path_third, label))

        ###  return video frame index  #####
        index.append(np.ones(img_count) * id)  # id: 0 : 379
    index = np.concatenate(index, axis=0)
    # index = index.astype(int)
    # pdb.set_trace()
    return imgs_first, imgs_second, imgs_third, index

