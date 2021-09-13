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

from __future__ import print_function
import torch
# print(torch.__version__)
import torch.utils.data
import torchvision.transforms as transforms
from basic_code import data_generator

cate2label = {'CK+':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6}}

def ckplus_faces_fan(video_root, video_list, fold, batchsize_train, batchsize_eval):

    train_dataset = data_generator.TenFold_TripleImageDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([
                                            transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='train',
                                        )

    val_dataset = data_generator.TenFold_VideoDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='test'
                                        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize_train, shuffle=True, num_workers=2,pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batchsize_eval, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader

def model_parameters(_structure, _parameterDir):
    # import sys

    # orig_stdout = sys.stdout
    # f = open('out.txt', 'w')
    # sys.stdout = f

    # print(_structure)
    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    # print("#############################################")
    # print(model_state_dict)
    # print("#############################################")
    for key in pretrained_state_dict:
        print(key)
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias') | (key.split('.')[1] == 'layer4')):
            # print(key)
            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    # print("###################################################################")
    # print(model_state_dict)

    # sys.stdout = orig_stdout
    # f.close()
    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model
