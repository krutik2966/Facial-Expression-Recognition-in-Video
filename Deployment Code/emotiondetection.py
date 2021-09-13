# import the necessary packages
import numpy as np
import imutils
import cv2
from torch_utils import get_prediction
import torch
from torchvision.transforms import transforms
import json


class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=21)


cate2label = {
    0: "Happy",
    1: "Angry",
    2: "Disgust",
    3: "Fear",
    4: "Sad",
    5: "Contempt",
    6: "Surprise",
}


class EmotionDetector:
    def __init__(self, frame_count):
        self.frame_count = frame_count
        self.image_list = torch.tensor([])
        self.trans = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    def update(self, image, total):
        # if the background model is None, initialize it
        if total > self.frame_count:
            t = self.image_list[1:, :, :, :]
            img_tensor = self.trans(image)
            img_tensor = img_tensor.view((1, 3, 224, 224))
            new_t = torch.cat((t, img_tensor), axis=0)
            self.image_list = new_t
        else:
            t = self.image_list
            img_tensor = self.trans(image)
            img_tensor = img_tensor.view((1, 3, 224, 224))
            new_t = torch.cat((t, img_tensor), axis=0)
            self.image_list = new_t

        return

    def detect(self):
        pred = get_prediction(self.image_list, self.frame_count)

        try:
            pred = pred.squeeze(0).tolist()
            pred_dict = []
            for i in range(7):
                pred_dict.append(cate2label[i])
                pred_dict.append(pred[i])

            # pred_dict = {
            #     0: {"name": cate2label[0], "value": pred[0]},
            #     1: {"name": cate2label[1], "value": pred[1]},
            #     2: {"name": cate2label[2], "value": pred[2]},
            #     3: {"name": cate2label[3], "value": pred[3]},
            #     4: {"name": cate2label[4], "value": pred[4]},
            #     5: {"name": cate2label[5], "value": pred[5]},
            #     6: {"name": cate2label[6], "value": pred[6]},
            # }

            # me = Object()

            # me.a = Object()
            # me.a.name = cate2label[0]
            # me.a.value = pred[0]

            # me.b = Object()
            # me.b.name = cate2label[1]
            # me.b.value = pred[1]

            # me.c = Object()
            # me.c.name = cate2label[2]
            # me.c.value = pred[2]

            # me.d = Object()
            # me.d.name = cate2label[3]
            # me.d.value = pred[3]
            # me.e = Object()
            # me.e.name = cate2label[4]
            # me.e.value = pred[4]
            # me.f = Object()
            # me.f.name = cate2label[5]
            # me.f.value = pred[5]

            # print("Pred dict :", pred_dict)
            # print("Type", type(pred_dict))
            return pred_dict
        except:
            return -1
