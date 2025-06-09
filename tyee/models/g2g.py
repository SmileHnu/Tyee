import torch.nn as nn
import torch
import math
import random
import numpy as np
from einops import rearrange
import copy
from sklearn.metrics import mutual_info_score
from torch import Tensor

import torchvision.models as models
import logging


class RelationAwareness(nn.Module):
    def __init__(self, head_num, input_size, location_size, expand_size):
        super(RelationAwareness, self).__init__()

        self.head = head_num
        self.input_size = input_size # eeg input size on each electrode, 5
        self.location_size = location_size # 3
        self.expand_size = expand_size # expand eeg, eye, and location to the same dimen, 10

        self.location_em = nn.Linear(self.location_size, self.head*self.expand_size) # 3 --> 6*10
        self.data_em = nn.Linear(self.input_size, self.head*self.expand_size) # 5 --> 6*10
        self.eye_em = nn.Linear(10, self.head*self.expand_size) # 10 --> 6*10
        self.relu = nn.ReLU()

        self.a = nn.Parameter(torch.empty(size=(2*self.expand_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, feature, location, eye):

        feature_embed = self.data_em(feature)
        location_embed = self.location_em(location)
        feature_local_embed = self.relu(feature_embed + location_embed)

        eye_embed = self.relu(self.eye_em(eye))
        eeg_eye_embed = torch.cat([feature_local_embed, eye_embed], 1)

        feature_ = rearrange(eeg_eye_embed, "b n (h d) -> b h n d", h=self.head)

        two_d_feature = self.cal_att_matrix(feature_)
        return two_d_feature

    def cal_att_matrix(self, feature):

        data = []
        batch_size, head,  N = feature.size(0), feature.size(1), feature.size(2)
        Wh1 = torch.matmul(feature, self.a[:self.expand_size, :])
        Wh2 = torch.matmul(feature, self.a[self.expand_size:, :])
        # broadcast add
        Wh2_T = rearrange(Wh2, "b n h d -> b n d h")
        e = Wh1 + Wh2_T
        return e


class ConvNet(nn.Module):
    def __init__(self, emb_size, cifar_flag=False):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 128
        self.last_hidden = self.hidden * 16 if not cifar_flag else self.hidden
        # self.last_hidden = self.hidden * 1 if not cifar_flag else self.hidden
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=12,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.max = nn.MaxPool2d(kernel_size=2)

        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_second = nn.Sequential(nn.Linear(in_features=self.last_hidden * 2 ,
                                                    out_features=self.emb_size, bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4 ,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        output_data = self.conv_4(out_3)
        output_data0 = self.max(out_3)
        out1 = self.layer_last(output_data.view(output_data.size(0), -1))
        out2 = self.layer_second(output_data0.view(output_data0.size(0), -1))

        out = torch.cat((out1, out2), dim=1)  # (batch_size, 256)

        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.resnet50(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class EncoderNet(nn.Module):
    def __init__(self, head_num=6, rand_ali_num=2, backbone="ResNet18", device="cuda:0",
                 input_size=5, location_size=3, expand_size=10, eeg_node_num=62, num_class=5, sup_node_num=6):
        super(EncoderNet, self).__init__()
        logger = logging.getLogger("model")
        self.resnet_embed = 256
        self.backbone_output =  self.resnet_embed * 2
        self.rand_ali_num = rand_ali_num
        self.sup_node_num = sup_node_num

        self.relationAwareness = RelationAwareness(head_num=head_num, input_size=input_size, 
                                                   location_size=location_size, expand_size=expand_size)
        self.rand_order = random_1D_node(rand_ali_num, eeg_node_num)
        print(self.rand_order)

        # define selected backbone
        self.backbone = None
        if backbone == "ConvNet":
            self.backbone =  ConvNet(self.resnet_embed)
        elif backbone == "ResNet18":
            self.backbone = ResNet18()
        elif backbone == "ResNet50":
            self.backbone = ResNet50()
        else:
            raise RuntimeError("Wrong backbone!")

        # get node location
        self.location = torch.from_numpy(return_coordinates()).to(device)

        self.l_relu = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm1d(self.backbone_output)
        self.bn_2D = nn.BatchNorm2d(12)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

        self.mlp_0 = nn.Linear(512, self.backbone_output)
        self.mlp_1 = nn.Linear(self.backbone_output, num_class)


    def forward(self, x):
        ran_list = []
        for index in range(self.rand_ali_num):
            x_eeg = x[:, :310]
            x_eye = x[:, 310:380]

            x_eeg = rearrange(x_eeg, 'b (h c) -> b h c', h=62) #(32,62,5)
            x_eye = rearrange(x_eye, 'b (h c) -> b h c', h=self.sup_node_num) #(32,6,10)

            x_random, coor_random = x_eeg[:, self.rand_order[index], :], self.location[self.rand_order[index], :]
            x_ = self.relationAwareness(x_random, coor_random, x_eye) # (batch_size, 62, 62, 3)

            ran_list.append(x_)

        x_ = torch.cat(tuple(ran_list), 1)  # (batch_size, self.rand_ali_num*self.head, N, N)
        x_ = self.bn_2D(x_)

        output = self.backbone(x_)

        x = self.dropout(output)
        x = self.mlp_0(x)
        x = self.l_relu(x)
        x = self.bn(x)
        x = self.mlp_1(x)

        return x


def return_coordinates():
    """
    Node location for SEED, SEED4, SEED5, MPED
    """
    m1 = [(-2.285379, 10.372299, 4.564709),
          (0.687462, 10.931931, 4.452579),
          (3.874373, 9.896583, 4.368097),
          (-2.82271, 9.895013, 6.833403),
          (4.143959, 9.607678, 7.067061),

          (-6.417786, 6.362997, 4.476012),
          (-5.745505, 7.282387, 6.764246),
          (-4.248579, 7.990933, 8.73188),
          (-2.046628, 8.049909, 10.162745),
          (0.716282, 7.836015, 10.88362),
          (3.193455, 7.889754, 10.312743),
          (5.337832, 7.691511, 8.678795),
          (6.842302, 6.643506, 6.300108),
          (7.197982, 5.671902, 4.245699),

          (-7.326021, 3.749974, 4.734323),
          (-6.882368, 4.211114, 7.939393),
          (-4.837038, 4.672796, 10.955297),
          (-2.677567, 4.478631, 12.365311),
          (0.455027, 4.186858, 13.104445),
          (3.654295, 4.254963, 12.205945),
          (5.863695, 4.275586, 10.714709),
          (7.610693, 3.851083, 7.604854),
          (7.821661, 3.18878, 4.400032),

          (-7.640498, 0.756314, 4.967095),
          (-7.230136, 0.725585, 8.331517),
          (-5.748005, 0.480691, 11.193904),
          (-3.009834, 0.621885, 13.441012),
          (0.341982, 0.449246, 13.839247),
          (3.62126, 0.31676, 13.082255),
          (6.418348, 0.200262, 11.178412),
          (7.743287, 0.254288, 8.143276),
          (8.214926, 0.533799, 4.980188),

          (-7.794727, -1.924366, 4.686678),
          (-7.103159, -2.735806, 7.908936),
          (-5.549734, -3.131109, 10.995642),
          (-3.111164, -3.281632, 12.904391),
          (-0.072857, -3.405421, 13.509398),
          (3.044321, -3.820854, 12.781214),
          (5.712892, -3.643826, 10.907982),
          (7.304755, -3.111501, 7.913397),
          (7.92715, -2.443219, 4.673271),

          (-7.161848, -4.799244, 4.411572),
          (-6.375708, -5.683398, 7.142764),
          (-5.117089, -6.324777, 9.046002),
          (-2.8246, -6.605847, 10.717917),
          (-0.19569, -6.696784, 11.505725),
          (2.396374, -7.077637, 10.585553),
          (4.802065, -6.824497, 8.991351),
          (6.172683, -6.209247, 7.028114),
          (7.187716, -4.954237, 4.477674),

          (-5.894369, -6.974203, 4.318362),
          (-5.037746, -7.566237, 6.585544),
          (-2.544662, -8.415612, 7.820205),
          (-0.339835, -8.716856, 8.249729),
          (2.201964, -8.66148, 7.796194),
          (4.491326, -8.16103, 6.387415),
          (5.766648, -7.498684, 4.546538),

          (-6.387065, -5.755497, 1.886141),
          (-3.542601, -8.904578, 4.214279),
          (-0.080624, -9.660508, 4.670766),
          (3.050584, -9.25965, 4.194428),
          (6.192229, -6.797348, 2.355135),
          ]

    m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1))
    m1 = np.float32(np.array(m1))
    return m1


def random_1D_node(num, node_num):

    rand_lists = []
    for index in range(num):
        grand_list = [i for i in range(node_num)]
        random.shuffle(grand_list)
        rand_tensor = torch.tensor(grand_list).view(1, node_num)
        rand_lists.append(rand_tensor)

    rand_torch = torch.cat(tuple(rand_lists), 0)
    return rand_torch