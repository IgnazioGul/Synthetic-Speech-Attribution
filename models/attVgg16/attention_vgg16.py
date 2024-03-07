import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from models.attVgg16.attention_block import AttentionBlock


class AttentionVgg16(nn.Module):
    def __init__(self, num_classes, normalize_attn=False, dropout=None, pretrained=False, freezed: bool = True, *args,
                 **kwargs):
        super(AttentionVgg16, self).__init__()
        net = models.vgg16_bn(pretrained=pretrained)

        # Create a new first layer with 3 input channels
        self.conv_block0 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Initialize the new first layer with the pretrained weights from the original first layer
        self.conv_block0.weight.data = net.features[0].weight.data[:, 0:1, :, :].clone().sum(dim=1,
                                                                                             keepdim=True) / 3

        # Copy the batch normalization parameters from the original first layer
        self.conv_block0.bias.data = net.features[0].bias.data.clone()
        # self.conv_block0.weight.data[:, 1:, :, :] = 0  # Set weights for additional channels to zero

        # change img channels from 3 to 1
        # self.conv_block0 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_block1 = nn.Sequential(*list(net.features.children())[1:6])
        self.conv_block2 = nn.Sequential(*list(net.features.children())[7:13])
        self.conv_block3 = nn.Sequential(*list(net.features.children())[14:23])
        self.conv_block4 = nn.Sequential(*list(net.features.children())[24:33])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[34:43])
        if freezed:
            self.set_require_grad(self.conv_block0, False)
            self.set_require_grad(self.conv_block1, False)
            self.set_require_grad(self.conv_block2, False)
            self.set_require_grad(self.conv_block3, False)
            self.set_require_grad(self.conv_block4, False)
            self.set_require_grad(self.conv_block5, False)

        self.pool = nn.AvgPool2d(7, stride=1)
        self.dpt = None
        if dropout is not None:
            self.dpt = nn.Dropout(dropout)
        self.cls = nn.Linear(in_features=512 + 512 + 256, out_features=num_classes, bias=True)

        # initialize the attention attVgg16 defined above
        self.attn1 = AttentionBlock(256, 512, 256, 4, normalize_attn=normalize_attn)
        self.attn2 = AttentionBlock(512, 512, 256, 2, normalize_attn=normalize_attn)

        self.reset_parameters(self.cls)
        self.reset_parameters(self.attn1)
        self.reset_parameters(self.attn2)

    def reset_parameters(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., 0.01)
                nn.init.constant_(m.bias, 0.)

    def set_require_grad(self, layer, val: bool = False):
        for param in layer.parameters():
            param.requires_grad = val

    def forward(self, x):
        block0 = self.conv_block0(x)
        block1 = self.conv_block1(block0)  # /1
        pool1 = F.max_pool2d(block1, 2, 2)  # /2
        block2 = self.conv_block2(pool1)  # /2
        pool2 = F.max_pool2d(block2, 2, 2)  # /4
        block3 = self.conv_block3(pool2)  # /4
        pool3 = F.max_pool2d(block3, 2, 2)  # /8
        block4 = self.conv_block4(pool3)  # /8
        pool4 = F.max_pool2d(block4, 2, 2)  # /16
        block5 = self.conv_block5(pool4)  # /16
        pool5 = F.max_pool2d(block5, 2, 2)  # /32
        N, __, __, __ = pool5.size()

        g = self.pool(pool5).view(N, 512)
        a1, g1 = self.attn1(pool3, pool5)
        a2, g2 = self.attn2(pool4, pool5)
        g_hat = torch.cat((g, g1, g2), dim=1)  # batch_size x C
        if self.dpt is not None:
            g_hat = self.dpt(g_hat)
        out = self.cls(g_hat)

        return [out, a1, a2]
