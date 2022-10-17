
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


def _SEbn_function_factory(se, norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(se(concated_features))))
        return bottleneck_output

    return bn_function


class conv_layer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(conv_layer, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_features, num_output_features, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_output_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

    def forward(self, x):
        return self.conv(x)


class _3Dconv_layer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_3Dconv_layer, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(num_input_features, num_output_features, kernel_size=(3,3,3), stride=(1,1,1),
                                padding=(1,1,1), bias=False)),
            ('norm0', nn.BatchNorm3d(num_output_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.conv(x)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _3D_DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_3D_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _3D_DenseSELayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_3D_DenseSELayer, self).__init__()
        self.add_module('se', SELayer3D(channel=num_input_features))
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _SEbn_function_factory(self.se, self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _3D_DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_3D_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _3D_DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _3D_DenseSEBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_3D_DenseSEBlock, self).__init__()
        for i in range(num_layers):
            layer = _3D_DenseSELayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=(1,1), stride=(1,1), bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _3D_Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_3D_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))



class _3D_D_Transition(nn.Sequential):
    '''
    This transition only downsize D channel.
    '''
    def __init__(self, num_input_features, num_output_features):
        super(_3D_D_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=(1,1,1), stride=(1,1,1), bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(2,1,1), stride=(2,1,1)))



def extract_loc_in_CAM(cam):
    maxv = torch.max(cam)
    if maxv < 0:
        return
    else:
        cam = cam/maxv


class DenseNet(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 8, 8, 6),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(DenseNet, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.norm5 = nn.BatchNorm2d(num_features)

        # Linear layer
        self.linear = nn.Linear(num_features, num_classes)
        self.sig = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.block3(self.trans3(out_block2))
        out_norm5 = self.norm5(out_block3)
        out = F.relu(out_norm5, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        linear_weight = self.linear.weight
        linear_bias = self.linear.bias
        B, C, H, W = out_norm5.shape
        cam = torch.matmul(linear_weight, out_norm5.view(B, C, H * W)).view(B, 2, H, W) + linear_bias.view(1, -1,1,1)
        return out, cam



class DenseBlock0123(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 8, 8, 6),
                 num_init_features=64, num_final_features=256, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(DenseBlock0123, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.conv4 = nn.Conv2d(num_features, num_final_features, kernel_size=(1, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        self.norm5 = nn.BatchNorm2d(num_final_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.conv4(self.block3(self.trans3(out_block2)))
        out_norm5 = self.norm5(out_block3)
        return out_norm5



class DenseBlock012(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 8, 8, 6),
                 num_init_features=64, num_final_features=256, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(DenseBlock012, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _Transition(num_features, num_final_features)
        self.norm5 = nn.BatchNorm2d(num_final_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.trans3(out_block2)
        out_norm5 = self.norm5(out_block3)
        return out_norm5



class _3D_DenseNet8(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D_DenseNet8, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(3,7,7), stride=(1, 2, 2),
                                padding=(3,3,3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _3D_DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _3D_DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _3D_DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _3D_DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.norm5 = nn.BatchNorm3d(num_features)

        # Linear layer
        self.linear = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.block3(self.trans3(out_block2))
        out_norm5 = self.norm5(out_block3)
        out = F.relu(out_norm5, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        linear_weight = self.linear.weight
        linear_bias = self.linear.bias
        B, C, D, H, W = out_norm5.shape
        cam = torch.matmul(linear_weight, out_norm5.view(B, C, D * H * W)).view(B, 2, H, W) + linear_bias.view(1,-1,1, 1)
        return out, cam




class _3D_DenseNet(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D_DenseNet16, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(7,7,7), stride=(1, 2, 2),
                                padding=(3,3,3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _3D_DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _3D_DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _3D_DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _3D_DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.norm5 = nn.BatchNorm3d(num_features)

        # Linear layer
        self.linear = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.block3(self.trans3(out_block2))
        out_norm5 = self.norm5(out_block3)
        out = F.relu(out_norm5, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        linear_weight = self.linear.weight
        linear_bias = self.linear.bias
        B, C, D, H, W = out_norm5.shape
        cam = torch.matmul(linear_weight, out_norm5.view(B, C, D * H * W)).view(B, 2, H, W) + linear_bias.view(1,-1,1, 1)
        return out, cam



class _3D_DenseNet16_outfeat(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D_DenseNet16_outfeat, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(7,7,7), stride=(1, 2, 2),
                                padding=(3,3,3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _3D_DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _3D_DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _3D_DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _3D_DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.norm5 = nn.BatchNorm3d(num_features)

        # Linear layer
        self.linear = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.block3(self.trans3(out_block2))
        out_norm5 = self.norm5(out_block3)
        out = F.relu(out_norm5, inplace=True)
        out_gap = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out_gap = torch.flatten(out_gap, 1)
        out = self.linear(out_gap)
        linear_weight = self.linear.weight
        linear_bias = self.linear.bias
        B, C, D, H, W = out_norm5.shape
        cam = torch.matmul(linear_weight, out_norm5.view(B, C, D * H * W)).view(B, 2, H, W) + linear_bias.view(1,-1,1, 1)
        return out, cam, out_gap


class fc_layer(nn.Module):
    def __init__(self, num_features, num_output=2):
        super(fc_layer, self).__init__()
        self.linear1 = nn.Linear(num_features, num_features//16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_features//16, num_features//16)
        self.linear3 = nn.Linear(num_features//16, num_output)
    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.linear3(y)
        return  y





class _3D_DenseNet32(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D_DenseNet32, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(7,7,7), stride=(2, 2, 2),
                                padding=(3,3,3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _3D_DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _3D_DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _3D_DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _3D_DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.norm5 = nn.BatchNorm3d(num_features)

        # Linear layer
        self.linear = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.block3(self.trans3(out_block2))
        out_norm5 = self.norm5(out_block3)
        out = F.relu(out_norm5, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        linear_weight = self.linear.weight
        linear_bias = self.linear.bias
        B, C, D, H, W = out_norm5.shape
        cam = torch.matmul(linear_weight, out_norm5.view(B, C, D * H * W)).view(B, 2, H, W) + linear_bias.view(1,-1,1, 1)
        return out, cam



class _3D_DenseBlock0123(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4),
                 num_init_features=64, num_final_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D_DenseBlock0123, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(7,7,7), stride=(1, 2, 2),
                                padding=(3,3,3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _3D_DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _3D_DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _3D_DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _3D_DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.conv4 = nn.Conv3d(num_features, num_final_features, kernel_size=(1,1,1), padding=(0,0,0))
        self.norm5 = nn.BatchNorm3d(num_final_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.conv4(self.block3(self.trans3(out_block2)))
        out_norm5 = self.norm5(out_block3)
        return out_norm5



class _3D_DenseBlock012(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4),
                 num_init_features=64, num_final_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D_DenseBlock012, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(7,7,7), stride=(1, 2, 2),
                                padding=(3,3,3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _3D_DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _3D_DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _3D_DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _3D_Transition(num_input_features=num_features, num_output_features=num_final_features)
        self.norm5 = nn.BatchNorm3d(num_final_features)


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.trans3(out_block2)
        out_norm4 = self.norm5(out_block3)
        return out_norm4



class _3D2D_2branch_DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config2D=(4, 4, 4, 4),block_config3D=(4, 4, 4, 4),
                 num_init_features=64, num_final_features = 256, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D2D_2branch_DenseNet, self).__init__()
        self._2Dnet = DenseBlock0123(block_config=block_config2D, num_init_features=num_init_features,
                                     num_final_features = num_final_features, drop_rate=drop_rate)
        self._3Dnet = _3D_DenseBlock0123(block_config=block_config3D, num_init_features=num_init_features,
                                         num_final_features = num_final_features, drop_rate=drop_rate)
        self.linear2D = nn.Linear(num_final_features, num_classes)
        self.linear3D = nn.Linear(num_final_features, num_classes)
        self.linear2D3D = nn.Linear(num_final_features*2, num_classes)

    def forward(self, x2d, x3d):
        feat2D = self._2Dnet(x2d)
        out2D = F.relu(feat2D, inplace=True)
        out2D_pool = F.adaptive_avg_pool2d(out2D, (1, 1))
        out2D = torch.flatten(out2D_pool, 1)
        out2D = self.linear2D(out2D)
        linear_weight = self.linear2D.weight
        linear_bias = self.linear2D.bias
        B, C, H, W = feat2D.shape
        cam2D = torch.matmul(linear_weight, feat2D.view(B, C, H * W)).view(B, 2, H, W) + linear_bias.view(1,-1,1,1)

        feat3D = self._3Dnet(x3d)
        out3D = F.relu(feat3D, inplace=True)
        out3D_pool = F.adaptive_avg_pool3d(out3D, (1, 1, 1))
        out3D = torch.flatten(out3D_pool, 1)
        out3D = self.linear3D(out3D)
        linear_weight = self.linear3D.weight
        linear_bias = self.linear3D.bias
        B, C, D, H, W = feat3D.shape
        cam3D = torch.matmul(linear_weight, feat3D.view(B, C, H * W)).view(B, 2, H, W) + linear_bias.view(1, -1, 1, 1)

        feat2D3D = torch.cat([feat2D, feat3D.view(B, C, H, W)], 1)
        out2D3D = torch.cat([out2D_pool, out3D_pool[:, :, :, :,0]], 1)
        out2D3D = torch.flatten(out2D3D, 1)
        out2D3D = self.linear2D3D(out2D3D)
        linear_weight = self.linear2D3D.weight
        linear_bias = self.linear2D3D.bias
        B, C, H, W = feat2D3D.shape
        cam2D3D = torch.matmul(linear_weight, feat2D3D.view(B, C, H * W)).view(B, 2, H, W) + linear_bias.view(1, -1, 1, 1)
        return out2D, out3D, out2D3D, cam2D, cam3D, cam2D3D



class _2branch_DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config2D=(8, 12,8,6), block_config3D=(4, 6, 4, 3), sharedblock_config = 4,
                 num_init_features=64, num_fuse_features = 256, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(_2branch_DenseNet, self).__init__()
        self._2Dnet = dense_cbam_outfeat(block_config=block_config2D)
        self._3Dnet = _3D_DenseNet16_outfeat(block_config=block_config3D)
        self.linear = nn.Linear(688, num_classes)

    def forward(self, x2d, x3d):
        out2d, cam2d, feat2D = self._2Dnet(x2d)
        out3d, cam3d, feat3D = self._3Dnet(x3d)

        feat_fusion = torch.cat([feat2D.detach(), feat3D.detach()], 1)
        out_fuse = self.linear(feat_fusion)
        return out2d, out3d, out_fuse, cam2d, cam3d


class _3D2D_DenseNet(nn.Module):
    r"""
    3D2D take as input 5D tensor. After two blocks, remove one dim and become 2D convolution.
    """

    def __init__(self, growth_rate=32, block_config=(2, 2, 4, 4, 4, 4),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D2D_DenseNet, self).__init__()
        # First convolution
        # input: N x 1 x D x H x W
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(7,7,7), stride=(1, 2, 2),
                                padding=(3,3,3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        # input: N x 64 x D/2 x H/4 x W/4
        # Each dense block
        num_features = num_init_features
        self._3D_block0 = _3D_DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate

        self._3D_D_trans0 = _3D_D_Transition(num_input_features=num_features, num_output_features=num_features)
        # input: N x 64 x D/4 x H/4 x W/4
        self._3D_block1 = _3D_DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate

        self._3D_trans1 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        # input: N x 64 x D/8 x H/8 x W/8
        self._3D_block2 = _3D_DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate

        self._3D_D_trans2 = _3D_D_Transition(num_input_features=num_features, num_output_features=num_features)
        # input: N x 64 x D/16 x H/8 x W/8
        self._3D_block3 = _3D_DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate

        self._3D_trans3 = _3D_Transition(num_input_features=num_features, num_output_features=num_features//2)
        num_features = num_features // 2
        # input: N x 64 x H/16 x W/16
        self._2D_block4 = _DenseBlock(num_layers=block_config[4], num_input_features=num_features, bn_size=bn_size,
                                      growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[4] * growth_rate

        self._2D_trans4 = _Transition(num_input_features=num_features, num_output_features=num_features//2)
        num_features = num_features // 2
        # # input: N x 64 x H/32 x W/32
        self._2D_block5 = _DenseBlock(num_layers=block_config[5], num_input_features=num_features, bn_size=bn_size,
                                      growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[5] * growth_rate

        self.norm5 = nn.BatchNorm2d(num_features)
        # Linear layer
        self.linear = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 1 x D x H x W
        x = self.conv_block(x)
        # N x 64 x D/2 x H/4 x W/4
        x = self._3D_block0(x)
        # N x 128 x D/4 x H/4 x W/4
        x = self._3D_D_trans0(x)
        # N x 64 x D/4 x H/4 x W/4
        x = self._3D_block1(x)
        # N x 128 x D/4 x H/4 x W/4
        x = self._3D_trans1(x)
        # N x 64 x D/8 x H/8 x W/8
        x = self._3D_block2(x)
        # N x (64 + num_layer_block2) x D/8 x H/8 x W/8
        x = self._3D_D_trans2(x)
        x = self._3D_block3(x)
        x = self._3D_trans3(x)

        x = self._2D_block4(x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4]))
        x = self._2D_trans4(x)
        x = self._2D_block5(x)
        out_norm5 = self.norm5(x)
        out = F.relu(out_norm5, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        linear_weight = self.linear.weight
        B, C, H, W = out_norm5.shape
        cam = torch.matmul(linear_weight, out_norm5.view(B, C, H * W)).view(B, 2, H, W)
        return out, cam



class _3D_DenseSENet(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(4, 8, 8, 4),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(_3D_DenseSENet, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(7, 7, 7), stride=(1, 2, 2),
                                padding=(3, 3, 3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _3D_DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _3D_DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _3D_DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _3D_Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _3D_DenseSEBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.norm5 = nn.BatchNorm3d(num_features)

        # Linear layer
        self.linear = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif (isinstance(m, nn.Linear)) and (m.bias is not None):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.block3(self.trans3(out_block2))
        out_norm5 = self.norm5(out_block3)
        out = F.relu(out_norm5, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        linear_weight = self.linear.weight
        B, C, D, H, W = out_norm5.shape
        cam = torch.matmul(linear_weight, out_norm5.view(B, C, H * W)).view(B, 2, H, W) + self.linear.bias.view(1, -1, 1, 1)
        return out, cam


class _3D_CNN(nn.Module):
    def __init__(self, num_init_features=64):
        super(_3D_CNN, self).__init__()
        self.pool3d = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.pool2d = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2),padding=(0,1,1))
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(3,7,7), stride=(1,2,2),
                                padding=(1,3,3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(num_init_features, num_init_features,  kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.conv2 = _3Dconv_layer(num_init_features, 128)
        self.conv3 = _3Dconv_layer(128, 256)
        self.conv4 = _3Dconv_layer(256, 512)
        self.conv5 = _3Dconv_layer(512, 512)
        self.conv6 = _3Dconv_layer(512, 768)
        self.conv7 = _3Dconv_layer(768, 768)
        self.linear = nn.Linear(768, 2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool3d(self.conv1(x))
        x = self.conv2(x)
        x = self.pool3d(self.conv3(x))
        x = self.conv4(x)
        x = self.pool3d(self.conv5(x))
        x = self.conv6(x)
        x = self.conv7(x)
        out_gap = F.adaptive_avg_pool3d(x, (1, 1, 1))
        out = self.linear(torch.flatten(out_gap, 1))
        linear_weight = self.linear.weight
        B, C, D, H, W = out_conv4.shape
        cam = torch.matmul(linear_weight, out_conv4.view(B, C, H * W)).view(B, 2, H, W) + self.linear.bias.view(1, -1, 1, 1)
        return out, cam


class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _,  _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class CAM(nn.Module):

    """Channel Attention Module

    """

    def __init__(self, in_channels, reduction_ratio=16):

        """
        :param in_channels: int

            Number of input channels.

        :param reduction_ratio: int

            Channels reduction ratio for MLP.
        """

        super().__init__()

        reduced_channels_num = (in_channels // reduction_ratio) if (in_channels > reduction_ratio) else 1

        pointwise_in = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=reduced_channels_num)

        pointwise_out = nn.Conv2d(kernel_size=1, in_channels=reduced_channels_num, out_channels=in_channels)

        # In the original paper there is a standard MLP with one hidden layer.

        # TODO: try linear layers instead of pointwise convolutions.

        self.MLP = nn.Sequential(pointwise_in, nn.ReLU(), pointwise_out,)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        h, w = input_tensor.size(2), input_tensor.size(3)



        # Get (channels, 1, 1) tensor after MaxPool

        max_feat = F.max_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))

        # Get (channels, 1, 1) tensor after AvgPool

        avg_feat = F.avg_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))

        # Throw maxpooled and avgpooled features into shared MLP

        max_feat_mlp = self.MLP(max_feat)

        avg_feat_mlp = self.MLP(avg_feat)

        # Get channel attention map of elementwise features sum.

        channel_attention_map = self.sigmoid(max_feat_mlp + avg_feat_mlp)

        return channel_attention_map


class SAM(nn.Module):

    """Spatial Attention Module"""



    def __init__(self, ks=7):

        """



        :param ks: int

            kernel size for spatial conv layer.

        """



        super().__init__()

        self.ks = ks

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(kernel_size=self.ks, in_channels=2, out_channels=1)



    def _get_padding(self, dim_in, kernel_size, stride):

        """Calculates \'SAME\' padding for conv layer for specific dimension.



        :param dim_in: int

            Size of dimension (height or width).

        :param kernel_size: int

            kernel size used in conv layer.

        :param stride: int

            stride used in conv layer.

        :return: int

            padding

        """



        padding = (stride * (dim_in - 1) - dim_in + kernel_size) // 2

        return padding



    def forward(self, input_tensor):

        c, h, w = input_tensor.size(1), input_tensor.size(2), input_tensor.size(3)


        # Permute input tensor for being able to apply MaxPool and AvgPool along the channel axis

        permuted = input_tensor.view(-1, c, h * w).permute(0,2,1)

        # Get (1, h, w) tensor after MaxPool

        max_feat = F.max_pool1d(permuted, kernel_size=c, stride=c)

        max_feat = max_feat.permute(0,2,1).view(-1, 1, h, w)


        # Get (1, h, w) tensor after AvgPool

        avg_feat = F.avg_pool1d(permuted, kernel_size=c, stride=c)

        avg_feat = avg_feat.permute(0,2,1).view(-1, 1, h, w)



        # Concatenate feature maps along the channel axis, so shape would be (2, h, w)

        concatenated = torch.cat([max_feat, avg_feat], dim=1)

        # Get pad values for SAME padding for conv2d

        h_pad = self._get_padding(concatenated.shape[2], self.ks, 1)

        w_pad = self._get_padding(concatenated.shape[3], self.ks, 1)

        # Get spatial attention map over concatenated features.

        self.conv.padding = (h_pad, w_pad)

        spatial_attention_map = self.sigmoid(

            self.conv(concatenated)

        )

        return spatial_attention_map


class CBAM(nn.Module):

    """Convolutional Block Attention Module

    https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf

    """

    def __init__(self, in_channels):

        """
        :param in_channels: int

            Number of input channels.

        """

        super().__init__()

        self.CAM = CAM(in_channels)

        self.SAM = SAM()


    def forward(self, input_tensor):

        # Apply channel attention module

        channel_att_map = self.CAM(input_tensor)

        # Perform elementwise multiplication with channel attention map.

        gated_tensor = torch.mul(input_tensor, channel_att_map)  # (bs, c, h, w) x (bs, c, 1, 1)

        # Apply spatial attention module

        spatial_att_map = self.SAM(gated_tensor)

        # Perform elementwise multiplication with spatial attention map.

        refined_tensor = torch.mul(gated_tensor, spatial_att_map)  # (bs, c, h, w) x (bs, 1, h, w)

        return refined_tensor


class dense_cbam(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels=1, growth_rate=32, block_config=(8, 12, 12, 8),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(dense_cbam, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 1
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM0 = CBAM(num_features)
        # block 2
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM1 = CBAM(num_features)
        # block 3
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM2 = CBAM(num_features)
        # block 4
        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)
        self.num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.sig = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.CBAM0(self.trans0(self.block0(out_conv)))
        out_block1 = self.CBAM1(self.trans1(self.block1(out_block0)))
        out_block2 = self.CBAM2(self.trans2(self.block2(out_block1)))
        out_block3 = self.block3(out_block2)
        out_norm = self.norm(out_block3)
        out_gap = F.adaptive_avg_pool2d(out_norm, (1, 1)).view(x.shape[0], -1)
        out = self.classifier(out_gap)

        linear_weight = self.classifier.weight
        linear_bias = self.classifier.bias
        B, C, H, W = out_norm.shape
        cam = torch.matmul(linear_weight, out_norm.view(B, C, H * W)).view(B, 2, H, W) + linear_bias.view(1, -1, 1, 1)
        return out, cam




class dense_cbam_outfeat(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels=1, growth_rate=32, block_config=(8, 12, 12, 8),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(dense_cbam_outfeat, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 1
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM0 = CBAM(num_features)
        # block 2
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,drop_rate=drop_rate)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM1 = CBAM(num_features)
        # block 3
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM2 = CBAM(num_features)
        # block 4
        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)
        self.num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        #self.sig = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.CBAM0(self.trans0(self.block0(out_conv)))
        out_block1 = self.CBAM1(self.trans1(self.block1(out_block0)))
        out_block2 = self.CBAM2(self.trans2(self.block2(out_block1)))
        out_block3 = self.block3(out_block2)
        out_norm = self.norm(out_block3)
        out_gap = F.adaptive_avg_pool2d(out_norm, (1, 1)).view(x.shape[0], -1)
        out = self.classifier(out_gap)

        linear_weight = self.classifier.weight
        linear_bias = self.classifier.bias
        B, C, H, W = out_norm.shape
        cam = torch.matmul(linear_weight, out_norm.view(B, C, H * W)).view(B, 2, H, W) + linear_bias.view(1, -1, 1, 1)
        return out, cam, out_gap








