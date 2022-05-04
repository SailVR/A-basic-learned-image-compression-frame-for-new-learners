import math
import torch.nn as nn
import torch
from .synthesis_prior import Synthesis_prior_net

class MaskedConv2d(nn.Conv2d):
    '''
    clone this function from https://github.com/thekoshkina/learned_image_compression/blob/master/masked_conv.py
    Implementation of the Masked convolution from the paper
    Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders."
    Advances in neural information processing systems. 2016.
    https://arxiv.org/pdf/1606.05328.pdf
    '''

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        # fill_将tensor中所有值都填充为指定value
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        # 这相当于对卷积核进行操作嘛
        self.weight.data *= self.mask
        # 调用maskedconv父类，进行卷积呗
        return super(MaskedConv2d, self).forward(x)


class Entropy(nn.Module):
    def __init__(self, num_filters=128):
        super(Entropy, self).__init__()
        self.maskedconv = MaskedConv2d('A', num_filters, num_filters*2, 5, stride=1, padding=2)
        torch.nn.init.xavier_uniform_(self.maskedconv.weight.data, gain=1)
        torch.nn.init.constant_(self.maskedconv.bias.data, 0.0)
        self.conv1 = nn.Conv2d(num_filters*3, 960, 1, stride=1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(960, 960, 1, stride=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(960, num_filters*2, 1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sigma, y):
        y = self.maskedconv(y)
        x = torch.cat([y, sigma], dim=1)
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.conv3(x)

        means, variances = torch.split(x, split_size_or_sections=int(x.shape[1]/2), dim=1)

        return means,variances

if __name__ == "__main__":
    y = torch.zeros([1,320,16,16])
    z = torch.zeros([1,128,8,8])
    entropy = Entropy(320)
    hyper_synthesis = Synthesis_prior_net()
    phi = hyper_synthesis(z)
    print("phi.shape",phi.shape)
    print("y.shape",y.shape)

    means, variances = entropy(phi, y)
    print("means: ", means.shape)
    print("variances: ", variances.shape)