#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
# from .basics import *
from .analysis import Analysis_net
import math
import torch.nn as nn
import torch

class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self):
        super(Analysis_prior_net, self).__init__()
        self.conv1 = nn.Conv2d(320, 128, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (320 + 128) / (320 + 320))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

        self.leakrelu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)

    def forward(self, x):
        x = self.conv2(self.leakrelu1(self.conv1(x)))
        return x


def build_model():
    input_image = torch.zeros([5, 3, 256, 256])
    analysis_net = Analysis_net()
    analysis_prior_net = Analysis_prior_net()

    feature = analysis_net(input_image)
    z = analysis_prior_net(feature)
    
    print(input_image.size())
    print(feature.size())
    print(z.size())


if __name__ == '__main__':
    build_model()
