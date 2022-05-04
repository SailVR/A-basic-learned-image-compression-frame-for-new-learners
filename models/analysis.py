#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN

class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self):
        super(Analysis_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + 48) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(48)
        self.conv2 = nn.Conv2d(48, 96, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2 * (48 + 96) / (48 + 48))))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(96)
        self.conv3 = nn.Conv2d(96, 192, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2 * (96 + 192) / (96 + 96))))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(192)
        self.conv4 = nn.Conv2d(192, 320, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (192 + 320) / (192 + 192))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.gdn4 = GDN(320)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.gdn4(self.conv4(x))
        return x


def build_model():
        input_image = torch.zeros([4, 3, 256, 256])

        analysis_net = Analysis_net()
        feature = analysis_net(input_image)

        print(feature.size())


if __name__ == '__main__':
    build_model()
