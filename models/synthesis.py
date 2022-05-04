import math
import torch.nn as nn
import torch
from .GDN import GDN
from .analysis import Analysis_net

class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self):
        super(Synthesis_net, self).__init__()

        self.igdn0 = GDN(320,inverse=True)
        self.deconv1 = nn.ConvTranspose2d(320, 192, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * (320 + 192) / (192 + 192))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)

        self.igdn1 = GDN(192, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, (math.sqrt(2 * (96 + 192) / (96 + 96))))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)

        self.igdn2 = GDN(96, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * (96 + 48) / (48 + 48))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        
        self.igdn3 = GDN(48, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(48, 3, 4, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (48 + 3) / (3 + 3))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn0(x)
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x

def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net()
    synthesis_net = Synthesis_net()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())

if __name__ == '__main__':
    build_model()
