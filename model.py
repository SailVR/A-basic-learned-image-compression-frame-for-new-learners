from re import X
import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *
from pad_crop import *
from pytorch_msssim import ms_ssim

def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

class ImageCompressor(nn.Module):
    def __init__(self):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net()
        self.Decoder = Synthesis_net()
        self.priorEncoder = Analysis_prior_net()
        self.priorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(128)

        self.entropy = Entropy(320)

    def forward(self, input_image):
        image_shape = input_image.shape
        input_image = pad_img(input_image)

        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]

        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + torch.rand_like(z) - 0.5
        else:
            compressed_z = torch.round(z)
        z_round = z + (torch.round(z)-z).detach()
        phi = self.priorDecoder(z_round)

        if self.training:
            compressed_feature = feature + torch.rand_like(feature) - 0.5
        else:
            compressed_feature = torch.round(feature)
        feature_round = feature + (torch.round(feature)-feature).detach()
        
        recon_mu, recon_sigma = self.entropy(phi, compressed_feature)

        recon_image = self.Decoder(feature_round)

        clipped_recon_image = recon_image.clamp(0., 1.)
        clipped_recon_image = unpad_img(clipped_recon_image,image_shape)

        mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))
        ms_ssim_loss = 1  - ms_ssim(clipped_recon_image, input_image, data_range=1.0,size_average=True)
        
        def feature_probs_based_sigma(feature, mu,sigma):
            sigma = torch.clip(torch.exp(sigma),1e-10,1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature, recon_mu, recon_sigma)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, mse_loss,ms_ssim_loss, bpp_feature, bpp_z, bpp

if __name__ == '__main__':

    input_image = torch.rand([4, 3, 256, 256])

    net = ImageCompressor()
    print(net)

    clipped_recon_image, mse_loss,ms_ssim_loss, bpp_feature, bpp_z, bpp = net(input_image)

    print(clipped_recon_image.size())
    print(mse_loss)

    total = sum([param.nelement() for param in net.parameters()])
    
    print("Number of parameter: %.2fM" % (total/1e6))