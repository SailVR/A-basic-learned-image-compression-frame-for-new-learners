import os
import argparse

from PIL import Image
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datasets import Datasets, TestKodakDataset
from torchvision import transforms
import pandas as pd

torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()

logger = logging.getLogger("ImageCompression")
parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('-p', '--pretrain', default = './checkpoints/compress_base_mse_0.013/iter_0.pth.tar',help='load pretrain model')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--val', dest='val_path', default='/home/dataset/Kodak', help='the path of validation dataset')

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    tb_logger = None
    logger.setLevel(logging.INFO)
    logger.info("image compression test")
    model = ImageCompressor()
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
    net = model.cuda()

    global test_loader

    test_dataset = TestKodakDataset(data_dir=args.val_path)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True)

    df = pd.DataFrame()

    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        files = os.listdir(args.val_path)
        # files.sort(key= lambda x:int(x[:-4]))
        # files.sort(key= lambda x:int(x.split('_')[1]))
        # print(files)
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(args.val_path,file)
                print(image_path)

                try:
                    im = Image.open(image_path).convert('RGB')
                except:
                    print(image_path,"error")
                    continue
                input = transforms.ToTensor()(im)
                input = input.unsqueeze(0)
                
                clipped_recon_image, mse_loss, ms_ssim_loss, bpp_feature, bpp_z, bpp = net(input.cuda())
                # imgRlt = transforms.ToPILImage()(clipped_recon_image.squeeze(0))
                # imgRlt.save("/home/attrsense_haihang/Variation/1.png")

                psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
                sumBpp += bpp
                sumPsnr += psnr
                msssimDB = -10 * (torch.log(ms_ssim_loss) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += 1 - ms_ssim_loss
                cnt += 1

                logger.info("Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, bpp, psnr, 1-ms_ssim_loss, msssimDB))

                df['name'] = [os.path.join(args.val_path,file)]
                df['bpp'] = [bpp.cpu().numpy()]
                df['psnr'] = [psnr.cpu().numpy()]
                # if cnt==1:
                #     df.to_csv("./df.csv",index=False)
                # else:
                #     df.to_csv("./df.csv",index=False,mode='a',header=None)
            
        logger.info("Test on Kodak dataset: model-{}".format(global_step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info("Dataset Average result---Dataset Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))