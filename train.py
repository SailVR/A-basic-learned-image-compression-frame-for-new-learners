import os
import argparse
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datasets import Datasets, TestKodakDataset
from Meter import AverageMeter
from pytorch_msssim import ms_ssim

torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4
print_freq = 100
cal_step = 50
batch_size = 16
tot_epoch = 1000000
tot_step = 1800000
decay_interval = 1500000
lr_decay = 0.1
image_size = 256
logger = logging.getLogger("ImageCompression")
global_step = 0
save_model_freq = 100000
test_step = 10000
parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('-n', '--name', default='compress_base', help='experiment name')
parser.add_argument('-p', '--pretrain', default='', help='load pretrain model')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--train', dest='train', default='/home/dataset/coco2017', help='the path of training dataset')
parser.add_argument('--val', dest='val', default='/home/dataset/Kodak', help='the path of validation dataset')
parser.add_argument("--lmbda",dest="lmbda",type=float,default=0.013,help="Bit-rate distortion parameter (default: %(default)s)",)
parser.add_argument("--metrics",type=str,default="mse",help="Optimized for (default: %(default)s)",)

def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    if global_step < decay_interval:
        lr = base_lr
    else:
        lr = base_lr * lr_decay
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch, global_step):
    logger.info("Epoch {} begin".format(epoch))
    net.train()
    global optimizer
    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, mse_losses = [AverageMeter(print_freq) for _ in range(7)]

    for batch_idx, input in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        clipped_recon_image, mse_loss,ms_ssim_loss, bpp_feature, bpp_z, bpp = net(input)

        distribution_loss = bpp
        if args.metrics=='mse':
            rd_loss = args.lmbda * 255 * 255 * mse_loss + distribution_loss
        elif args.metrics=='ms-ssim':
            rd_loss = args.lmbda * ms_ssim_loss + distribution_loss

        optimizer.zero_grad()
        rd_loss.backward()

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(optimizer, 5)
        optimizer.step()

        if (global_step % cal_step) == 0:
            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item())
            else:
                psnrs.update(100)
 
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(bpp.item())
            bpp_features.update(bpp_feature.item())
            bpp_zs.update(bpp_z.item())
            mse_losses.update(mse_loss.item())

        if (global_step % print_freq) == 0:
            process = global_step / tot_step * 100.0
            log = (' | '.join([
                f'Step [{global_step}/{tot_step}={process:.2f}%]',
                f'Epoch {epoch}',
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'Lr {cur_lr}',
                f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                f'Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})',
                f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
            ]))
            logger.info(log)

        if (global_step % save_model_freq) == 0:
            save_model(model, global_step, save_path)
        if (global_step % test_step) == 0:
            testKodak(global_step)
            net.train()
        if global_step >= tot_step:
            save_model(model, global_step, save_path)
            logger.info("训练结束")
            sys.exit(0)

    return global_step

def testKodak(step):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            clipped_recon_image, mse_loss, ms_ssim_loss, bpp_feature, bpp_z, bpp = net(input)
            mse_loss, bpp_feature, bpp_z, bpp = torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            
            mse_loss = torch.mean((clipped_recon_image.cpu() - input.cpu()).pow(2)) 

            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input, data_range=1.0,size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            logger.info("Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(bpp, psnr, msssim, msssimDB))
            cnt += 1

        logger.info("Test on Kodak dataset: model-{}".format(step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    save_path = os.path.join('checkpoints', args.name+'_'+args.metrics+'_'+str(args.lmbda))
    if args.name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("image compression training")

    model = ImageCompressor()
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)

    net = torch.nn.DataParallel(model.cuda(), device_ids=[0])
    logger.info(net)

    parameters = net.parameters()
    global test_loader
    test_dataset = TestKodakDataset(data_dir=args.val)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
    optimizer = optim.Adam(parameters, lr=base_lr)
    global train_loader
    train_data_dir = args.train
    train_dataset = Datasets(train_data_dir, image_size)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=2)
    steps_epoch = global_step // (len(train_dataset) // (batch_size))
    save_model(model, global_step, save_path)
    for epoch in range(steps_epoch, tot_epoch):
        adjust_learning_rate(optimizer, global_step)
        global_step = train(epoch, global_step)