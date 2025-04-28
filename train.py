import torch
import torch.nn as nn
import torch.optim as optim
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset import get_loader
from loss import *
from config import Config
from evaluation.dataloader import EvalDataset
from evaluation.evaluator import Eval_thread


from models.main import *

import torch.nn.functional as F
import pytorch_toolbelt.losses as PTL

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Parameter from command line
parser = argparse.ArgumentParser(description='')

parser.add_argument('--loss',
                    default='Scale_IoU',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='CoCo',
                    type=str,
                    help="Options: 'CoCo'")
parser.add_argument('--testsets',
                    default='CoCA',
                    type=str,
                    help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default='./checkpoint2', help='Temporary folder')
parser.add_argument('--best_epochs', default='./checkpoint2/best_epochs/', help='保存最好的那些模型')
parser.add_argument('--save_root', default='./pred_temp', type=str, help='Output folder')

args = parser.parse_args()
config = Config()

# Prepare dataset
if args.trainset == 'CoCo':
    train_img_path = '/home/huangjiu/projects/DCFM/data4/images/COCO9213-os/'
    train_gt_path = '/home/huangjiu/projects/DCFM/data4/gts/COCO9213-os/'
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              args.bs,
                              max_num=8, #20,
                              istrain=True,
                              shuffle=False,
                              num_workers=8, #4  作者初始代码是8 不知为何
                              pin=True)

else:
    print('Unkonwn train dataset')
    print(args.dataset)

for testset in ['CoCA']:
    if testset == 'CoCA':
        test_img_path = '/home/huangjiu/projects/DCFM/data/images/CoCA/'
        test_gt_path = '/home/huangjiu/projects/DCFM/data/gts/CoCA/'

        saved_root = os.path.join(args.save_root, 'CoCA')
    elif testset == 'CoSOD3k':
        test_img_path = '/home/huangjiu/projects/DCFM/data/images/CoSOD3k/'
        test_gt_path = '/home/huangjiu/projects/DCFM/data/gts/CoSOD3k/'
        saved_root = os.path.join(args.save_root, 'CoSOD3k')
    elif testset == 'CoSal2015':
        test_img_path = '/home/huangjiu/projects/DCFM/data/images/CoSal2015/'
        test_gt_path = '/home/huangjiu/projects/DCFM/data/gts/CoSal2015/'
        saved_root = os.path.join(args.save_root, 'CoSal2015')
    elif testset == 'CoCo':
        test_img_path = '/home/huangjiu/projects/DCFM/data/images/CoCo/'
        test_gt_path = '/home/huangjiu/projects/DCFM/data/gts/CoCo/'
        saved_root = os.path.join(args.save_root, 'CoCo')
    else:
        print('Unkonwn test dataset')
        print(args.dataset)

    test_loader = get_loader(
        test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

# make dir for tmp
os.makedirs(args.tmp, exist_ok=True)

# make dir for best_epochs
os.makedirs(args.best_epochs, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.tmp, "log.txt"))
set_seed(123)

# Init model
device = torch.device("cuda")

model = DCFM()
model = model.to(device)
model.apply(weights_init)

model.dcfmnet.backbone._initialize_weights(torch.load('/home/huangjiu/projects/DCFM/models/vgg16-397923af.pth'))

backbone_params = list(map(id, model.dcfmnet.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.dcfmnet.parameters())

all_params = [{'params': base_params}, {'params': model.dcfmnet.backbone.parameters(), 'lr': args.lr*0.1}]

# Setting optimizer
optimizer = optim.Adam(params=all_params,lr=args.lr, weight_decay=1e-4, betas=[0.9, 0.99])

for key, value in model.named_parameters():
    if 'dcfmnet.backbone' in key and 'dcfmnet.backbone.conv5.conv5_3' not in key:
        value.requires_grad = False

for key, value in model.named_parameters():
    print(key,  value.requires_grad)

# log model and optimizer pars
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
# logger.info(scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
exec('from loss import ' + args.loss)
IOUloss = eval(args.loss+'()')


def sclloss(x, xt, xb):
    cosc = (1+compute_cos_dis(x, xt))*0.5
    cosb = (1+compute_cos_dis(x, xb))*0.5
    loss = -torch.log(cosc+1e-5)-torch.log(1-cosb+1e-5)
    return loss.sum()


def train(epoch):
    # Switch to train mode
    model.train()
    model.set_mode('train')
    loss_sum = 0.0
    loss_sumkl = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        pred, proto, protogt, protobg = model(inputs, gts)
        # loss_iou返回几张预测图loss的相加之和
        loss_iou = IOUloss(pred, gts)
        loss_scl = sclloss(proto, protogt, protobg)
        loss = loss_iou+0.1*loss_scl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss_iou.detach().item()

        if batch_idx % 20 == 0:
            logger.info('正在训练第{0}个epoch, 总epoch:{1}, 迭代次数: Iter[{2}/{3}]  '
                        '训练损失: loss_iou: {4:.3f}, loss_scl: {5:.3f} '.format(
                            (epoch+1),
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss_iou,
                            loss_scl,
                        ))
    loss_mean = loss_sum / len(train_loader)
    return loss_sum


def validate(model, test_loaders, testsets):
    model.eval()

    testsets = testsets.split('+')
    measures = []
    for testset in testsets[:1]:
        print('Validating {}...'.format(testset))
        #test_loader = test_loaders[testset]

        saved_root = os.path.join(args.save_root, testset)

        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs, gts)[-1].sigmoid()

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))

        eval_loader = EvalDataset(
            saved_root,  # preds
            os.path.join('/home/huangjiu/projects/DCFM/data/gts', testset)  # GT
        )
        evaler = Eval_thread(eval_loader, cuda=True)
        # Use S_measure for validation
        s_measure = evaler.Eval_Smeasure()
        if s_measure > config.val_measures['Smeasure']['CoCA'] and 0:
            # TODO: evluate others measures if s_measure is very high.
            e_max = evaler.Eval_Emeasure().max().item()
            f_max = evaler.Eval_fmeasure().max().item()
            print('Emax: {:4.f}, Fmax: {:4.f}'.format(e_max, f_max))
        measures.append(s_measure)

    model.train()
    return measures


def main():
    val_measures = []
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.dcfmnet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    print(args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(epoch)
        if config.validation:
            measures = validate(model, test_loader, args.testsets)
            # val_measures:保存每个measures（Smeasures）
            # 例如 val_measures = [[0.4386], [0.4433], [0.4633]]
            # np.array(val_measures)[:, 0]打印结果为：[0.4386 0.4433 0.4633]
            # np.argmax得到最大值的下标索引
            # np.max得到最大的值
            val_measures.append(measures)
            print(
                'Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                    (epoch+1), measures[0], (np.argmax(np.array(val_measures)[:, 0].squeeze())+1),
                    np.max(np.array(val_measures)[:, 0]))
            )
            # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.dcfmnet.state_dict(),
                #'scheduler': scheduler.state_dict(),
            },
            path=args.tmp)
        # if config.validation:
        #     # 如果本次评估得到的Smeasure为最大值
        #     if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
        #         best_weights_before = [os.path.join(args.tmp, weight_file) for weight_file in
        #                                os.listdir(args.tmp) if 'best_' in weight_file]
        #         # best_weights_before只保存之前的最好模型 例如['/home/huangjiu/projects/DCFM/checkpoint/best_ep12_Smeasure0.4437.pth']
        #         for best_weight_before in best_weights_before:
        #             os.remove(best_weight_before)
        #         torch.save(model.dcfmnet.state_dict(), os.path.join(args.tmp, 'best_ep{}_Smeasure{:.4f}.pth'.format((epoch+1), measures[0])))

########################################################################################################################################################

        if config.validation:
            if epoch <= 9:
                # 例：ep23_Smeasure0.6014.pth
                torch.save(model.dcfmnet.state_dict(), os.path.join(args.best_epochs, 'ep{}_Smeasure{:.4f}.pth'.format((epoch+1), measures[0])))
            if epoch > 9:
                # 例如epoch_names = ['ep12_Smeasure0.4437.pth','ep15_Smeasure0.4625.pth','ep19_Smeasure0.4786.pth',]
                epoch_names = os.listdir(args.best_epochs)
                # best_Sm用来存放最好的十个Sm值
                best_Sm = []
                for epoch_name in epoch_names:
                    ep1 = epoch_name.split('.pth')[0]
                    now_Sm = ep1.split('Smeasure')[-1]
                    # 将字符串转为浮点数
                    now_Sm = float(now_Sm)
                    best_Sm.append(now_Sm)

                # idex_min为Sm值最小的下标
                idex_min = np.argmin(best_Sm)
                # 如果本次epoch的Sm大于best_Sm十个中的最小值
                if (measures[0] >= np.min(best_Sm)):
                    # 删除最小的那个pth文件
                    os.remove(os.path.join(args.best_epochs,epoch_names[idex_min]))
                    # 保存本次epoch的pth文件
                    torch.save(model.dcfmnet.state_dict(), os.path.join(args.best_epochs, 'ep{}_Smeasure{:.4f}.pth'.format((epoch+1), measures[0])))
                

########################################################################################################################################################


        # # 隔25个epoch保存一次
        # if (epoch + 1) % 30 == 0 or epoch == 0:
        #     torch.save(model.dcfmnet.state_dict(), args.tmp + '/model-' + str(epoch + 1) + '.pt')
       
        # # 从195个epoch开始保存
        # if (epoch+1) > 195:
        #     torch.save(model.dcfmnet.state_dict(), args.tmp+'/model-' + str(epoch + 1) + '.pt')
    #dcfmnet_dict = model.dcfmnet.state_dict()
    #torch.save(dcfmnet_dict, os.path.join(args.tmp, 'final.pth'))
        logger.info("Epoch{}'s Smeasure:  {:.4f}".format((epoch+1), measures[0]))
        logger.info("------------------------------------------------------------------------------------")
if __name__ == '__main__':
    main()
