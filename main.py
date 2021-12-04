# used as reference https://github.com/princeton-vl/SimpleView/blob/master/main.py
import argparse
import random
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset import ModelNetDataset
from pointnet.model import PointNet, feature_transform_regularizer

from simpleview.model import MVModel

from utils.loss import smooth_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    print(' Using CPU')

blue = lambda x: '\033[94m' + x + '\033[0m'


def get_dataset(cfg):
    if cfg.dataset_type == 'modelnet40':

        convert_data = True

        for fname in os.listdir(os.path.join(cfg.dataset,"airplane", "test")):
            if fname.endswith('.ply'):
                convert_data=False
                break
        
        dataset = ModelNetDataset(
            root=cfg.dataset,
            cfg_args=cfg,
            npoints=cfg.num_points,
            split='trainval',
            data_augmentation=True,
            pointwolf=cfg.PointWOLF,
            convert_off_to_ply=convert_data
            )

        test_dataset = ModelNetDataset(
            root=cfg.dataset,
            cfg_args=cfg,
            split='test',
            npoints=cfg.num_points,
            data_augmentation=False,
            convert_off_to_ply=convert_data)
    else:
        raise KeyError("Invalid Dataset Type Specified.")

    return dataset, test_dataset

def get_model(model_name, dataset, feature_transform, task='cls'):
    if model_name == 'simpleview':

        model = MVModel(
            task=task,
            dataset=dataset,
            backbone='resnet18',
            feat_size = 16)

    elif model_name == 'pointnet':

        model = PointNet(
            dataset=dataset,
            task=task)
    else:
        raise KeyError("Invalid Model Type Specified")

    return model

def run_forward(points, target, model, cfg):
    target = target[:, 0]
    
    points, target = points.cuda(), target.cuda()
    out = model(points)

    if cfg.model_name == 'pointnet':
        loss = F.cross_entropy(out['logit'], target)
    elif cfg.model_name == 'simpleview':
        loss = smooth_loss(out['logit'], target)
    else:
        raise TypeError("Unsupported Model Type")

    if cfg.feature_transform:
        loss += feature_transform_regularizer(out['trans_feat']) * 0.001

    return out, loss

def entry_train(cfg):
    
    dataset, test_dataset = get_dataset(cfg)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batchSize,
        shuffle=True,
        num_workers=int(cfg.workers))

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.batchSize,
            shuffle=True,
            num_workers=int(cfg.workers))

    print(
        "Training points:", len(dataset), 
        " Test points:", len(test_dataset), 
        " Classes:", len(dataset.classes)
        )


    model = get_model(cfg.model_name, dataset, feature_transform=cfg.feature_transform, task='cls')
    model.to(DEVICE)

    print(model)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    if cfg.model != '':
        checkpoint = torch.load(cfg.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        epochs = checkpoint['epochs']
        start_epoch = checkpoint['epochs'][-1] + 1
    else:
        train_losses = []
        test_losses = []
        epochs = []
        start_epoch = 1

    for epoch in range(start_epoch, start_epoch + cfg.nepoch):
        print(f"\nEpoch {epoch}:")
        epochs.append(epoch)
        train_loss, test_loss, train_correct, test_correct, train_num, test_num = 0, 0, 0.0, 0.0, 0, 0
        model.train()

        for data in tqdm(dataloader):
            points, target = data
            out, loss = run_forward(points, target, model, cfg)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred_choice = out['logit'].data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            train_correct += correct.item()
            train_num += points.size()[0]

        scheduler.step()
        ave_train_loss = train_loss / len(dataloader) # all batch sizes may not be same
        train_losses.append(ave_train_loss)
        print('train loss: %f accuracy: %f' % (ave_train_loss, train_correct/train_num))

        with torch.no_grad():
            model.eval()

            for data in tqdm(testdataloader):

                points, target = data
                out, loss = run_forward(points, target, model, cfg)

                test_loss += loss.item()

                pred_choice = out['logit'].data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                test_correct += correct.item()
                test_num += points.size()[0]

        ave_test_loss = test_loss / len(testdataloader) # all batch sizes may not be same
        test_losses.append(ave_test_loss)
        print('%s loss: %f accuracy: %f' % (blue('test'), ave_test_loss, test_correct/test_num))
        
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'epochs': epochs},
                    '%s/pointnet_%d.pth' % (cfg.outf, epoch))

    # plotting train and val losses
    ax1 = plt.subplot()
    ax1.plot(epochs, train_losses, label='train')
    ax1.plot(epochs, test_losses, label='test')

    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    # ax1.set_yscale('log')
    ax1.legend()

    ax1.set_title(f"epochs={cfg.nepoch}, batchsz={cfg.batchSize}")
    plt.tight_layout()

    plt.savefig(f"cls/training.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--entry', type=str, default="train")
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model_name', type=str, default="pointnet", help='model to run, pointnet|simpleview')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--seed', default=42)

    # PointWOLF Arguments
    parser.add_argument('--PointWOLF', action='store_true', help='Use PointWOLF')
    
    parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point' ) 
    parser.add_argument('--w_sample_type', type=str, default='fps', help='Sampling method for anchor point, option : (fps, random)') 
    parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')  

    parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
    parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
    parser.add_argument('--w_T_range', type=float, default=0.25, help='Maximum translation range of local transformation')

    cmd_args = parser.parse_args()

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if cmd_args.entry == "train":

        entry_train(cmd_args)

    elif cmd_args.entry in ["test", "valid"]:

        raise NotImplementedError("Its on my to do list.")

    else:
        assert False