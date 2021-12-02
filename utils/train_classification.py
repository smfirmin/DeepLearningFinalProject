from __future__ import print_function

import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ModelNetDataset, ShapeNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)

        
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

num_classes = len(dataset.classes)
print("Training points:", len(dataset), " Test points:", len(test_dataset), " Classes:", num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform).cuda()
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if opt.model != '':
    checkpoint = torch.load(opt.model)
    classifier.load_state_dict(checkpoint['model_state_dict'])
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

# classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(start_epoch, start_epoch + opt.nepoch):
    print(f"\nEpoch {epoch}:")
    epochs.append(epoch)
    train_loss, test_loss, train_correct, test_correct, train_num, test_num = 0, 0, 0.0, 0.0, 0, 0
    classifier.train()

    for data in tqdm(dataloader):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        train_correct += correct.item()
        train_num += points.size()[0]

    scheduler.step()
    ave_train_loss = train_loss / len(dataloader) # all batch sizes may not be same
    train_losses.append(ave_train_loss)
    print('train loss: %f accuracy: %f' % (ave_train_loss, train_correct/train_num))

    with torch.no_grad():
        classifier.eval()
        for data in tqdm(testdataloader):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            test_loss += loss.item()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            test_correct += correct.item()
            test_num += points.size()[0]

    ave_test_loss = test_loss / len(testdataloader) # all batch sizes may not be same
    test_losses.append(ave_test_loss)
    print('%s loss: %f accuracy: %f' % (blue('test'), ave_test_loss, test_correct/test_num))
    
    torch.save({'model_state_dict': classifier.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'train_losses': train_losses,
                'test_losses': test_losses,
                'epochs': epochs},
                '%s/pointnet_%d.pth' % (opt.outf, epoch))

# plotting train and val losses
ax1 = plt.subplot()
ax1.plot(epochs, train_losses, label='train')
ax1.plot(epochs, test_losses, label='test')

ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')
# ax1.set_yscale('log')
ax1.legend()

ax1.set_title(f"epochs={opt.nepoch}, batchsz={opt.batchSize}")
plt.tight_layout()

plt.savefig(f"cls/training.png")
