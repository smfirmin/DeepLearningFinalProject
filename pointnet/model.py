from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, device):
        super(STN3d, self).__init__()
        self.device = device
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        """
        x: [batch_size, dimension, npoints]
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.tensor([1,0,0,0,1,0,0,0,1], device=self.device).view(1,9).repeat(batchsize,1)

        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, device, k=64):
        super(STNkd, self).__init__()
        self.device = device
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        """
        x: [batch_size, dimension, npoints]
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=self.device).flatten().view(1,self.k*self.k).repeat(batchsize,1)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class AttnEncoderBlock(nn.Module):
    def __init__(self, device, embed_dim=64, num_heads=1, norm='batch1d', dim_ff=128):
        super(AttnEncoderBlock, self).__init__()
        self.device = device
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dim_ff = dim_ff
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
        self.ff1 = nn.Linear(64, self.dim_ff)
        self.ff2 = nn.Linear(self.dim_ff, 64)
        self.relu = nn.ReLU()

        if norm == "batch1d":
            self.norm1 = nn.BatchNorm1d(2500)
            self.norm2 = nn.BatchNorm1d(2500)
        elif norm == "layer":
            self.norm1 = nn.LayerNorm(64)
            self.norm2 = nn.LayerNorm(64)

    def forward(self, x):
        x = x.transpose(2, 1)
        sa = self.attn(x,x,x)[0]
        x = self.norm1(x + sa)
        x = self.norm2(x + self.ff2(self.relu(self.ff1(x))))
        x = x.transpose(2, 1)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, device, global_feat = True, feature_transform = False, attention = False):
        super(PointNetfeat, self).__init__()
        self.device = device
        self.stn = STN3d(self.device)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        if self.feature_transform:
            self.fstn = STNkd(device, k=64)

        self.attention = attention
        if self.attention:
            self.attn1 = AttnEncoderBlock(self.device)         
            self.attn2 = AttnEncoderBlock(self.device)            

    def forward(self, x):
        """
        x: [batch_size, dimension, npoints]
        """
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        if self.attention:
            x = self.attn1(x)
            x = self.attn2(x)

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, device, k=2, feature_transform=False, attention=False):
        super(PointNetCls, self).__init__()
        self.device = device
        self.feature_transform = feature_transform
        self.attention = attention
        self.feat = PointNetfeat(self.device, global_feat=True, feature_transform=feature_transform, attention=self.attention)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: [batch_size, dimension, npoints]
        """
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, device, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.device = device
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(self.device, global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans, device):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d, device=device)[None, :, :]

    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':

    DEVICE= 'cpu'

    sim_data = torch.rand(32,3,2500, device=DEVICE)
    trans = STN3d(DEVICE).to(DEVICE)
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out, DEVICE))

    sim_data_64d = torch.rand(32, 64, 2500, device=DEVICE)
    trans = STNkd(DEVICE, k=64).to(DEVICE)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out, DEVICE))

    pointfeat = PointNetfeat(DEVICE, global_feat=True).to(DEVICE)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(DEVICE, global_feat=False).to(DEVICE)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(DEVICE, k = 5).to(DEVICE)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(DEVICE, k = 3).to(DEVICE)
    out, _, _ = seg(sim_data)
    print('seg', out.size())


# https://github.com/princeton-vl/SimpleView/blob/master/models/pointnet.py
class PointNet(nn.Module):

    def __init__(self, dataset, task, device, feature_transform=True, attention=True):
        super().__init__()
        self.task = task
        self.device = device
        self.feature_transform = feature_transform
        self.attention = attention
        num_class = len(dataset.classes)
        self.model =  PointNetCls(self.device, k=num_class, feature_transform=self.feature_transform, attention=self.attention)

    def forward(self, pc, cls=None):
        pc = pc.transpose(2, 1).float()
        logit, _, trans_feat = self.model(pc)
        out = {'logit': logit, 'trans_feat': trans_feat}
        return out