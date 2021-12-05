# from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import meshio
from pathlib import Path
import numpy as np
import sys
import json
from plyfile import PlyData

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True,
                 convert_off_to_ply=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), './misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

        if convert_off_to_ply:
            p = Path(self.root)
            print('searching path', p, 'to convert .off files to .ply')
            l = list(p.glob('**/*.off'))
            for in_file in l:
                print(in_file)
                with in_file.open() as f:
                    # some OFF files in original dataset had OFF345 345 344 where 
                    # OFF collided with the number. Needs \n
                    lines = f.readlines()
                if lines[0] != 'OFF\n':
                    # print(lines[0:3])
                    lines[0] = lines[0][0:3] + '\n' + lines[0][3:]
                    # print(lines[0:3])
                    with in_file.open('w') as f:
                        lines = "".join(lines)
                        f.write(lines)

                mesh = meshio.read(in_file, file_format="off")
                # out_file = in_file.replace(in_file.with_suffix('.ply'))
                out_file = in_file.with_suffix('.ply')

                print(out_file)
                mesh.write(out_file, file_format='ply')

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

        # I think we can experiment with this ordering
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

