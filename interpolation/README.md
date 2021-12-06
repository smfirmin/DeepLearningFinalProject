# Meta-PU: An Arbitrary-Scale Upsampling Network for Point Cloud

Point cloud upsampling is vital for the quality of the mesh in three-dimensional reconstruction. Recent research on point cloud upsampling has achieved great success due to the development of deep learning. However, the existing methods regard point cloud upsampling of different scale factors as independent tasks. Thus, the methods need to train a specific model for each scale factor, which is both inefficient and impractical for storage and computation in real applications. To address this limitation, in this work, we propose a novel method called ``Meta-PU" to firstly support point cloud upsampling of arbitrary scale factors with a single model. In the Meta-PU method, besides the backbone network consisting of residual graph convolution (RGC) blocks, a meta-subnetwork is learned to adjust the weights of the RGC blocks dynamically, and a farthest sampling block is adopted to sample different numbers of points. Together, these two blocks enable our Meta-PU to continuously upsample the point cloud with arbitrary scale factors by using only a single model. In addition, the experiments reveal that training on multiple scales simultaneously is beneficial to each other. Thus, Meta-PU even outperforms the existing methods trained for a specific scale factor only.



[[tvcg paper]](https://arxiv.org/abs/2102.04317)

## Dataset Preparing

Put train dataset file Patches_noHole_and_collected.h5 into model/data/, you can download it from [onedrive train data](https://portland-my.sharepoint.com/:u:/g/personal/shuquanye2-c_my_cityu_edu_hk/Ec30f3ITZwdKuPzBQnTjhssBha_M2GI76_tnvoV5o1CO-g?e=LJiycf).

Unzip and put test dataset files all_testset.zip for variable scales into model/data/all_testset/, you can download it from [onedrive test data](https://portland-my.sharepoint.com/:u:/g/personal/shuquanye2-c_my_cityu_edu_hk/EUcCveufh7VMgQOLLOeqR4MBzXX6vGWbvjenT0H0nv_Ldw?e=GkyJVT).

## Environment & Installation

This codebase was tested with the following environment configurations.

- Ubuntu 18.04
- CUDA 10.1
- python v3.7
- torch>=1.0
- torchvision

`pip install -r requirements.txt`

`python setup.py build_ext --inplace`
or:
`pip install -e .`

## Training & Testing

Train:

`python main_gan.py --phase train --dataset model/data/Patches_noHole_and_collected.h5 --log_dir model/new --batch_size 8 --model model_res_mesh_pool --max_epoch 60 --gpu 0 --replace --FWWD --learning_rate 0.0001 --num_workers_each_gpu 3`

- You can easily reproduce the results in our paper with a batch size of only 8 in a single RTX 2080 Ti (11GB).

Test with scale R:

`python main_gan.py --phase test --dataset model/data/all_testset/${R}/input --log_dir model/new --batch_size 1 --model model_res_mesh_pool --model_path 60 --gpu 0 --test_scale ${R}`

Evaluation with scale R:

`cd evaluation_code/`

`python evaluation_cd.py --pre_path ../model/new/result/${R}input/ --gt_path model/data/all_testset/${R}/gt`

## Citing Meta-PU

`@article{Ye2021MetaPUAA,`

`title={Meta-PU: An Arbitrary-Scale Upsampling Network for Point Cloud},`

`author={S. Ye and Dongdong Chen and Songfang Han and Ziyu Wan and Jing Liao},` 

 `journal={IEEE transactions on visualization and computer graphics},`  

`year={2021},`

 `volume={PP}`

`}`

