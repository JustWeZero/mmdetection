_base_ = 'faster_rcnn_r50_lka_fpn_1x_VisDrone640.py'

model = dict(
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=True,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'
    ),
    neck=dict(
        type='lka_FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5)
)