# Done image [100/ 2000], fps: 9.7 img / s, times per image: 103.6 ms / img
# 精度需要再提一下，用蒸馏看看能不能提。这个速度抽帧也许能搞一下。
# backbone换成shufflenetv2或者是apple的repvgg速度能不能更快呢？
export GPU=4 && LR=0.08 && CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIABLE_DEVICE=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs2/VisDrone/lka_fpn/${CONFIG}.py \
       ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/epoch_12.pth \
       --launcher pytorch

# Done image [50 / 2000], fps: 4.5 img / s, times per image: 222.4 ms / img
# teacher网络的速度减慢了一半多，但是结果倒是还行。
export GPU=4 && LR=0.08 && CONFIG="faster_rcnn_r50_lka_fpn_outch64_1x_VisDrone640"
CUDA_VISIABLE_DEVICE=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs2/VisDrone/lka_fpn/${CONFIG}.py \
       ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/epoch_12.pth \
       --launcher pytorch

# Done image [50 / 2000], fps: 3.3 img / s, times per image: 302.7 ms / img
export GPU=4 && LR=0.08 && CONFIG="faster_rcnn_r18_lka_fpn_outch256_1x_VisDrone640"
CUDA_VISIABLE_DEVICE=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs2/VisDrone/lka_fpn/${CONFIG}.py \
       ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/epoch_12.pth \
       --launcher pytorch

# Done image [100/ 2000], fps: 2.4 img / s, times per image: 412.3 ms / img
export GPU=1 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_VisDrone640"
CUDA_VISIABLE_DEVICE=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs2/VisDrone/lka_fpn/${CONFIG}.py \
       ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/epoch_12.pth \
       --launcher pytorch

# Done image [150/ 2000], fps: 3.1 img / s, times per image: 319.3 ms / img
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_fpn_1x_VisDrone640"
CUDA_VISIABLE_DEVICE=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs2/VisDrone/base/${CONFIG}.py \
       ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g//epoch_12.pth \
       --launcher pytorch