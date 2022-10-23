export GPU=1 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_VisDrone640"
tools/dist_train.sh configs2/VisDrone/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=16

export GPU=1 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_VisDrone640"
python tools/test.py configs2/VisDrone/lka_fpn/${CONFIG}.py \
     ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12.pth \
    --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
    --show-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/test_result \
    --show-score-thr 0.7 --fuse-conv-bn

export GPU=4 && LR=0.16 && CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
python tools/test.py configs2/VisDrone/lka_fpn/${CONFIG}.py \
     ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12.pth \
    --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
    --show-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/test_result \
    --show-score-thr 0.7 --fuse-conv-bn

# 知识蒸馏后的模型的可视化效果
# 没看到结果
export GPU=4 && LR=0.08 && CONFIG="cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
python tools/test.py configs2/VisDrone/lka_fpn/faster_rcnn_r18_lka_fpn_1x_VisDrone640.py \
     ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
    --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
    --show-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/test_result \
    --show-score-thr 0.7 --fuse-conv-bn
# OrderedDict([('bbox_mAP', 0.176), ('bbox_mAP_50', 0.363), ('bbox_mAP_75', 0.155), ('bbox_mAP_s', 0.1), ('bbox_mAP_m', 0.263), ('bbox_mAP_l', 0.314),
# ('bbox_mAP_copypaste', '0.176 0.363 0.155 0.100 0.263 0.314')])
# 是不是因为show-score-thr搞的点低了，换掉试试
export GPU=4 && LR=0.08 && CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
python tools/test.py configs2/VisDrone/lka_fpn/faster_rcnn_r18_lka_fpn_1x_VisDrone640.py \
     ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
    --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
    --show-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/test_result \
    --show-score-thr 0.7 --fuse-conv-bn --eval bbox

# OrderedDict([('bbox_mAP', 0.176), ('bbox_mAP_50', 0.363), ('bbox_mAP_75', 0.155), ('bbox_mAP_s', 0.1), ('bbox_mAP_m', 0.263), ('bbox_mAP_l', 0.314),
# ('bbox_mAP_copypaste', '0.176 0.363 0.155 0.100 0.263 0.314')])
python tools/test.py configs2/VisDrone/lka_fpn/faster_rcnn_r18_lka_fpn_1x_VisDrone640.py \
     ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
    --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
    --eval bbox
# haishi 0.176
export GPU=4 && LR=0.08 CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 python tools/test.py configs2/VisDrone/lka_fpn/${CONFIG}.py \
   ../VisDrone_cache/lka_fpn/cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
  --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
  --eval bbox

# yeshi 0.176……
# 理轩指正后对了，OrderedDict([('bbox_mAP', 0.213), ('bbox_mAP_50', 0.436), ('bbox_mAP_75', 0.186), ('bbox_mAP_s', 0.137), ('bbox_mAP_m', 0.308), ('bbox_mAP_l', 0.367),
# ('bbox_mAP_copypaste', '0.213 0.436 0.186 0.137 0.308 0.367')])
export GPU=4 && LR=0.08 CONFIG="test_frcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 python tools/test.py configs2/VisDrone/lka_fpn/${CONFIG}.py \
   ../VisDrone_cache/lka_fpn/cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
  --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
  --eval bbox

export GPU=4 && LR=0.08 CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 python tools/test.py configs2/VisDrone/lka_fpn/${CONFIG}.py \
   ../VisDrone_cache/lka_fpn/cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
  --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
  --eval bbox