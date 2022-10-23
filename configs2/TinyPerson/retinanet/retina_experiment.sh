export GPU=4 && LR=0.04 && CONFIG="retinanet_r50_fpn_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 NCCL_DEBUG=INFO tools/dist_train.sh configs2/TinyPerson/retinanet/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

export GPU=1 && LR=0.01 && CONFIG="retinanet_r50_fpn_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 NCCL_DEBUG=INFO tools/dist_train.sh configs2/TinyPerson/retinanet/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=1