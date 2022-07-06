# add by lzj 导出模型为onnx查看网络结构，发现CBAM模块被移动到最前面去了，怀疑是这方面的原因，但是不知道怎么解决这方面的问题
python tools/deployment/pytorch2onnx.py \
    configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py \
    ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640_baseline_0.4987mAP/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
    --output-file ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640_baseline_0.4987mAP/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 640 512

python tools/deployment/pytorch2onnx.py \
    configs2/VisDrone/lka_fpn/faster_rcnn_r50_lka_fpn_outch64_1x_VisDrone640.py \
    ../VisDrone_cache/lka_fpn/faster_rcnn_r50_lka_fpn_outch64_1x_VisDrone640/slice_640x640_lr0.08_1x_4g/best_bbox_mAP_epoch_12.pth \
    --output-file ../VisDrone_cache/lka_fpn/faster_rcnn_r50_lka_fpn_outch64_1x_VisDrone640/slice_640x640_lr0.08_1x_4g/best_bbox_mAP_epoch_12.onnx \
    --input-img data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg \
    --test-img data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg \
    --shape 640 512

python tools/deployment/pytorch2onnx.py \
    configs2/VisDrone/lka_fpn/faster_rcnn_r18_lka_fpn_1x_VisDrone640.py \
    ../VisDrone_cache/lka_fpn/cwd_bbop_fpnlvl_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640/slice_640x640_lr0.08_1x_4g/best_bbox_mAP_epoch_12.pth \
    --output-file ../VisDrone_cache/lka_fpn/cwd_bbop_fpnlvl_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640/slice_640x640_lr0.08_1x_4g/best_bbox_mAP_epoch_12.onnx \
    --input-img data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg \
    --test-img data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg \
    --shape 640 512