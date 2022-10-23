docker run --gpus all --shm-size=8g -d\
 -v /home/lzj/data:/mmdetection/data \
 -v /home/lzj/mmdetection:/mmdetection \
 -v /home/lzj/TOV_mmdetection_cache:/TOV_mmdetection_cache \
 -v /home/lzj/VisDrone_cache:/VisDrone_cache \
 mmdetection