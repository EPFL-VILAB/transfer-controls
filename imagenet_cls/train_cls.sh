python imagenet_ssl_epoch.py \
    -a resnet50 \
    --pretrained \
    --seed 0 \
    --epochs 100 \
    --warmup 5 \
    --lr_step 30 \
    --lr 0.1 \
    --batch-size 256 \
    --max_images 15000 \
    --model_name pirl \
    --gpu 0 \
    /datasets/imagenet