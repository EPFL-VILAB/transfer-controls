python scratch_imagenet.py \
  -a resnet50 \
  --lr 0.1 \
  --schedule 200000 400000 \
  --batch-size 256 \
  --max_images 100000 \
  --max_steps 500000 \
  --val_steps 1000 \
  --model_name scratch \
  --gpu 0 \
  /datasets/imagenet