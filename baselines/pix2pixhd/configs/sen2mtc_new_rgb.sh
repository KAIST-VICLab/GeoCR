# pix2pixHD on Sen2_MTC_New (RGB): 20000 updates at batch 32 on the trainval split.
TRAIN_ARGS="--dataroot geocr:sen2mtc_new_rgb --phase trainval --label_nc 0 --no_instance --resize_or_crop none \
 --loadSize 256 --fineSize 256 --netG global --batchSize 32 --lr 0.000475683 --max_steps 20000 \
 --gpu_ids 0 --nThreads 4"
