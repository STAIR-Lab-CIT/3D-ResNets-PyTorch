# plain resnext101 with bunch of data augmentation
# jumping frames
# this must be with n_val_samples = 1

python3 main.py \
       --sample_duration 16 \
       --root_path /data4/SA4HDDv20170626 \
       --video_path jpg \
       --annotation_path 3dresnet_STAIRACTIONS.json \
       --result_path results_jumpingframes_resnext101_112_050x050 \
       --dataset kinetics \
       --model resnext --model_depth 101 \
       --initial_scale 1.5 --n_scales 7 --scale_step 0.834 --train_crop 'random' \
       --n_classes 100 --sample_size 112 --n_val_samples 1 \
       --batch_size 128 --n_threads 16 --checkpoint 5 --manual_seed 13

