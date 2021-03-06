# plain resnext101 with bunch of data augmentation
python3 ft_main.py \
       --fine_tune resnext-101-kinetics.pth \
       --sample_duration 16 \
       --root_path /data4/SA4HDDv20170626 \
       --video_path jpg \
       --annotation_path 3dresnet_STAIRACTIONS.json \
       --result_path results_finetune_plain_resnext101_112_dur16_scl7_pj7 \
       --dataset kinetics --model resnext \
       --initial_scale 1.5 --n_scales 7 --scale_step 0.834 --train_crop 'random' \
       --projection 7 \
       --n_classes 100 --sample_size 112 --n_val_samples 3 \
       --model_depth 101 --batch_size 128 --n_threads 8 --checkpoint 5 --manual_seed 13

