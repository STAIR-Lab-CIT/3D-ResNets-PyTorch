# plain resnext101 with bunch of data augmentation
python3 tempactrec.py \
       --sample_duration 16 \
       --root_path /data4/SA4HDDv20170626 \
       --video_path jpg \
       --annotation_path 3dresnet_STAIRACTIONS.json \
       --result_path results_tempactrec_resnext101_112 \
       --dataset kinetics \
       --model resnext --model_depth 101 \
       --initial_scale 1.5 --n_scales 7 --scale_step 0.834 --train_crop 'random' \
       --n_classes 100 --sample_size 112 --n_val_samples 3 \
       --batch_size 32 --n_threads 8 --checkpoint 5 --manual_seed 13

