# batch_size 128 crashes python nccl
#    10G+ for each 16 GPU
# batch_size 64 -> 1G for each 16 GPU
# whatever batch_size, at the end of epoch the program crashes with NcclError
# This is true for cuda-9.0 and 8.0
# torch.cuda.nccl.NcclError: Unhandled Cuda Error (1)
# this is for resized stair actions dataset
<<EOF 
 python3 main.py \
       --root_path /home/ubuntu/data3 \
       --video_path jpg \
       --annotation_path stairactions.json \
       --result_path results \
       --dataset kinetics --model resnet \
       --n_classes 100 --sample_size 112 --n_val_samples 3 \
       --model_depth 34 --batch_size 64 --n_threads 4 --checkpoint 5 --manual_seed 13

EOF

# this is for yoshikawa datasets (after removing duplicates)
# batch_size 128 -> unhandled cuda error
python3 main.py \
       --sample_duration 16 \
       --root_path /data4/SA4HDDv20170626 \
       --video_path jpg \
       --annotation_path 3dresnet_STAIRACTIONS.json \
       --result_path results_resnext152_224_dur16_scl7_pj7 \
       --dataset kinetics --model resnext \
       --initial_scale 1.5 --n_scales 7 --scale_step 0.834 --train_crop 'random' \
       --projection 7 \
       --n_classes 100 --sample_size 224 --n_val_samples 3 \
       --model_depth 152 --batch_size 64 --n_threads 8 --checkpoint 5 --manual_seed 13
