CUDA_VISIBLE_DEVICES=0 python main.py \
                        --method vidt \
                        --backbone_name swin_tiny \
                        --epochs 12 \
                        --lr 1e-4 \
                        --min-lr 1e-7 \
                        --batch_size 7 \
                        --num_workers 14 \
                        --aux_loss True \
                        --with_box_refine True \
                        --coco_path /path/to/coco \
                        --output_dir /path/to/output/dir \
                        --start_epoch 0 \
                        --lr_drop 20 \
                        --warmup-epochs 0 \
                        --resume /path/to/model \
                        --dataset quickdraw