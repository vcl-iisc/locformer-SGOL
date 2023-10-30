## Acknowledgement
This codebase has been developed on [ViDT](https://github.com/peternara/vidt-transformer-detection). We thanks the authors of ViDT for providing the code.

## Installation Instructions
This codebase has been developed with the setting used in [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR): <br>Linux, CUDA>=9.2, GCC>=5.4, Python>=3.7, PyTorch>=1.5.1, and torchvision>=0.6.1.

We recommend you to use Anaconda to create a conda environment:
```bash
conda create -n deformable_detr python=3.7 pip
conda activate deformable_detr
conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
```

#### Compiling CUDA operators for deformable attention
```bash
cd ./ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

#### Other requirements
```bash
pip install -r requirements.txt
```

## Training

The training script is provided in this code base. Run the following script

```
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
                        
```

## Evaluation

For evaluation run the following script

```
CUDA_VISIBLE_DEVICES=0 python main.py \
                        --method vidt \
                        --backbone_name swin_tiny \
                        --batch_size 7 \
                        --num_workers 14 \
                        --coco_path /path/to/coco \
                        --output_dir /path/to/output/dir \
                        --resume /path/to/model \
                        --eval True
                     
```

## Datasets

We use the folowing datasets in our work:

1. [MS-COCO](https://cocodataset.org/)
2. [Sketchy](https://sketchy.eye.gatech.edu/)
3. [QuickDraw](https://quickdraw.withgoogle.com/data)
