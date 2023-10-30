# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
# from __future__ import annotations
from hashlib import new
import json
from pathlib import Path
from scipy import rand
import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
import datasets.transforms as T
# from tqdm import tqdm
import pickle
from collections import defaultdict
import random
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import time


import ndjson
import os


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def convert_to_np_raw(drawing, width=224, height=224):
    img = np.zeros((width, height))
    pil_img = convert_to_PIL(drawing)
    pil_img.thumbnail((width, height), Image.ANTIALIAS)
    pil_img = pil_img.convert('RGB')
    pixels = pil_img.load()

    for i in range(0, width):
        for j in range(0, height):
            img[i,j] = 1- pixels[j,i][0]/255.0
    # return img
    return pil_img

def convert_to_PIL(drawing, width=224, height=224): # 256 before
    pil_img = Image.new('RGB', (width, height), 'white')
    pixels = pil_img.load()
    draw = ImageDraw.Draw(pil_img)
    for x,y in drawing:
        for i in range(1, len(x)):
            draw.line((x[i-1], y[i-1], x[i], y[i]), fill=0)
    return pil_img

def normalize_transform():
    return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


def make_coco_transforms_for_eval(image_set, args):
    
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        scales = [480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800]
    
        print("Resolution: shortest at most", max(scales))
        if image_set == 'train':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                    ])
                ),
                normalize,
            ])
    
        print(args.eval_size)
    
        if image_set == 'val':
            return T.Compose([
                T.RandomResize([args.eval_size], max_size=args.eval_size * 1333 // 800),
                normalize,
            ])
    
        raise ValueError(f'unknown {image_set}')

class CocoDetectionQD(torchvision.datasets.CocoDetection):
    def __init__(self, image_set, img_folder, ann_file, transforms, return_masks):
        json_file = json.load(open(ann_file))
        ROOT = '/path/to/coco/annotations'
        classes = json_file['categories']
        self.id2class = {}
        self.class2id = {}

        for cat in classes:
            self.id2class[cat['id']] = cat['name']
            self.class2id[cat['name']] = cat['id']

        self.all_categories = ['elephant', 'bear', 'cat', 'zebra', 'bus', 'horse', 'giraffe', 'airplane', 'bed', 'dog', 'scissors', 'train', 'sandwich', 'pizza', 'cow',
                                'broccoli', 'umbrella', 'sheep', 'bird', 'stop sign', 'toothbrush', 'bicycle', 'hot dog', 'laptop', 'toaster', 'microwave', 'banana', 'baseball bat',
                                'donut', 'couch', 'keyboard', 'cake', 'oven', 'carrot', 'bench', 'suitcase', 'fire hydrant', 'fork', 'chair', 'wine glass', 'apple', 'truck', 'cell phone',
                                'cup', 'car', 'knife', 'toilet', 'clock', 'backpack', 'spoon', 'vase', 'book', 'skateboard', 'sink', 'mouse', 'traffic light']
        
   
        unseen_cats = [self.all_categories[i] for i in range(len(self.all_categories)) if i%4==0]
        images = []
        annotate = []
        selected_image_ids = []
        to_remove_image_ids = []

        for anno in json_file['annotations']:
            img_id = anno['image_id']
            if self.id2class[anno['category_id']] in self.all_categories:
                annotate.append(anno)
                selected_image_ids.append(img_id)
            

        selected_image_ids = set(selected_image_ids)
        
        for image in json_file['images']:
            img_id = image['id']
            if img_id in selected_image_ids:
                images.append(image)
        
        json_file['annotations'] = annotate
        json_file['images'] = images
        json.dump(json_file, open(ROOT+'/'+'temp_json_file_'+image_set+'.json', 'w'))
        temp_ann_file = os.path.join(ROOT, 'temp_json_file_'+image_set+'.json')
        super(CocoDetectionQD, self).__init__(img_folder, temp_ann_file)
        self.image_set = image_set
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)


        self.transforms_sketch = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize_transform()
            ])
        _quickdraw_path = "/path/to/processed_quick_draw_paths_purified.pkl"
        _quickdraw_path = pickle.load(open(_quickdraw_path, 'rb'))

        print("Loading Quick,Draw! ...")
        if image_set == 'train':
            _quickdraw_path = _quickdraw_path['train_x']
        else:
            _quickdraw_path = _quickdraw_path['valid_x']

        self.class2quick = defaultdict(list)
        
        for path in _quickdraw_path:
            cat = path.split('/')[-2]
            self.class2quick[cat].append(path)
        self.image_set = image_set

    def __getitem__(self, idx):
        img, target = super(CocoDetectionQD, self).__getitem__(idx)
        if not self.image_set == 'train':
            random.seed(14)
        else:
            t = 1000 * time.time()
            random.seed(t)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        # img.save("saved_images/target_image"+str(idx)+".png")
        img, target = self.prepare(img, target)
        categories = list(set(target['labels'].tolist()))
        new_target = {}
        selected_cat = random.choice(categories)
        
        keep = target['labels']==selected_cat

        selected_keys = ['boxes', 'labels', 'area', 'iscrowd', 'masks']


        for key, value in target.items():
            if key in selected_keys:
                new_target[key] = value[keep]
            else:
                new_target[key] = value
        
        new_target['labels'] = torch.ones_like(new_target['labels'])
        
        selected_cat = self.id2class[selected_cat]

        sketches = random.choices(self.class2quick[selected_cat], k=5)
        sketch_list = []
        i = 0
        for sketch in sketches:

            sketch = pickle.load(open(sketch, 'rb'))
            key = list(sketch.keys())[0]
            sketch = convert_to_np_raw(sketch[key])
            if i == 2:
                sketch.save("saved_images_local_5/query_sketch"+str(idx)+".png")
            i += 1
            sketch = 255 - np.asarray(sketch)
            sketch = Image.fromarray(sketch)
            sketch = self.transforms_sketch(sketch)
            sketch_list.append(sketch.unsqueeze(0))
        sketch_list = torch.cat(sketch_list, dim=0)

        old_boxes = new_target['boxes'].clone()
        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)

        if not self.image_set == 'train':
            new_target['new_boxes'] = new_target['boxes']
            new_target['boxes'] = old_boxes

        return img, new_target, sketch_list


class CocoDetectionSketchy(torchvision.datasets.CocoDetection):
    def __init__(self, image_set, img_folder, ann_file, transforms, return_masks):
        # super(CocoDetection, self).__init__(img_folder, ann_file)
        json_file = json.load(open(ann_file))
        ROOT = '/path/to/coco/annotations'
        classes = json_file['categories']
        sketchy_path = "/path/to/sketchy_dataset.pkl"
        sketchy_path = pickle.load(open(sketchy_path, 'rb'))
        self.id2class = {}
        self.class2id = {}

        for cat in classes:
            self.id2class[cat['id']] = cat['name']
            self.class2id[cat['name']] = cat['id']

        self.all_categories = ['elephant', 'bear', 'cat', 'zebra',  'horse', 'giraffe', 'airplane', 'dog', 'scissors', 'pizza', 'cow',
                                'umbrella', 'sheep', 'bicycle', 'hot dog', 'banana', 'couch','bench', 'chair', 'apple','cup', 'car', 
                                'knife', 'clock', 'spoon', 'mouse', 'motorcycle']
        

        images = []
        annotate = []
        selected_image_ids = []
        to_remove_image_ids = []

        for anno in json_file['annotations']:
            img_id = anno['image_id']
            if self.id2class[anno['category_id']] in self.all_categories:
                annotate.append(anno)
                selected_image_ids.append(img_id)

        to_remove_image_ids  = []
        selected_image_ids = set(selected_image_ids)
                
        for image in json_file['images']:
            img_id = image['id']
            if img_id in selected_image_ids:
                images.append(image)
        
        json_file['annotations'] = annotate
        json_file['images'] = images
        json.dump(json_file, open(ROOT+'/'+'temp_json_file_'+image_set+'.json', 'w'))
        temp_ann_file = os.path.join(ROOT, 'temp_json_file_'+image_set+'.json')
        super(CocoDetectionSketchy, self).__init__(img_folder, temp_ann_file)
        self.image_set = image_set
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)


        self.transforms_sketch = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            normalize_transform()
            ])
        
        print("Loading Sketchy! ...")

        if image_set == 'train':
            self.class2quick = sketchy_path['train']
        else:
            self.class2quick = sketchy_path['valid']
        
        self.sketchy_root = "/path/to/sketchy/images"
        self.image_set = image_set
    

    def __getitem__(self, idx):
        img, target = super(CocoDetectionSketchy, self).__getitem__(idx)
        if not self.image_set == 'train':
            random.seed(14)
        else:
            t = 1000 * time.time()
            random.seed(t)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        categories = list(set(target['labels'].tolist()))
        new_target = {}
        selected_cat = random.choice(categories)
        
        keep = target['labels']==selected_cat

        selected_keys = ['boxes', 'labels', 'area', 'iscrowd', 'masks']


        for key, value in target.items():
            if key in selected_keys:
                new_target[key] = value[keep]
            else:
                new_target[key] = value
        
        new_target['labels'] = torch.ones_like(new_target['labels'])
        
        selected_cat = self.id2class[selected_cat]

        sketches = random.choices(self.class2quick[selected_cat], k=5)
        sketch_list = []
        for sketch in sketches:
            sketch = Image.open(os.path.join(self.sketchy_root, sketch+'.png')).convert('RGB')
            sketch = self.transforms_sketch(sketch)


            sketch_list.append(sketch.unsqueeze(0))

        sketch_list = sketch_list[0].squeeze(0)
        
        
        old_boxes = new_target['boxes'].clone()
        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)

        if not self.image_set == 'train':
            new_target['new_boxes'] = new_target['boxes']
            new_target['boxes'] = old_boxes        

        return img, new_target, sketch_list

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800]

    print("Resolution: shortest at most", max(scales))
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                ])
            ),
            normalize,
        ])

    print(args.eval_size)

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([args.eval_size], max_size=args.eval_size * 1333 // 800),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(image_set, img_folder, ann_file,
                            transforms=make_coco_transforms(image_set, args),
                            return_masks=True)
    return dataset
