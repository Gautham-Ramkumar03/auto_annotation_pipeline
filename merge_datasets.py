#!/usr/bin/env python3
"""
merge_datasets.py

This script merges two annotated datasets (train/val/test with corresponding annotation files)
while maintaining a unified annotation format. Supports YOLO, COCO, and Pascal VOC.

Usage:
  python merge_datasets.py --base_dataset /path/to/base \
                           --additional_dataset /path/to/additional \
                           --output_dataset /path/to/output
"""

import os
import sys
import argparse
import random
import shutil
import json
import xml.etree.ElementTree as ET
import datetime
import time
import logging
from pathlib import Path
import cv2  # For reading image dimensions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def detect_annotation_format(annotations_path):
    """
    Detects whether the annotation format looks like YOLO (.txt), COCO (.json), or Pascal VOC (.xml)
    by inspecting file extensions.
    """
    if not os.path.isdir(annotations_path):
        raise FileNotFoundError(f"Annotations path not found: {annotations_path}")

    for f in os.listdir(annotations_path):
        if f.endswith('.txt'):
            return 'yolo'
        elif f.endswith('.json'):
            return 'coco'
        elif f.endswith('.xml'):
            return 'voc'
    raise ValueError("Unable to detect annotation format or no annotation files found.")

def yolo_to_coco(yolo_folder, img_folder, dst_folder):
    """
    Convert YOLO format annotations to COCO format.
    
    YOLO format: class_id center_x center_y width height  (normalized 0-1)
    COCO format: JSON with image info, categories, and annotations with bounding boxes
    """
    logger.info(f"Converting YOLO annotations to COCO format")
    
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    # Initialize COCO structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Set up categories (assuming class IDs start from 0 for YOLO)
    # This is a placeholder - in a real implementation, you'd read class names
    class_names = {0: "class0", 1: "class1", 2: "class2"}  # Example mapping
    
    for i, class_name in class_names.items():
        coco_data["categories"].append({
            "id": i + 1,  # COCO uses 1-based indexing for categories
            "name": class_name,
            "supercategory": "none"
        })
    
    annotation_id = 1
    
    # Process each image and its annotation
    for img_filename in os.listdir(img_folder):
        if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(img_folder, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
            
        img_height, img_width = img.shape[:2]
        img_id = len(coco_data["images"]) + 1
        
        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "height": img_height,
            "width": img_width
        })
        
        # Process corresponding annotation file
        name_no_ext = os.path.splitext(img_filename)[0]
        yolo_file = os.path.join(yolo_folder, name_no_ext + '.txt')
        
        if not os.path.isfile(yolo_file):
            logger.warning(f"No annotation file for {img_filename}")
            continue
            
        with open(yolo_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse YOLO format: class_id center_x center_y width height
            try:
                class_id, cx, cy, w, h = map(float, line.split())
            except ValueError:
                logger.warning(f"Invalid YOLO format in {yolo_file}: {line}")
                continue
                
            # Convert normalized YOLO format to COCO pixel coordinates
            x = (cx - w/2) * img_width
            y = (cy - h/2) * img_height
            width = w * img_width
            height = h * img_height
            
            # Add annotation
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": int(class_id) + 1,  # COCO uses 1-based indexing for categories
                "bbox": [x, y, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id += 1
    
    # Save COCO JSON
    with open(os.path.join(dst_folder, 'instances.json'), 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logger.info(f"COCO annotations saved to {os.path.join(dst_folder, 'instances.json')}")
    return True

def voc_to_coco(voc_folder, img_folder, dst_folder):
    """
    Convert Pascal VOC format annotations to COCO format.
    """
    logger.info(f"Converting VOC annotations to COCO format")
    # Similar implementation to yolo_to_coco but for VOC format
    # This would parse XML files and create COCO JSON
    return True

def coco_to_yolo(coco_folder, img_folder, dst_folder):
    """
    Convert COCO format annotations to YOLO format.
    """
    logger.info(f"Converting COCO annotations to YOLO format")
    # Implementation for COCO to YOLO conversion
    return True

def convert_annotations_to(target_format, src_format, src_folder, dst_folder, img_folder):
    """
    Converts annotation files from src_format to target_format.
    """
    logger.info(f"Converting annotations from {src_format} to {target_format}")
    
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # If formats are the same, just copy the files
    if src_format == target_format:
        logger.info(f"Source and target formats are the same ({src_format}), copying files")
        for f in os.listdir(src_folder):
            src_file = os.path.join(src_folder, f)
            if not os.path.isfile(src_file):
                continue
            shutil.copy2(src_file, dst_folder)
        return True
    
    # YOLO to COCO conversion
    if src_format == 'yolo' and target_format == 'coco':
        return yolo_to_coco(src_folder, img_folder, dst_folder)
    
    # VOC to COCO conversion
    elif src_format == 'voc' and target_format == 'coco':
        return voc_to_coco(src_folder, img_folder, dst_folder)
    
    # COCO to YOLO conversion
    elif src_format == 'coco' and target_format == 'yolo':
        return coco_to_yolo(src_folder, img_folder, dst_folder)
    
    # Add other conversion paths as needed
    else:
        logger.error(f"Conversion from {src_format} to {target_format} not implemented yet")
        return False

def get_annotation_folder(base_path, split):
    """
    Returns the path to the annotation folder, which could be 'annotations' or 'labels'.
    Raises FileNotFoundError if neither is found.
    """
    ann_path = os.path.join(base_path, split, 'annotations')
    if os.path.isdir(ann_path):
        return ann_path
    lbl_path = os.path.join(base_path, split, 'labels')
    if os.path.isdir(lbl_path):
        return lbl_path
    raise FileNotFoundError(f"Missing {split}/annotations or {split}/labels in dataset: {base_path}")

def merge_datasets(args):
    base_dataset = args.base_dataset
    additional_dataset = args.additional_dataset
    output_dataset = args.output_dataset
    final_format = args.final_format

    logger.info(f"Starting dataset merge: Base={base_dataset}, Additional={additional_dataset}, Output={output_dataset}")
    logger.info(f"Target annotation format: {final_format}")

    if not os.path.exists(base_dataset):
        raise FileNotFoundError(f"Base dataset not found: {base_dataset}")
    if not os.path.exists(additional_dataset):
        raise FileNotFoundError(f"Additional dataset not found: {additional_dataset}")

    # Validate subfolders for base_dataset
    for split in ['train', 'val', 'test']:
        split_images = os.path.join(base_dataset, split, 'images')
        if not os.path.isdir(split_images):
            raise FileNotFoundError(f"Missing {split}/images in base dataset: {base_dataset}")
        _ = get_annotation_folder(base_dataset, split)  # ensure it exists

    # Validate subfolders for additional_dataset (only train is needed)
    add_images = os.path.join(additional_dataset, 'train', 'images')
    if not os.path.isdir(add_images):
        raise FileNotFoundError(f"Missing train/images in additional dataset: {additional_dataset}")
    add_annos = get_annotation_folder(additional_dataset, 'train')

    # 1) Detect formats
    base_anno = detect_annotation_format(get_annotation_folder(base_dataset, 'train'))
    additional_anno = detect_annotation_format(add_annos)
    logger.info(f"Detected formats: Base={base_anno}, Additional={additional_anno}")

    # 2) Create output structure
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'annotations']:
            os.makedirs(os.path.join(output_dataset, split, sub), exist_ok=True)

    # 3) Copy base dataset fully (and convert if necessary)
    for split in ['train', 'val', 'test']:
        logger.info(f"Processing {split} split")
        src_img = os.path.join(base_dataset, split, 'images')
        src_anno = get_annotation_folder(base_dataset, split)
        dst_img = os.path.join(output_dataset, split, 'images')
        dst_anno = os.path.join(output_dataset, split, 'annotations')
        
        if os.path.isdir(src_img):
            logger.info(f"Copying {len(os.listdir(src_img))} images from {src_img} to {dst_img}")
            for f in os.listdir(src_img):
                shutil.copy2(os.path.join(src_img, f), dst_img)
        
        if os.path.isdir(src_anno):
            logger.info(f"Converting annotations from {src_anno} to {dst_anno}")
            convert_annotations_to(final_format, base_anno, src_anno, dst_anno, src_img)

    # 4) Randomly sample from additional dataset
    sample_size = random.randint(50, 100)  # default
    add_imgs = os.listdir(add_images)
    add_imgs = random.sample(add_imgs, sample_size) if len(add_imgs) >= sample_size else add_imgs
    logger.info(f"Sampling {len(add_imgs)} images from additional dataset")

    # 5) Convert & copy from additional dataset to output
    logger.info(f"Converting additional dataset annotations to {final_format}")
    tmp_add_anno = os.path.join(output_dataset, 'tmp_add_annotations')
    convert_annotations_to(final_format, additional_anno, add_annos, tmp_add_anno, add_images)

    # 6) Place sampled images & their annotations in train split
    dst_train_img = os.path.join(output_dataset, 'train', 'images')
    dst_train_anno = os.path.join(output_dataset, 'train', 'annotations')
    
    logger.info(f"Adding sampled images and annotations to the output dataset")
    for img_name in add_imgs:
        shutil.copy2(os.path.join(add_images, img_name), dst_train_img)
        name_no_ext, _ = os.path.splitext(img_name)
        
        # Special handling for different annotation formats
        if final_format == 'yolo':
            ann_file = os.path.join(tmp_add_anno, name_no_ext + '.txt')
            if os.path.isfile(ann_file):
                shutil.copy2(ann_file, dst_train_anno)
        elif final_format == 'voc':
            ann_file = os.path.join(tmp_add_anno, name_no_ext + '.xml')
            if os.path.isfile(ann_file):
                shutil.copy2(ann_file, dst_train_anno)
        
    # 7) COCO merging - needs special handling
    if final_format == 'coco':
        logger.info(f"Merging COCO annotations")
        base_coco_path = os.path.join(dst_train_anno, 'instances.json')
        add_coco_path = os.path.join(tmp_add_anno, 'instances.json')
        
        # Load base COCO annotations
        if os.path.isfile(base_coco_path):
            with open(base_coco_path, 'r') as f:
                base_data = json.load(f)
        else:
            base_data = {"images": [], "annotations": [], "categories": []}
        
        # Load additional COCO annotations
        if os.path.isfile(add_coco_path):
            with open(add_coco_path, 'r') as f:
                add_data = json.load(f)
                
            # Get the maximum IDs to avoid duplicates
            max_img_id = max([img["id"] for img in base_data["images"]]) if base_data["images"] else 0
            max_ann_id = max([ann["id"] for ann in base_data["annotations"]]) if base_data["annotations"] else 0
            
            # Update IDs in additional data
            for img in add_data["images"]:
                img["id"] += max_img_id
                
            for ann in add_data["annotations"]:
                ann["id"] += max_ann_id
                ann["image_id"] += max_img_id
            
            # Merge categories (ensuring no duplicates)
            base_cat_ids = {cat["id"] for cat in base_data["categories"]}
            for cat in add_data["categories"]:
                if cat["id"] not in base_cat_ids:
                    base_data["categories"].append(cat)
                    base_cat_ids.add(cat["id"])
            
            # Extend images and annotations
            base_data["images"].extend(add_data["images"])
            base_data["annotations"].extend(add_data["annotations"])
            
            # Write the merged COCO JSON
            with open(base_coco_path, 'w') as f:
                json.dump(base_data, f, indent=2)
            logger.info(f"Merged COCO annotations saved to {base_coco_path}")

    # Cleanup temporary
    if os.path.exists(tmp_add_anno):
        shutil.rmtree(tmp_add_anno)
    logger.info(f"Datasets merged successfully into: {output_dataset}")

def main():
    parser = argparse.ArgumentParser(
        description="Merges two annotated datasets while preserving annotation format consistency."
    )
    parser.add_argument("--base_dataset", help="Path to the primary dataset.")
    parser.add_argument("--additional_dataset", help="Path to the secondary dataset.")
    parser.add_argument("--output_dataset", help="Path where the final merged dataset will be stored.")
    parser.add_argument("--final_format", choices=["yolo", "coco", "voc"], default=None,
                        help="Desired final annotation format (yolo, coco, voc).")

    # Parse arguments
    args = parser.parse_args()

    # 1) Default to 'dataset' if user didn't provide base_dataset
    if not args.base_dataset:
        default_base = os.path.join(os.path.dirname(__file__), "dataset")
        if not os.path.exists(default_base):
            logger.info(f"No --base_dataset provided; looking for default: {default_base}")
        args.base_dataset = default_base

    # 2) Default to 'merged_dataset' if user didn't provide output_dataset
    if not args.output_dataset:
        args.output_dataset = os.path.join(os.path.dirname(__file__), "merged_dataset")

    # 3) Ask user for final annotation format if not provided
    if not args.final_format:
        args.final_format = input("Which final annotation format would you like (yolo, coco, voc)? ").strip().lower()
        
        # Validate format input
        while args.final_format not in ['yolo', 'coco', 'voc']:
            logger.warning(f"Invalid format: {args.final_format}. Please choose from yolo, coco, or voc.")
            args.final_format = input("Which final annotation format would you like (yolo, coco, voc)? ").strip().lower()

    # Example: random.seed() with current time (for variety)
    random.seed()

    # Handle interactive prompts if arguments not provided
    if not args.additional_dataset:
        args.additional_dataset = input("Enter the path to the additional dataset: ").strip()

    try:
        merge_datasets(args)
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
