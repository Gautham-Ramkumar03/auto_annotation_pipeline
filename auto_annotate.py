#!/usr/bin/env python3
"""
Auto Annotation Script for Road Damage Detection

This script processes a folder of images using a YOLOv8 model to automatically detect and annotate
road damages/cracks. It can output annotations in multiple formats (YOLO, COCO, Pascal VOC) and 
organize the dataset into train/val/test splits or a single folder.

Usage:
    python auto_annotate.py [options]

Options:
    --input-dir PATH         Directory containing images to process (default: img_data)
    --output-dir PATH        Directory to save annotated dataset (default: dataset)
    --model PATH             Path to YOLOv8 model file (default: yolov8n.pt)
    --format FORMAT          Annotation format: yolo, coco, voc (default: yolo)
    --conf-threshold FLOAT   Confidence threshold for detections (default: 0.25)
    --iou-threshold FLOAT    IoU threshold for NMS (default: 0.7)
    --batch-size INT         Batch size for processing (default: 8)
    --single-class           Treat all detections as a single class
    --no-split               Don't split dataset, put all images in train folder
    --split RATIO            Custom split ratios: "train,val,test" (default: "70,20,10")

Examples:
    # Basic usage with default settings
    python auto_annotate.py
    
    # Custom input/output directories with COCO format
    python auto_annotate.py --input-dir my_images --output-dir my_dataset --format coco
    
    # Custom model and thresholds with single class and no splitting
    python auto_annotate.py --model road_damage_model.pt --conf-threshold 0.4 --single-class --no-split
"""

import os
import sys
import glob
import shutil
import argparse
import logging
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional, Any

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: YOLOv8 not installed. Please install with: pip install ultralytics")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('auto_annotate')


class AutoAnnotator:
    """Main class for automatic image annotation using YOLOv8."""
    
    def __init__(self, args):
        """
        Initialize the Auto Annotator with command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.model_path = args.model
        self.format = args.format
        self.conf_threshold = args.conf_threshold
        self.iou_threshold = args.iou_threshold
        self.batch_size = args.batch_size
        self.single_class = args.single_class
        self.no_split = args.no_split
        
        # Parse split ratios
        if args.split:
            try:
                ratios = [float(x) for x in args.split.split(',')]
                if len(ratios) != 3 or sum(ratios) != 100 or any(r < 0 for r in ratios):
                    logger.error("Invalid split ratios. Must be three comma-separated values summing to 100.")
                    sys.exit(1)
                self.split_ratios = {'train': ratios[0], 'val': ratios[1], 'test': ratios[2]}
            except ValueError:
                logger.error("Invalid split format. Use comma-separated numbers (e.g. '70,20,10')")
                sys.exit(1)
        else:
            self.split_ratios = {'train': 70, 'val': 20, 'test': 10}
        
        # Load the model
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            sys.exit(1)
        
        # Get image file paths
        self.image_paths = self._get_image_paths()
        if not self.image_paths:
            logger.error(f"No image files found in {self.input_dir}")
            sys.exit(1)
        logger.info(f"Found {len(self.image_paths)} images to process")

    def _get_image_paths(self) -> List[str]:
        """Get paths of all image files in the input directory."""
        if not os.path.exists(self.input_dir):
            logger.error(f"Input directory does not exist: {self.input_dir}")
            sys.exit(1)
            
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.input_dir, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(self.input_dir, f"*{ext.upper()}")))
        
        return sorted(image_paths)

    def _create_output_dirs(self) -> Dict[str, str]:
        """Create output directory structure based on split settings."""
        if os.path.exists(self.output_dir):
            logger.warning(f"Output directory {self.output_dir} already exists. Files may be overwritten.")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.no_split:
            # Single folder structure
            split_dirs = {'train': 'train'}
            for folder in ['train']:
                img_dir = os.path.join(self.output_dir, folder, 'images')
                os.makedirs(img_dir, exist_ok=True)
                if self.format == 'yolo':
                    label_dir = os.path.join(self.output_dir, folder, 'labels')
                    os.makedirs(label_dir, exist_ok=True)
                elif self.format in ['coco', 'voc']:
                    label_dir = os.path.join(self.output_dir, folder, 'annotations')
                    os.makedirs(label_dir, exist_ok=True)
        else:
            # Split folder structure
            split_dirs = {'train': 'train', 'val': 'val', 'test': 'test'}
            for folder in ['train', 'val', 'test']:
                img_dir = os.path.join(self.output_dir, folder, 'images')
                os.makedirs(img_dir, exist_ok=True)
                if self.format == 'yolo':
                    label_dir = os.path.join(self.output_dir, folder, 'labels')
                    os.makedirs(label_dir, exist_ok=True)
                elif self.format in ['coco', 'voc']:
                    label_dir = os.path.join(self.output_dir, folder, 'annotations')
                    os.makedirs(label_dir, exist_ok=True)
        
        # Create a dataset.yaml file for YOLOv8 training
        self._create_dataset_yaml(split_dirs)
        
        return split_dirs

    def _create_dataset_yaml(self, split_dirs: Dict[str, str]) -> None:
        """Create dataset.yaml file for YOLOv8 training."""
        yaml_path = os.path.join(self.output_dir, 'dataset.yaml')
        
        # Get class names from the model
        class_names = self.model.names
        
        if self.single_class:
            class_names = {'0': 'damage'}  # Override with single class
        
        # Construct the yaml content
        yaml_content = [
            f"# Auto-generated dataset config for YOLOv8 training",
            f"# Created by auto_annotate.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"path: {os.path.abspath(self.output_dir)}",
            f"train: {os.path.join(split_dirs['train'], 'images')}",
        ]
        
        if not self.no_split:
            yaml_content.extend([
                f"val: {os.path.join(split_dirs['val'], 'images')}",
                f"test: {os.path.join(split_dirs['test'], 'images')}" 
            ])
        
        yaml_content.extend([
            f"",
            f"nc: {len(class_names)}",
            f"names: {list(class_names.values())}"
        ])
        
        # Write to file
        with open(yaml_path, 'w') as f:
            f.write('\n'.join(yaml_content))
        
        logger.info(f"Created dataset YAML file at: {yaml_path}")

    def process_images(self) -> None:
        """Process all images in batches and generate annotations."""
        # Create output directories
        split_dirs = self._create_output_dirs()
        
        # Determine which split each image goes into
        if self.no_split:
            # All images go to train
            image_destinations = {path: 'train' for path in self.image_paths}
        else:
            # Random split based on ratios
            np.random.seed(42)  # For reproducibility
            indices = np.random.permutation(len(self.image_paths))
            train_end = int(len(self.image_paths) * self.split_ratios['train'] / 100)
            val_end = train_end + int(len(self.image_paths) * self.split_ratios['val'] / 100)
            
            image_destinations = {}
            for i, path in enumerate(self.image_paths):
                if i < train_end:
                    image_destinations[path] = 'train'
                elif i < val_end:
                    image_destinations[path] = 'val'
                else:
                    image_destinations[path] = 'test'
        
        # Process images in batches
        logger.info(f"Processing images in batches of {self.batch_size}...")
        
        # Initialize COCO dataset if needed
        if self.format == 'coco':
            coco_datasets = self._initialize_coco_datasets(split_dirs)
        
        # Process in batches
        for i in range(0, len(self.image_paths), self.batch_size):
            batch_paths = self.image_paths[i:i+self.batch_size]
            
            # Run inference on batch
            try:
                results = self.model.predict(
                    batch_paths,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"Inference failed for batch {i//self.batch_size+1}: {str(e)}")
                continue
            
            # Process each result
            for img_path, result in zip(batch_paths, results):
                if result.boxes is None or len(result.boxes) == 0:
                    logger.debug(f"No detections for {os.path.basename(img_path)}")
                    # Still copy the image even if no detections
                    split = image_destinations[img_path]
                    self._copy_image_to_dataset(img_path, split)
                    continue
                
                # Get detections
                img_height, img_width = result.orig_shape
                boxes = result.boxes.xyxy.cpu().numpy()
                if result.boxes.conf is not None:
                    confidences = result.boxes.conf.cpu().numpy()
                else:
                    confidences = np.ones(len(boxes))
                    
                if result.boxes.cls is not None:
                    if self.single_class:
                        # Force all to class 0
                        classes = np.zeros(len(boxes), dtype=np.int64)
                    else:
                        classes = result.boxes.cls.cpu().numpy().astype(np.int64)
                else:
                    classes = np.zeros(len(boxes), dtype=np.int64)
                
                # Convert and save annotations based on format
                split = image_destinations[img_path]
                                
                if self.format == 'yolo':
                    self._save_yolo_annotation(img_path, split, boxes, classes, img_width, img_height)
                elif self.format == 'coco':
                    self._add_to_coco_dataset(coco_datasets[split], img_path, split, 
                                              boxes, classes, confidences, img_width, img_height)
                elif self.format == 'voc':
                    self._save_voc_annotation(img_path, split, boxes, classes, confidences, 
                                             img_width, img_height)
                
                # Copy the image to the dataset directory
                self._copy_image_to_dataset(img_path, split)
        
        # Save COCO datasets if needed
        if self.format == 'coco':
            for split, coco_data in coco_datasets.items():
                label_dir = os.path.join(self.output_dir, split, 'annotations')
                with open(os.path.join(label_dir, 'instances.json'), 'w') as f:
                    json.dump(coco_data, f, indent=2)
        
        logger.info("Processing complete!")

    def _copy_image_to_dataset(self, img_path: str, split: str) -> None:
        """Copy an image to the appropriate dataset directory."""
        dst_dir = os.path.join(self.output_dir, split, 'images')
        shutil.copy2(img_path, os.path.join(dst_dir, os.path.basename(img_path)))

    def _save_yolo_annotation(self, img_path: str, split: str, 
                             boxes: np.ndarray, classes: np.ndarray,
                             img_width: int, img_height: int) -> None:
        """Save annotations in YOLO format."""
        label_dir = os.path.join(self.output_dir, split, 'labels')
        label_path = os.path.join(label_dir, Path(img_path).stem + '.txt')
        
        with open(label_path, 'w') as f:
            for box, cls in zip(boxes, classes):
                # Convert to YOLO format: class x_center y_center width height
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Write to file (ensuring values are within 0-1 range)
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _initialize_coco_datasets(self, split_dirs: Dict[str, str]) -> Dict[str, Dict]:
        """Initialize COCO dataset structures for each split."""
        coco_datasets = {}
        
        # Get class names from the model
        if self.single_class:
            categories = [{"id": 0, "name": "damage", "supercategory": "damage"}]
        else:
            categories = [{"id": i, "name": name, "supercategory": name} 
                         for i, name in self.model.names.items()]
        
        for split in split_dirs.keys():
            coco_datasets[split] = {
                "info": {
                    "description": f"Road damage dataset - {split} set",
                    "url": "",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "auto_annotate.py",
                    "date_created": datetime.now().strftime("%Y/%m/%d")
                },
                "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
                "images": [],
                "annotations": [],
                "categories": categories
            }
        
        return coco_datasets

    def _add_to_coco_dataset(self, coco_data: Dict, img_path: str, split: str, 
                            boxes: np.ndarray, classes: np.ndarray, confidences: np.ndarray,
                            img_width: int, img_height: int) -> None:
        """Add image and annotations to COCO dataset."""
        # Create image entry
        image_id = len(coco_data["images"]) + 1
        filename = os.path.basename(img_path)
        
        coco_data["images"].append({
            "id": image_id,
            "width": img_width,
            "height": img_height,
            "file_name": filename,
            "license": 1,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Create annotation entries
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # COCO format uses [x, y, width, height] for bbox
            bbox = [float(x1), float(y1), float(width), float(height)]
            
            # Calculate area
            area = width * height
            
            annotation_id = len(coco_data["annotations"]) + 1
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
                "score": float(conf)
            })

    def _save_voc_annotation(self, img_path: str, split: str, 
                            boxes: np.ndarray, classes: np.ndarray, confidences: np.ndarray,
                            img_width: int, img_height: int) -> None:
        """Save annotations in Pascal VOC XML format."""
        label_dir = os.path.join(self.output_dir, split, 'annotations')
        label_path = os.path.join(label_dir, Path(img_path).stem + '.xml')
        
        # Get class names
        if self.single_class:
            class_names = {0: "damage"}
        else:
            class_names = self.model.names
        
        # Create XML structure
        root = ET.Element('annotation')
        
        # Add basic image information
        folder = ET.SubElement(root, 'folder')
        folder.text = split
        
        filename_elem = ET.SubElement(root, 'filename')
        filename_elem.text = os.path.basename(img_path)
        
        path_elem = ET.SubElement(root, 'path')
        path_elem.text = os.path.abspath(os.path.join(self.output_dir, split, 'images', os.path.basename(img_path)))
        
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        
        # Add size information
        size = ET.SubElement(root, 'size')
        width_elem = ET.SubElement(size, 'width')
        width_elem.text = str(img_width)
        height_elem = ET.SubElement(size, 'height')
        height_elem.text = str(img_height)
        depth_elem = ET.SubElement(size, 'depth')
        depth_elem.text = '3'  # Assuming RGB
        
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        
        # Add object information for each detection
        for box, cls, conf in zip(boxes, classes, confidences):
            obj = ET.SubElement(root, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = class_names[int(cls)]
            
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'
            
            confidence = ET.SubElement(obj, 'confidence')
            confidence.text = f"{conf:.4f}"
            
            bndbox = ET.SubElement(obj, 'bndbox')
            
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(max(0, int(box[0])))
            
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(max(0, int(box[1])))
            
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(min(img_width, int(box[2])))
            
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(min(img_height, int(box[3])))
        
        # Write to XML file
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(label_path, 'w') as f:
            f.write(xml_str)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Auto Annotation Tool for Road Damage Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, default='img_data',
                        help='Directory containing images to process')
    parser.add_argument('--output-dir', type=str, default='dataset',
                        help='Directory to save annotated dataset')
    parser.add_argument('--model', type=str, default='models/best.pt',
                        help='Path to YOLOv8 model file')
    parser.add_argument('--format', type=str, choices=['yolo', 'coco', 'voc'], default='yolo',
                        help='Annotation format to use')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.7,
                        help='IoU threshold for NMS')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--single-class', action='store_true',
                        help='Treat all detections as a single class')
    parser.add_argument('--no-split', action='store_true',
                        help="Don't split dataset, put all images in train folder")
    parser.add_argument('--split', type=str,
                        help='Custom split ratios: "train,val,test" (default: "70,20,10")')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    return args


def main():
    """Main function to run the annotation process."""
    args = parse_args()
    logger.info("Starting auto annotation process...")
    
    try:
        annotator = AutoAnnotator(args)
        annotator.process_images()
        logger.info(f"Annotation complete! Dataset saved to {args.output_dir}")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
