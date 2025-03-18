#!/usr/bin/env python3
"""
Simple Annotation Visualization Tool

A tool that automatically detects and visualizes annotations in datasets.
Supports multiple annotation formats (YOLO, COCO, VOC).

Usage:
    python visualize_annotations.py [/path/to/dataset]
"""

import os
import sys
import argparse
import logging
import glob
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('visualize')

class AnnotationVisualizer:
    """
    Class for visualizing image annotations in various formats.
    """
    
    def __init__(self):
        """Initialize the visualization tool."""
        self.images = []
        self.annotations = []
        self.current_index = 0
        self.class_names = {}
        self.class_colors = {}
        
        # Supported image formats
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # Filtering options
        self.show_labels = True
        self.filter_classes = []
        self.confidence_threshold = 0.0
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        
        # Try to import visualization libraries
        try:
            import cv2
            self.cv2 = cv2
            self.has_cv2 = True
        except ImportError:
            logger.warning("OpenCV not found. Install with: pip install opencv-python")
            self.has_cv2 = False
            
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.has_plt = True
        except ImportError:
            logger.warning("Matplotlib not found. Install with: pip install matplotlib")
            self.has_plt = False
    
    def detect_format(self, dataset_dir):
        """
        Auto-detect the dataset format.
        
        Args:
            dataset_dir: Path to the dataset directory
            
        Returns:
            str: Detected format ('yolo', 'coco', 'voc') or None if unknown
        """
        print("Detecting dataset format...")
        
        # First, count all image and potential annotation files
        image_files = []
        txt_files = []
        xml_files = []
        json_files = []
        
        # Walk through the directory structure
        print("Scanning directory structure...")
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # Check if it's an image
                if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    image_files.append(file_path)
                # Check if it's a txt file (potential YOLO annotation)
                elif file_ext == '.txt':
                    txt_files.append(file_path)
                # Check if it's an xml file (potential VOC annotation)
                elif file_ext == '.xml':
                    xml_files.append(file_path)
                # Check if it's a json file (potential COCO annotation)
                elif file_ext == '.json':
                    json_files.append(file_path)
        
        print(f"Found {len(image_files)} images, {len(txt_files)} txt files, {len(xml_files)} XML files, {len(json_files)} JSON files")
        
        # Check for RDD2022 specific structure
        if self._check_rdd2022_format(dataset_dir, image_files, txt_files):
            print("✓ Detected RDD2022 dataset format")
            return 'rdd2022'
        
        # Check for standard YOLO structure
        if (os.path.exists(os.path.join(dataset_dir, 'images')) and 
            os.path.exists(os.path.join(dataset_dir, 'labels'))) or \
           (os.path.exists(os.path.join(dataset_dir, 'train/images')) and 
            os.path.exists(os.path.join(dataset_dir, 'train/labels'))) or \
           (os.path.exists(os.path.join(dataset_dir, 'val/images')) and 
            os.path.exists(os.path.join(dataset_dir, 'val/labels'))):
            print("✓ Detected YOLO format")
            return 'yolo'
        
        # Check for flat YOLO structure (images and txt in same folders)
        if len(image_files) > 0 and len(txt_files) > 0:
            # Check if txt files match image filenames
            matching_pairs = 0
            for img_path in image_files[:min(100, len(image_files))]:
                base_name = os.path.splitext(img_path)[0]
                if os.path.exists(f"{base_name}.txt"):
                    matching_pairs += 1
            
            if matching_pairs > 0:
                percentage = matching_pairs / min(100, len(image_files)) * 100
                print(f"✓ Detected flat YOLO format ({percentage:.1f}% of checked images have txt annotations)")
                return 'yolo_flat'
        
        # Check for COCO format
        if os.path.exists(os.path.join(dataset_dir, 'annotations')):
            coco_jsons = glob.glob(os.path.join(dataset_dir, 'annotations', '*.json'))
            if coco_jsons:
                # Verify it's COCO by checking one file
                try:
                    with open(coco_jsons[0], 'r') as f:
                        data = json.load(f)
                        if 'images' in data and 'annotations' in data and 'categories' in data:
                            print("✓ Detected COCO format")
                            return 'coco'
                except:
                    pass
        
        # Check for loose COCO files
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'images' in data and 'annotations' in data and 'categories' in data:
                        print(f"✓ Detected COCO format JSON file: {json_file}")
                        return 'coco_loose'
            except:
                continue
        
        # Check for VOC format
        if os.path.exists(os.path.join(dataset_dir, 'Annotations')) and \
           os.path.exists(os.path.join(dataset_dir, 'JPEGImages')):
            xml_files = glob.glob(os.path.join(dataset_dir, 'Annotations', '*.xml'))
            if xml_files:
                print("✓ Detected VOC format")
                return 'voc'
        
        # Check for loose VOC-style XML files
        if xml_files:
            # Check a few XML files to see if they look like VOC format
            voc_like = 0
            for xml_file in xml_files[:min(10, len(xml_files))]:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    if root.tag == 'annotation' and root.find('object') is not None:
                        voc_like += 1
                except:
                    continue
            
            if voc_like > 0:
                print(f"✓ Detected {voc_like} VOC-style XML files")
                return 'voc_loose'
        
        print("❌ Could not detect dataset format automatically")
        print("Trying common formats...")
        return None
    
    def _check_rdd2022_format(self, dataset_dir, image_files, txt_files):
        """Check if the dataset matches RDD2022 format"""
        # RDD2022 typically has country subfolders and images in a specific structure
        countries = ['China', 'Czech', 'India', 'Japan', 'Norway', 'United_States']
        country_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d in countries]
        
        if country_dirs:
            print(f"Found potential RDD2022 structure with {len(country_dirs)} country folders: {', '.join(country_dirs)}")
            
            # Check for annotations folder in RDD format
            annotations_present = False
            for country in country_dirs:
                annotations_dir = os.path.join(dataset_dir, country, 'annotations')
                if os.path.exists(annotations_dir):
                    annotations_present = True
                    print(f"Found annotations directory for {country}")
                    break
            
            # Also check for image/label pairs with matching names
            if annotations_present or len(txt_files) > 0:
                return True
        
        return False
    
    def load_dataset(self, dataset_dir, format_type=None, split=None):
        """
        Load a dataset with annotations.
        
        Args:
            dataset_dir: Path to the dataset directory
            format_type: Annotation format ('yolo', 'coco', 'voc') or None for auto-detect
            split: Dataset split to load ('train', 'val', 'test', None for all)
        """
        if format_type is None:
            format_type = self.detect_format(dataset_dir)
            
        # If still None, try all formats
        if format_type is None:
            formats_to_try = ['yolo', 'yolo_flat', 'coco', 'coco_loose', 'voc', 'voc_loose', 'rdd2022']
            for fmt in formats_to_try:
                try:
                    print(f"Trying {fmt} format...")
                    if fmt == 'yolo':
                        self._load_yolo_dataset(dataset_dir, split)
                    elif fmt == 'yolo_flat':
                        self._load_yolo_flat_dataset(dataset_dir)
                    elif fmt == 'coco':
                        self._load_coco_dataset(dataset_dir, split)
                    elif fmt == 'coco_loose':
                        self._load_coco_loose_dataset(dataset_dir)
                    elif fmt == 'voc':
                        self._load_voc_dataset(dataset_dir, split)
                    elif fmt == 'voc_loose':
                        self._load_voc_loose_dataset(dataset_dir)
                    elif fmt == 'rdd2022':
                        self._load_rdd2022_dataset(dataset_dir)
                        
                    if self.images:
                        print(f"✓ Successfully loaded data using {fmt} format!")
                        format_type = fmt
                        break
                except Exception as e:
                    print(f"Failed to load as {fmt}: {str(e)}")
        else:
            # Load with specified format
            logger.info(f"Loading {format_type} dataset from {dataset_dir}")
            if format_type == 'yolo':
                self._load_yolo_dataset(dataset_dir, split)
            elif format_type == 'yolo_flat':
                self._load_yolo_flat_dataset(dataset_dir)
            elif format_type == 'coco':
                self._load_coco_dataset(dataset_dir, split)
            elif format_type == 'coco_loose':
                self._load_coco_loose_dataset(dataset_dir)
            elif format_type == 'voc':
                self._load_voc_dataset(dataset_dir, split)
            elif format_type == 'voc_loose':
                self._load_voc_loose_dataset(dataset_dir)
            elif format_type == 'rdd2022':
                self._load_rdd2022_dataset(dataset_dir)
            else:
                raise ValueError(f"Unsupported annotation format: {format_type}")
            
        # Log summary
        logger.info(f"Loaded {len(self.images)} images with annotations")
        
        # Initialize class colors
        if self.images:
            self._initialize_colors()

    def _load_yolo_dataset(self, dataset_dir, split=None):
        """Load a YOLO format dataset."""
        # Reset lists
        self.images = []
        self.annotations = []
        
        # Check for YOLO-style dataset structure
        if os.path.exists(os.path.join(dataset_dir, 'images')) and os.path.exists(os.path.join(dataset_dir, 'labels')):
            # Standard YOLO structure
            splits = ['train', 'val', 'test'] if split is None else [split]
            valid_splits = []
            
            for s in splits:
                img_dir = os.path.join(dataset_dir, s, 'images')
                label_dir = os.path.join(dataset_dir, s, 'labels')
                
                if os.path.exists(img_dir) and os.path.exists(label_dir):
                    valid_splits.append((s, img_dir, label_dir))
                
            if not valid_splits:
                img_dir = os.path.join(dataset_dir, 'images')
                label_dir = os.path.join(dataset_dir, 'labels')
                
                if os.path.exists(img_dir) and os.path.exists(label_dir):
                    valid_splits.append(('all', img_dir, label_dir))
                    
            if not valid_splits:
                raise ValueError(f"Could not find valid YOLO dataset structure in {dataset_dir}")
                
            for split_name, img_dir, label_dir in valid_splits:
                logger.info(f"Loading {split_name} split")
                self._load_yolo_split(img_dir, label_dir, split_name)
                
        else:
            # Simple flat structure (images and labels in same directory)
            image_files = []
            for ext in self.image_extensions:
                image_files.extend(glob.glob(os.path.join(dataset_dir, f"*{ext}")))
                image_files.extend(glob.glob(os.path.join(dataset_dir, f"*{ext.upper()}")))
                
            for img_path in image_files:
                base_name = os.path.splitext(img_path)[0]
                label_path = f"{base_name}.txt"
                
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    self.annotations.append((label_path, 'yolo', 'all'))
                    
        # Try to load classes.txt if it exists
        classes_file = os.path.join(dataset_dir, 'classes.txt')
        if os.path.exists(classes_file):
            self._load_classes(classes_file)
            
    def _load_yolo_split(self, images_dir, labels_dir, split_name):
        """Load a single split of YOLO format dataset."""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
            
        for img_path in image_files:
            img_filename = os.path.basename(img_path)
            base_name = os.path.splitext(img_filename)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            if os.path.exists(label_path):
                self.images.append(img_path)
                self.annotations.append((label_path, 'yolo', split_name))
    
    def _load_yolo_flat_dataset(self, dataset_dir):
        """Load a YOLO format dataset with flat structure (txt files next to images)."""
        # Reset lists
        self.images = []
        self.annotations = []
        
        # Find all images
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(glob.glob(os.path.join(dataset_dir, f"**/*{ext}"), recursive=True))
            
        print(f"Found {len(image_files)} images in dataset directory")
        
        # Check each image for corresponding annotation
        for img_path in image_files:
            base_name = os.path.splitext(img_path)[0]
            label_path = f"{base_name}.txt"
            
            if os.path.exists(label_path):
                self.images.append(img_path)
                self.annotations.append((label_path, 'yolo', 'all'))
                
        print(f"Found {len(self.images)} images with matching annotations")
                
        # Try to load classes.txt if it exists (check in multiple locations)
        classes_files = [
            os.path.join(dataset_dir, 'classes.txt'),
            os.path.join(dataset_dir, 'data', 'classes.txt'),
            os.path.join(dataset_dir, 'labels', 'classes.txt'),
            os.path.join(os.path.dirname(dataset_dir), 'classes.txt')
        ]
        
        for classes_file in classes_files:
            if os.path.exists(classes_file):
                self._load_classes(classes_file)
                break

    def _load_coco_dataset(self, dataset_dir, split=None):
        """Load a COCO format dataset."""
        # Implementation for COCO format
        if split is None:
            json_files = glob.glob(os.path.join(dataset_dir, 'annotations', '*.json'))
        else:
            json_files = [os.path.join(dataset_dir, 'annotations', f'instances_{split}.json')]
            
        if not json_files:
            raise ValueError(f"No COCO annotation files found in {dataset_dir}/annotations/")
            
        for json_file in json_files:
            with open(json_file, 'r') as f:
                coco_data = json.load(f)
                
            # Extract class names
            for category in coco_data['categories']:
                self.class_names[category['id']] = category['name']
                
            # Map image IDs to filenames
            image_map = {}
            for img in coco_data['images']:
                image_map[img['id']] = {
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height']
                }
                
            # Group annotations by image
            image_to_annots = defaultdict(list)
            for ann in coco_data['annotations']:
                image_to_annots[ann['image_id']].append(ann)
                
            # Process each image
            split_name = os.path.splitext(os.path.basename(json_file))[0]
            for img_id, annots in image_to_annots.items():
                img_info = image_map[img_id]
                img_path = os.path.join(dataset_dir, 'images', img_info['file_name'])
                
                if os.path.exists(img_path):
                    self.images.append(img_path)
                    self.annotations.append((annots, 'coco', split_name))

    def _load_coco_loose_dataset(self, dataset_dir):
        """Load COCO format dataset with loose structure (look for JSON files)."""
        # Reset lists
        self.images = []
        self.annotations = []
        
        # Find all JSON files that might be COCO format
        json_files = glob.glob(os.path.join(dataset_dir, "**/*.json"), recursive=True)
        
        coco_jsons = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'images' in data and 'annotations' in data and 'categories' in data:
                        coco_jsons.append((json_file, data))
            except:
                continue
                
        if not coco_jsons:
            raise ValueError("No valid COCO annotation files found")
            
        print(f"Found {len(coco_jsons)} COCO annotation files")
        
        # Process each COCO file
        for json_file, coco_data in coco_jsons:
            # Extract class names
            for category in coco_data['categories']:
                self.class_names[category['id']] = category['name']
                
            # Map image IDs to filenames
            image_map = {}
            for img in coco_data['images']:
                image_map[img['id']] = {
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height']
                }
                
            # Group annotations by image
            image_to_annots = defaultdict(list)
            for ann in coco_data['annotations']:
                image_to_annots[ann['image_id']].append(ann)
                
            # Try to find images (either in same directory as JSON, or relative to JSON file)
            json_dir = os.path.dirname(json_file)
            possible_img_dirs = [
                json_dir,
                os.path.join(json_dir, 'images'),
                os.path.join(dataset_dir, 'images'),
                dataset_dir
            ]
            
            # Process each image
            split_name = os.path.splitext(os.path.basename(json_file))[0]
            for img_id, annots in image_to_annots.items():
                if not annots:
                    continue
                    
                img_info = image_map[img_id]
                img_filename = img_info['file_name']
                
                # Try to find the image in various locations
                img_path = None
                for img_dir in possible_img_dirs:
                    candidate_path = os.path.join(img_dir, img_filename)
                    if os.path.exists(candidate_path):
                        img_path = candidate_path
                        break
                
                if img_path and os.path.exists(img_path):
                    self.images.append(img_path)
                    self.annotations.append((annots, 'coco', split_name))
    
    def _load_voc_dataset(self, dataset_dir, split=None):
        """Load a VOC format dataset."""
        # Implementation for VOC format
        if split is None:
            xml_files = glob.glob(os.path.join(dataset_dir, 'Annotations', '*.xml'))
        else:
            with open(os.path.join(dataset_dir, 'ImageSets', 'Main', f'{split}.txt'), 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
            xml_files = [os.path.join(dataset_dir, 'Annotations', f'{img_id}.xml') for img_id in image_ids]
            
        if not xml_files:
            raise ValueError(f"No VOC annotation files found in {dataset_dir}/Annotations/")
            
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get filename
            filename = root.find('filename').text
            img_path = os.path.join(dataset_dir, 'JPEGImages', filename)
            
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.annotations.append((xml_file, 'voc', 'voc'))
                
                # Extract class names
                for obj in root.findall('./object'):
                    class_name = obj.find('name').text
                    if class_name not in self.class_names.values():
                        next_id = len(self.class_names)
                        self.class_names[next_id] = class_name

    def _load_voc_loose_dataset(self, dataset_dir):
        """Load VOC format dataset with loose structure (find XML files)."""
        # Reset lists
        self.images = []
        self.annotations = []
        
        # Find all XML files
        xml_files = glob.glob(os.path.join(dataset_dir, "**/*.xml"), recursive=True)
        
        # Filter for VOC-like XMLs
        voc_xmls = []
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                if root.tag == 'annotation' and root.find('object') is not None:
                    voc_xmls.append(xml_file)
            except:
                continue
                
        if not voc_xmls:
            raise ValueError("No valid VOC annotation files found")
            
        print(f"Found {len(voc_xmls)} VOC-style XML files")
        
        # For each XML, look for corresponding image
        for xml_file in voc_xmls:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get filename from XML
                filename_elem = root.find('filename')
                if filename_elem is None:
                    continue
                    
                filename = filename_elem.text
                
                # Try different locations for the image
                xml_dir = os.path.dirname(xml_file)
                possible_img_dirs = [
                    xml_dir,
                    os.path.join(os.path.dirname(xml_dir), 'JPEGImages'),
                    os.path.join(dataset_dir, 'JPEGImages'),
                    dataset_dir
                ]
                
                # Check if file exists in any of the possible directories
                img_path = None
                for img_dir in possible_img_dirs:
                    candidate_path = os.path.join(img_dir, filename)
                    if os.path.exists(candidate_path):
                        img_path = candidate_path
                        break
                
                if img_path and os.path.exists(img_path):
                    self.images.append(img_path)
                    self.annotations.append((xml_file, 'voc', 'all'))
                    
                    # Extract class names
                    for obj in root.findall('./object'):
                        class_name = obj.find('name').text
                        if class_name not in self.class_names.values():
                            next_id = len(self.class_names)
                            self.class_names[next_id] = class_name
            except Exception as e:
                print(f"Error processing XML file {xml_file}: {str(e)}")

    def _load_rdd2022_dataset(self, dataset_dir):
        """Load RDD2022 format dataset."""
        # Reset lists
        self.images = []
        self.annotations = []
        
        # RDD2022 class names
        rdd_classes = {
            0: 'D00', 1: 'D01', 2: 'D10', 3: 'D11', 4: 'D20',
            5: 'D40', 6: 'D43', 7: 'D44', 8: 'D50', 9: 'D0w0'
        }
        self.class_names = rdd_classes
        
        # Check for country folders
        countries = ['China', 'Czech', 'India', 'Japan', 'Norway', 'United_States']
        country_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d in countries]
        
        if not country_dirs:
            # Try looking for images and annotations directly
            self._load_yolo_flat_dataset(dataset_dir)
            return
            
        print(f"Found {len(country_dirs)} country folders in RDD2022 dataset")
        
        # RDD2022 can have different annotation formats - try to handle them
        for country in country_dirs:
            country_dir = os.path.join(dataset_dir, country)
            
            # Method 1: Look for annotations folder with json files
            annotations_dir = os.path.join(country_dir, 'annotations')
            if os.path.exists(annotations_dir):
                json_files = glob.glob(os.path.join(annotations_dir, '*.json'))
                if json_files:
                    # Try COCO-style loading
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                                
                            # Check if it's a valid annotation file
                            if isinstance(data, list):
                                # Handle list of annotations format
                                images_dir = os.path.join(country_dir, 'images')
                                if not os.path.exists(images_dir):
                                    # Try using the same folder as annotations
                                    images_dir = annotations_dir
                                
                                # Process each annotation
                                for annotation in data:
                                    if 'name' in annotation:
                                        # Look for image path
                                        img_name = annotation['name']
                                        for ext in ['.jpg', '.jpeg', '.png']:
                                            img_path = os.path.join(images_dir, img_name + ext)
                                            if os.path.exists(img_path):
                                                self.images.append(img_path)
                                                self.annotations.append((annotation, 'rdd2022_json', country))
                                                break
                            
                            # If COCO style with image/annotations keys
                            elif 'images' in data and 'annotations' in data:
                                # Try to find images
                                images_dir = os.path.join(country_dir, 'images')
                                if not os.path.exists(images_dir):
                                    # Try using the parent directory
                                    images_dir = country_dir
                                
                                # Map image IDs to filenames
                                image_map = {}
                                for img in data['images']:
                                    image_map[img['id']] = {
                                        'file_name': img['file_name'],
                                        'width': img.get('width', 0),
                                        'height': img.get('height', 0)
                                    }
                                
                                # Group annotations by image
                                image_to_annots = defaultdict(list)
                                for ann in data['annotations']:
                                    image_to_annots[ann['image_id']].append(ann)
                                
                                # Process each image
                                for img_id, annots in image_to_annots.items():
                                    if img_id in image_map:
                                        img_info = image_map[img_id]
                                        img_path = os.path.join(images_dir, img_info['file_name'])
                                        
                                        if os.path.exists(img_path):
                                            self.images.append(img_path)
                                            self.annotations.append((annots, 'coco', country))
                                            
                        except Exception as e:
                            print(f"Error loading JSON file {json_file}: {str(e)}")
                    
            # Method 2: Look for YOLO-style .txt annotations alongside images
            image_files = []
            for ext in self.image_extensions:
                image_files.extend(glob.glob(os.path.join(country_dir, f"**/*{ext}"), recursive=True))
            
            for img_path in image_files:
                base_name = os.path.splitext(img_path)[0]
                label_path = f"{base_name}.txt"
                
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    self.annotations.append((label_path, 'yolo', country))
        
        print(f"Found {len(self.images)} images with annotations in RDD2022 dataset")
    
    def _load_classes(self, classes_file):
        """Load class names from a text file."""
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            
            for i, name in enumerate(classes):
                self.class_names[i] = name
                
            logger.info(f"Loaded {len(classes)} classes from {classes_file}")
        except Exception as e:
            logger.warning(f"Could not load classes from {classes_file}: {str(e)}")
    
    def _initialize_colors(self):
        """Initialize colors for each class."""
        import random
        random.seed(42)  # For consistent colors
        
        for class_id in self.class_names:
            r = random.randint(100, 255)
            g = random.randint(100, 255)
            b = random.randint(100, 255)
            self.class_colors[class_id] = (b, g, r)  # BGR for OpenCV
    
    def start_visualization(self):
        """Start the visualization interface."""
        if not self.has_cv2:
            print("OpenCV is required for visualization. Please install with: pip install opencv-python")
            return
        
        if not self.images:
            print("No images with annotations found.")
            return
        
        print("\n=== Annotation Visualization ===")
        print(f"Loaded {len(self.images)} images with annotations")
        print("Press any key to begin viewing...")
        input()
        
        self.current_index = 0
        self._show_current_image()
        
        while True:
            key = self.cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == 83 or key == ord('n'):  # right arrow or 'n'
                self.current_index = (self.current_index + 1) % len(self.images)
                self._show_current_image()
            elif key == 81 or key == ord('p'):  # left arrow or 'p'
                self.current_index = (self.current_index - 1) % len(self.images)
                self._show_current_image()
            elif key == ord('l'):  # Toggle labels
                self.show_labels = not self.show_labels
                self._show_current_image()
            elif key == ord('+') or key == ord('='):  # Zoom in
                self.zoom_factor = min(5.0, self.zoom_factor * 1.2)
                self._show_current_image()
            elif key == ord('-'):  # Zoom out
                self.zoom_factor = max(0.2, self.zoom_factor / 1.2)
                self._show_current_image()
            elif key == ord('0'):  # Reset zoom and pan
                self.zoom_factor = 1.0
                self.pan_offset = (0, 0)
                self._show_current_image()
            elif key == ord('w'):  # Pan up
                self.pan_offset = (self.pan_offset[0], self.pan_offset[1] - 20)
                self._show_current_image()
            elif key == ord('s'):  # Pan down
                self.pan_offset = (self.pan_offset[0], self.pan_offset[1] + 20)
                self._show_current_image()
            elif key == ord('a'):  # Pan left
                self.pan_offset = (self.pan_offset[0] - 20, self.pan_offset[1])
                self._show_current_image()
            elif key == ord('d'):  # Pan right
                self.pan_offset = (self.pan_offset[0] + 20, self.pan_offset[1])
                self._show_current_image()
        
        self.cv2.destroyAllWindows()
    
    def _show_current_image(self):
        """Show the current image with its annotations."""
        img_path = self.images[self.current_index]
        annot_data, format_type, split = self.annotations[self.current_index]
        
        # Load image
        img = self.cv2.imread(img_path)
        if img is None:
            logger.error(f"Could not read image: {img_path}")
            return
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Create a copy of the image for display
        display_img = img.copy()
        
        # Draw annotations based on format if showing labels
        if self.show_labels:
            if format_type == 'yolo':
                display_img = self._draw_yolo_annotations(display_img, annot_data, width, height)
            elif format_type == 'coco':
                display_img = self._draw_coco_annotations(display_img, annot_data)
            elif format_type == 'voc':
                display_img = self._draw_voc_annotations(display_img, annot_data)
        
        # Apply zoom and pan
        if self.zoom_factor != 1.0 or self.pan_offset != (0, 0):
            # Calculate new dimensions
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            
            # Apply resize for zoom
            zoomed_img = self.cv2.resize(display_img, (new_width, new_height))
            
            # Create a black canvas of original size
            display_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate offsets for centering the zoomed image
            x_offset = max(0, int(width/2 - new_width/2) + self.pan_offset[0])
            y_offset = max(0, int(height/2 - new_height/2) + self.pan_offset[1])
            
            # Paste the visible portion of the zoomed image
            display_img[
                y_offset:min(y_offset + new_height, height),
                x_offset:min(x_offset + new_width, width)
            ] = zoomed_img[
                max(0, -y_offset):min(new_height, height - y_offset),
                max(0, -x_offset):min(new_width, width - x_offset)
            ]
        
        # Create window title with navigation info
        title = f"Image {self.current_index + 1}/{len(self.images)} - {os.path.basename(img_path)}"
        
        # Draw help text overlay
        self._draw_help_overlay(display_img)
        
        # Show image
        self.cv2.imshow("Annotation Viewer", display_img)
        self.cv2.setWindowTitle("Annotation Viewer", title)
        
    def _draw_help_overlay(self, img):
        """Draw help text on the image."""
        h, w = img.shape[:2]
        overlay_opacity = 0.7
        
        # Draw semi-transparent background
        help_bg = img.copy()
        self.cv2.rectangle(help_bg, (10, h-120), (250, h-10), (0, 0, 0), -1)
        self.cv2.addWeighted(help_bg, overlay_opacity, img, 1 - overlay_opacity, 0, img)
        
        # Draw help text
        help_text = [
            "Controls:",
            "  Next/Prev: Arrow keys",
            "  Toggle labels: L",
            "  Zoom: +/-",
            "  Pan: W,A,S,D",
            "  Quit: Q or ESC"
        ]
        
        y = h - 100
        for line in help_text:
            self.cv2.putText(img, line, (20, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 18
    
    def _draw_yolo_annotations(self, img, label_path, width, height):
        """Draw YOLO format annotations on an image."""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    
                    # Skip if class is filtered
                    if self.filter_classes and class_id not in self.filter_classes:
                        continue
                        
                    # Handle confidence threshold
                    if len(parts) >= 6:
                        conf = float(parts[5])
                        if conf < self.confidence_threshold:
                            continue
                    
                    # Rest of the annotation drawing logic
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    # Calculate bounding box coordinates
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    # Get class color and name
                    color = self.class_colors.get(class_id, (0, 255, 0))
                    class_name = self.class_names.get(class_id, f"Class {class_id}")
                    
                    # Draw bounding box
                    self.cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    text = f"{class_name}"
                    if len(parts) >= 6:  # If confidence is included
                        conf = float(parts[5])
                        text += f" {conf:.2f}"
                        
                    text_size, _ = self.cv2.getTextSize(text, self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    self.cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                    
                    # Draw class label
                    self.cv2.putText(img, text, (x1, y1 - 2), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return img
                    
        except Exception as e:
            logger.error(f"Error drawing YOLO annotations: {str(e)}")
            return img
    
    def _draw_coco_annotations(self, img, annotations):
        """Draw COCO format annotations on an image."""
        try:
            for ann in annotations:
                bbox = ann.get('bbox')
                category_id = ann.get('category_id')
                
                if bbox and category_id is not None:
                    # Skip if class is filtered
                    if self.filter_classes and category_id not in self.filter_classes:
                        continue
                        
                    # Handle confidence threshold
                    if 'score' in ann:
                        score = ann['score']
                        if score < self.confidence_threshold:
                            continue
                    
                    # COCO format bbox is [x, y, width, height]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[0] + bbox[2])
                    y2 = int(bbox[1] + bbox[3])
                    
                    # Get class color and name
                    color = self.class_colors.get(category_id, (0, 255, 0))
                    class_name = self.class_names.get(category_id, f"Class {category_id}")
                    
                    # Draw bounding box
                    self.cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    text = class_name
                    if 'score' in ann:
                        text += f" {ann['score']:.2f}"
                        
                    text_size, _ = self.cv2.getTextSize(text, self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    self.cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                    
                    # Draw class label
                    self.cv2.putText(img, text, (x1, y1 - 2), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return img
                    
        except Exception as e:
            logger.error(f"Error drawing COCO annotations: {str(e)}")
            return img
    
    def _draw_voc_annotations(self, img, xml_path):
        """Draw VOC format annotations on an image."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # Process each object
            for obj in root.findall('./object'):
                class_name = obj.find('name').text
                
                # Find class ID (if not in class_names, add it)
                class_id = None
                for cid, cname in self.class_names.items():
                    if cname == class_name:
                        class_id = cid
                        break
                
                if class_id is None:
                    class_id = len(self.class_names)
                    self.class_names[class_id] = class_name
                    # Generate a new color for this class
                    import random
                    r = random.randint(100, 255)
                    g = random.randint(100, 255)
                    b = random.randint(100, 255)
                    self.class_colors[class_id] = (b, g, r)
                
                # Skip if class is filtered
                if self.filter_classes and class_id not in self.filter_classes:
                    continue
                
                # Get bounding box
                bbox = obj.find('bndbox')
                if bbox is not None:
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Get color
                    color = self.class_colors.get(class_id, (0, 255, 0))
                    
                    # Draw bounding box
                    self.cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    # Draw label
                    text = class_name
                    text_size, _ = self.cv2.getTextSize(text, self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    self.cv2.rectangle(img, (xmin, ymin - text_size[1] - 4), (xmin + text_size[0], ymin), color, -1)
                    self.cv2.putText(img, text, (xmin, ymin - 2), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return img
                
        except Exception as e:
            logger.error(f"Error drawing VOC annotations: {str(e)}")
            return img

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Simple Annotation Visualization Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('dataset_dir', type=str, nargs='?', default=None,
                        help='Path to dataset directory')
    
    return parser.parse_args()


def main():
    """Main function to run the visualization."""
    args = parse_args()
    
    try:
        dataset_dir = args.dataset_dir
        
        # If no dataset provided, ask for one
        if dataset_dir is None:
            print("\n=== Simple Annotation Visualization Tool ===")
            dataset_dir = input("Enter path to dataset directory: ")
            if not dataset_dir:
                print("No directory provided. Exiting.")
                return
        
        # Verify the directory exists
        if not os.path.isdir(dataset_dir):
            print(f"Error: Directory '{dataset_dir}' does not exist.")
            return
        
        # Initialize visualizer
        visualizer = AnnotationVisualizer()
        
        # Show loading progress
        print(f"\nScanning dataset at: {os.path.abspath(dataset_dir)}")
        start_time = time.time()
        
        # Load the dataset (auto-detect format)
        visualizer.load_dataset(dataset_dir)
        
        load_time = time.time() - start_time
        print(f"Dataset loaded in {load_time:.2f} seconds.")
        
        # Start visualization if we found annotations
        if visualizer.images:
            visualizer.start_visualization()
        else:
            print("\nNo images with annotations were found.")
            print("Available files in the directory:")
            for item in sorted(os.listdir(dataset_dir))[:20]:  # Show first 20 items
                item_path = os.path.join(dataset_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
            
            if len(os.listdir(dataset_dir)) > 20:
                print(f"  ... and {len(os.listdir(dataset_dir)) - 20} more items")
    
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
