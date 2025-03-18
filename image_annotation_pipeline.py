#!/usr/bin/env python3
"""
Image Annotation Pipeline for Road Damage Detection

This script creates a pipeline for annotating image datasets:
1. Processes a folder containing images (including nested directories)
2. Checks for existing annotations and uses them if available
3. For images without annotations, applies a YOLOv8 model for auto-annotation
4. Organizes the annotated images into a proper dataset structure

Usage:
    python image_annotation_pipeline.py

The script provides an interactive CLI for configuring the annotation process.
"""

import os
import sys
import logging
import argparse
import shutil
import concurrent.futures
import time
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('image_pipeline')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ImageAnnotationPipeline:
    """
    Main class for the image annotation pipeline, focusing on processing
    existing image datasets with intelligent annotation.
    """
    
    def __init__(self, args=None):
        """
        Initialize the annotation pipeline with command-line arguments or defaults.
        
        Args:
            args: Optional parsed command-line arguments
        """
        self.args = args or self._get_default_args()
        
        # Initialize configuration dict
        self.config = {
            'input': {},
            'annotation': {},
            'output': {}
        }
        
        # Set up temp and output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(os.getcwd(), "dataset")
        
        # Define supported image formats
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # Check if YOLOv8 is available
        try:
            from ultralytics import YOLO
            self.yolo_available = True
        except ImportError:
            self.yolo_available = False
            logger.warning("YOLOv8 not detected. Will attempt to install when needed.")

    def _get_default_args(self):
        """Create default arguments when none are provided."""
        class DefaultArgs:
            def __init__(self):
                self.debug = False
                self.config = None
        return DefaultArgs()
    
    def run_interactive_cli(self):
        """Run the interactive command-line interface to collect parameters."""
        print("\n=== Image Annotation Pipeline ===")
        print("This tool will process an image dataset and automatically annotate detected objects.")
        
        # === Input Configuration ===
        print("\n=== Input Configuration ===")
        
        # Input folder
        self.config['input']['folder'] = self._prompt_input_folder()
        
        # Recursive search
        self.config['input']['recursive'] = self._prompt_recursive()
        
        # === Annotation Configuration ===
        print("\n=== Annotation Configuration ===")
        
        # Model selection
        self.config['annotation']['model_path'] = self._prompt_model_path()
        
        # Annotation format
        self.config['annotation']['format'] = self._prompt_annotation_format()
        
        # Use existing annotations
        self.config['annotation']['use_existing'] = self._prompt_use_existing()
        
        # Manual review option
        self.config['annotation']['manual_review'] = self._prompt_manual_review()
        
        # === Output Configuration ===
        print("\n=== Output Configuration ===")
        
        # Output folder
        self.config['output']['folder'] = self._prompt_output_folder()
        
        # Dataset splitting
        split_option = self._prompt_split_option()
        if split_option == 'no_split':
            self.config['output']['no_split'] = True
            self.config['output']['split'] = None
        else:
            self.config['output']['no_split'] = False
            self.config['output']['split'] = self._prompt_split_ratios()
        
        # Parallel processing
        self.config['annotation']['parallel'] = self._prompt_parallel_processing()
            
        print("\n=== Configuration Summary ===")
        self._print_config_summary()
        
        return self.config
    
    def _prompt_input_folder(self):
        """Prompt user for folder path containing image files."""
        while True:
            folder_path = input("Enter the folder path containing image files [./images]: ").strip()
            if not folder_path:
                folder_path = "./images"
            if os.path.isdir(folder_path):
                return folder_path
            print(f"The path '{folder_path}' is not a valid directory. Please try again.")

    def _prompt_recursive(self):
        """Prompt user about recursive folder search."""
        choice = input("Search for images in subdirectories? (y/n) [y]: ").strip().lower()
        return not choice or choice.startswith('y')

    def _prompt_model_path(self):
        """Prompt user for model path."""
        default_model = 'yolov8n.pt'
        model_path = input(f"Enter YOLOv8 model path [models/best.pt]: ").strip()
        
        if not model_path:
            model_path = 'models/best.pt'
        
        # Check if model exists, if not offer to download default
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            if not self.yolo_available:
                choice = input("Would you like to install YOLOv8 and download a default model? (y/n) [y]: ").strip().lower()
                if not choice or choice.startswith('y'):
                    self._install_yolov8()
            
            if self.yolo_available:
                choice = input(f"Would you like to download {default_model}? (y/n) [y]: ").strip().lower()
                if not choice or choice.startswith('y'):
                    try:
                        from ultralytics import YOLO
                        logger.info(f"Downloading {default_model}...")
                        model = YOLO(default_model)
                        return default_model
                    except Exception as e:
                        logger.error(f"Error downloading model: {e}")
        
        return model_path
    
    def _install_yolov8(self):
        """Install YOLOv8 using pip."""
        logger.info("Installing YOLOv8...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            from ultralytics import YOLO
            self.yolo_available = True
            logger.info("YOLOv8 installed successfully!")
        except Exception as e:
            logger.error(f"Failed to install YOLOv8: {e}")
            logger.info("Please install manually using: pip install ultralytics")
    
    def _prompt_annotation_format(self):
        """Prompt user for annotation format."""
        print("\nSelect annotation format:")
        formats = ['yolo', 'coco', 'voc']
        for i, fmt in enumerate(formats, 1):
            print(f"{i}. {fmt.upper()}")
            
        while True:
            try:
                choice = input(f"Enter your choice (1-{len(formats)}) [1]: ").strip()
                if not choice:
                    return 'yolo'  # Default
                choice = int(choice)
                if 1 <= choice <= len(formats):
                    return formats[choice-1]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def _prompt_use_existing(self):
        """Prompt user about using existing annotations."""
        choice = input("Use existing annotations if available? (y/n) [y]: ").strip().lower()
        return not choice or choice.startswith('y')

    def _prompt_manual_review(self):
        """Prompt user about manual review of annotations."""
        choice = input("Enable manual review of auto-annotations? (y/n) [n]: ").strip().lower()
        return choice and choice.startswith('y')

    def _prompt_output_folder(self):
        """Prompt user for output folder path."""
        folder_path = input("Enter the output folder path [./dataset]: ").strip()
        if not folder_path:
            folder_path = "./dataset"
        return folder_path

    def _prompt_split_option(self):
        """Prompt user for dataset splitting option."""
        print("\nSelect dataset splitting method:")
        print("1. Split dataset into train/val/test")
        print("2. Use all images for training (no split)")
        
        while True:
            try:
                choice = input("Enter your choice (1-2) [1]: ").strip()
                if not choice:
                    return 'split'  # Default
                choice = int(choice)
                if choice == 1:
                    return 'split'
                elif choice == 2:
                    return 'no_split'
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def _prompt_split_ratios(self):
        """Prompt user for custom split ratios."""
        default_split = "70,20,10"
        while True:
            split_input = input(f"Enter split ratios as train,val,test (must sum to 100) [{default_split}]: ").strip()
            if not split_input:
                return default_split
                
            try:
                ratios = [float(x) for x in split_input.split(',')]
                if len(ratios) != 3:
                    print("Please provide exactly three values separated by commas.")
                    continue
                    
                if sum(ratios) != 100 or any(r < 0 for r in ratios):
                    print("Values must sum to 100 and be non-negative.")
                    continue
                    
                return split_input
            except ValueError:
                print("Invalid format. Please use comma-separated numbers (e.g., '70,20,10').")

    def _prompt_parallel_processing(self):
        """Prompt user about parallel processing."""
        print("\nParallel processing options:")
        print("0. Disabled (process one image at a time)")
        print("1. Low (use 25% of available CPU cores)")
        print("2. Medium (use 50% of available CPU cores)")
        print("3. High (use 75% of available CPU cores)")
        print("4. Maximum (use all available CPU cores)")
        
        while True:
            choice = input("Select parallel processing level (0-4) [2]: ").strip()
            if not choice:
                return 2  # Default - Medium
            try:
                level = int(choice)
                if 0 <= level <= 4:
                    return level
                print("Invalid level. Please enter a number between 0 and 4.")
            except ValueError:
                print("Please enter a valid number.")
    
    def _print_config_summary(self):
        """Print summary of the selected configuration."""
        print("\nInput Configuration:")
        print(f"- Input folder: {self.config['input']['folder']}")
        print(f"- Recursive search: {'Yes' if self.config['input']['recursive'] else 'No'}")
        
        print("\nAnnotation Configuration:")
        print(f"- Model: {self.config['annotation']['model_path']}")
        print(f"- Format: {self.config['annotation']['format']}")
        print(f"- Use existing annotations: {'Yes' if self.config['annotation']['use_existing'] else 'No'}")
        print(f"- Manual review: {'Yes' if self.config['annotation']['manual_review'] else 'No'}")
        print(f"- Parallel processing: Level {self.config['annotation']['parallel']}")
        
        print("\nOutput Configuration:")
        print(f"- Output folder: {self.config['output']['folder']}")
        
        if self.config['output']['no_split']:
            print("- Dataset split: No split (all images for training)")
        else:
            print(f"- Dataset split: {self.config['output']['split']}")
    
    def find_images(self) -> List[str]:
        """Find all image files in the input directory."""
        input_dir = self.config['input']['folder']
        recursive = self.config['input']['recursive']
        
        logger.info(f"Searching for images in {input_dir} (recursive: {recursive})")
        
        image_files = []
        
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file.lower().endswith(ext) for ext in self.supported_image_formats):
                        image_files.append(file_path)
        else:
            for file in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in self.supported_image_formats):
                    image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} image files")
        return image_files

    def find_existing_annotation(self, image_path: str) -> Optional[str]:
        """
        Check if an annotation file exists for the given image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to annotation file if it exists, None otherwise
        """
        # For YOLO format, check for a .txt file with the same name
        if self.config['annotation']['format'] == 'yolo':
            base_path = os.path.splitext(image_path)[0]
            txt_path = f"{base_path}.txt"
            if os.path.exists(txt_path):
                return txt_path
        
        # For COCO format, implementation would be more complex
        # as we would need to check a JSON file for the image
        
        # For VOC format, check for matching XML file
        elif self.config['annotation']['format'] == 'voc':
            base_path = os.path.splitext(image_path)[0]
            xml_path = f"{base_path}.xml"
            if os.path.exists(xml_path):
                return xml_path
                
        return None

    def process_image(self, image_path: str) -> Tuple[str, bool]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing (image_path, is_existing_annotation)
        """
        # Check for existing annotation
        existing_annotation = None
        if self.config['annotation']['use_existing']:
            existing_annotation = self.find_existing_annotation(image_path)
        
        if existing_annotation:
            # Copy both the image and its annotation to the output directory
            return (image_path, True)
        else:
            # Image needs to be annotated
            return (image_path, False)

    def annotate_image(self, image_path: str) -> bool:
        """
        Annotate an image using the YOLOv8 model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if annotation was successful, False otherwise
        """
        try:
            from ultralytics import YOLO
            model = YOLO(self.config['annotation']['model_path'])
            
            # Run prediction on the image
            results = model.predict(image_path, save=False, save_txt=True)
            
            # Check if any objects were detected
            if len(results) > 0 and len(results[0].boxes) > 0:
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error annotating {image_path}: {str(e)}")
            return False

    def create_dataset_structure(self, image_paths: List[str], has_existing_annotation: List[bool]):
        """
        Create the dataset directory structure and copy/move files.
        
        Args:
            image_paths: List of paths to image files
            has_existing_annotation: List of booleans indicating if each image has an annotation
        """
        output_dir = self.config['output']['folder']
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine if dataset should be split
        if self.config['output']['no_split']:
            # No split - all images go to train folder
            train_dir = os.path.join(output_dir, 'train')
            os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
            
            # Copy all images and labels to train folder
            for i, (image_path, has_annotation) in enumerate(zip(image_paths, has_existing_annotation)):
                self._copy_to_dataset(image_path, has_annotation, 'train')
                
        else:
            # Split dataset into train/val/test
            split_ratios = [float(x)/100 for x in self.config['output']['split'].split(',')]
            
            # Create the necessary directories
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
            
            # Shuffle and split the dataset
            import random
            combined = list(zip(image_paths, has_existing_annotation))
            random.shuffle(combined)
            
            # Calculate split indices
            total = len(combined)
            train_end = int(total * split_ratios[0])
            val_end = train_end + int(total * split_ratios[1])
            
            # Copy files to each split
            for i, (image_path, has_annotation) in enumerate(combined):
                if i < train_end:
                    split = 'train'
                elif i < val_end:
                    split = 'val'
                else:
                    split = 'test'
                    
                self._copy_to_dataset(image_path, has_annotation, split)

    def _copy_to_dataset(self, image_path: str, has_annotation: bool, split: str):
        """
        Copy an image and its annotation to the appropriate dataset folder.
        
        Args:
            image_path: Path to the image file
            has_annotation: Whether the image has an existing annotation
            split: Dataset split ('train', 'val', or 'test')
        """
        output_dir = self.config['output']['folder']
        
        # Define destination paths
        image_filename = os.path.basename(image_path)
        image_dest = os.path.join(output_dir, split, 'images', image_filename)
        
        # Copy image
        shutil.copy(image_path, image_dest)
        
        # Handle annotation
        if has_annotation:
            # Copy existing annotation
            annotation_path = self.find_existing_annotation(image_path)
            if annotation_path:
                annotation_filename = os.path.basename(annotation_path)
                annotation_dest = os.path.join(output_dir, split, 'labels', annotation_filename)
                shutil.copy(annotation_path, annotation_dest)
        else:
            # Generate new annotation using direct path specification
            if self.config['annotation']['format'] == 'yolo':
                # Define base filename without extension
                base_name = os.path.splitext(image_filename)[0]
                
                # Check for labels in the runs/detect/predict/labels directory (created by YOLOv8 predict)
                yolo_label_path = os.path.join('runs', 'detect', 'predict', 'labels', f"{base_name}.txt")
                
                # If label exists, copy it to the correct destination
                if os.path.exists(yolo_label_path):
                    annotation_dest = os.path.join(output_dir, split, 'labels', f"{base_name}.txt")
                    shutil.copy(yolo_label_path, annotation_dest)
                    logger.debug(f"Copied annotation from {yolo_label_path} to {annotation_dest}")
                else:
                    logger.warning(f"No annotation found for {image_path}")

    def manual_review_annotations(self, image_paths: List[str], has_existing_annotation: List[bool]):
        """
        Allow user to manually review auto-generated annotations.
        
        Args:
            image_paths: List of paths to image files
            has_existing_annotation: List of booleans indicating if each image has an annotation
        """
        if not self.config['annotation']['manual_review']:
            return
            
        try:
            import cv2
            from ultralytics import YOLO
            
            model = YOLO(self.config['annotation']['model_path'])
            
            # Filter images that need annotation
            review_images = [img for img, has_annot in zip(image_paths, has_existing_annotation) if not has_annot]
            
            total_images = len(review_images)
            logger.info(f"Manual review mode: {total_images} images to review")
            print(f"\nStarting manual review of {total_images} annotations.")
            print("Press 'a' to accept, 'r' to reject, 'q' to quit review")
            print("(If the review window becomes unresponsive, click on it and press ESC to skip to the next image)")
            
            try:
                for idx, img_path in enumerate(review_images):
                    print(f"\nReviewing image {idx+1}/{total_images}: {os.path.basename(img_path)}")
                    
                    # Run detection
                    results = model.predict(img_path)
                    
                    # Show image with predictions
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.warning(f"Could not read image {img_path}, skipping")
                        continue
                        
                    # Get the annotated image from the results
                    if len(results) > 0:
                        annotated_img = results[0].plot()
                        
                        # Display image with a window name that includes the index
                        window_name = f"Annotation Review ({idx+1}/{total_images})"
                        cv2.imshow(window_name, annotated_img)
                        
                        # Wait for key press with timeout (5 seconds before allowing ESC to skip)
                        key_pressed = False
                        start_time = time.time()
                        
                        while not key_pressed and (time.time() - start_time) < 300:  # 5-minute timeout
                            key = cv2.waitKey(100) & 0xFF  # Check every 100ms
                            
                            if key == ord('a'):  # Accept
                                logger.info(f"Accepted annotation for {img_path}")
                                key_pressed = True
                            elif key == ord('r'):  # Reject
                                logger.info(f"Rejected annotation for {img_path}")
                                # Remove the annotation file if it was created
                                base_name = os.path.splitext(img_path)[0]
                                if self.config['annotation']['format'] == 'yolo':
                                    annotation_file = f"{base_name}.txt"
                                    if os.path.exists(annotation_file):
                                        os.remove(annotation_file)
                                key_pressed = True
                            elif key == ord('q'):  # Quit review
                                logger.info("Manual review stopped by user")
                                cv2.destroyAllWindows()
                                return
                            elif key == 27:  # ESC key
                                logger.info(f"Skipped review for {img_path}")
                                key_pressed = True
                        
                        # Close the current window
                        cv2.destroyWindow(window_name)
                    else:
                        logger.info(f"No detections for {img_path}, skipping review")
                
                # Close all windows at the end
                cv2.destroyAllWindows()
                logger.info("Manual review completed")
                print("\nManual review completed")
                
            except KeyboardInterrupt:
                logger.info("Manual review interrupted by user")
                print("\nManual review interrupted by user")
                cv2.destroyAllWindows()
                
        except ImportError:
            logger.error("OpenCV not installed, cannot perform manual review")
            print("Manual review requires OpenCV. Please install with: pip install opencv-python")
    
    def run(self):
        """Run the image annotation pipeline."""
        logger.info("Starting Image Annotation Pipeline")
        
        start_time = time.time()
        
        try:
            # Step 1: Find all images
            image_paths = self.find_images()
            if not image_paths:
                logger.error("No images found in the specified directory")
                print("No images found in the specified directory. Please check the path and try again.")
                return
                
            # Step 2: Process images to determine which need annotation
            logger.info("Processing images to check for existing annotations")
            
            # Determine parallel processing level
            parallel_level = self.config['annotation']['parallel']
            if parallel_level == 0:
                # Sequential processing
                results = []
                for img_path in tqdm(image_paths, desc="Checking images"):
                    results.append(self.process_image(img_path))
            else:
                # Parallel processing
                import multiprocessing
                max_workers = multiprocessing.cpu_count()
                workers = max(1, int(max_workers * (parallel_level * 0.25)))
                
                logger.info(f"Using {workers} workers for parallel processing")
                
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_path = {executor.submit(self.process_image, img_path): img_path for img_path in image_paths}
                    
                    with tqdm(total=len(future_to_path), desc="Checking images") as pbar:
                        for future in concurrent.futures.as_completed(future_to_path):
                            result = future.result()
                            results.append(result)
                            pbar.update(1)
            
            # Unpack results
            image_paths = [r[0] for r in results]
            has_existing_annotation = [r[1] for r in results]
            
            # Step 3: Annotate images without existing annotations
            images_to_annotate = [img for img, has_annot in zip(image_paths, has_existing_annotation) if not has_annot]
            logger.info(f"Found {len(images_to_annotate)} images without annotations")
            
            if images_to_annotate:
                logger.info("Starting auto-annotation process")
                
                # First ensure the output directory exists and is clean
                predict_dir = os.path.join('runs', 'detect', 'predict')
                labels_dir = os.path.join(predict_dir, 'labels')
                
                # Clean up previous results if they exist
                if os.path.exists(predict_dir):
                    shutil.rmtree(predict_dir)
                
                # Load the YOLO model once
                try:
                    from ultralytics import YOLO
                    model = YOLO(self.config['annotation']['model_path'])
                    
                    # Process images in batches with consistent paths
                    batch_size = 16  # Default batch size
                    
                    for i in tqdm(range(0, len(images_to_annotate), batch_size), desc="Annotating images"):
                        batch = images_to_annotate[i:i+batch_size]
                        # Use consistent project and name parameters to ensure labels go to the same directory
                        model.predict(batch, save=True, save_txt=True, save_conf=True, 
                                     project='runs/detect', name='predict')
                        
                    logger.info(f"Auto-annotation completed for {len(images_to_annotate)} images")
                    
                except Exception as e:
                    logger.error(f"Error during auto-annotation: {str(e)}")
                    
            # Step 4: Manual review if enabled and requested
            if self.config['annotation']['manual_review']:
                try:
                    self.manual_review_annotations(image_paths, has_existing_annotation)
                except KeyboardInterrupt:
                    print("\nManual review interrupted. Proceeding with dataset creation.")
                    logger.info("Manual review interrupted by user. Proceeding with dataset creation.")
                
            # Step 5: Create dataset structure
            logger.info("Creating dataset structure")
            self.create_dataset_structure(image_paths, has_existing_annotation)
            
            # Calculate and show stats
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            logger.info(f"Processed {len(image_paths)} images")
            logger.info(f"Used existing annotations for {sum(has_existing_annotation)} images")
            logger.info(f"Created new annotations for {len(image_paths) - sum(has_existing_annotation)} images")
            
            print(f"\nPipeline completed! Your annotated dataset is available at: {self.config['output']['folder']}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Images processed: {len(image_paths)}")
            print(f"Existing annotations used: {sum(has_existing_annotation)}")
            print(f"New annotations created: {len(image_paths) - sum(has_existing_annotation)}")
        
        except KeyboardInterrupt:
            print("\nPipeline interrupted. Attempting to save partial results...")
            logger.info("Pipeline interrupted by user. Saving partial results.")
            
            try:
                # Try to create the dataset with what we have so far
                self.create_dataset_structure(image_paths, has_existing_annotation)
                print(f"Partial dataset saved to: {self.config['output']['folder']}")
            except Exception as e:
                logger.error(f"Failed to save partial results: {str(e)}")
                print("Failed to save partial results.")
            
            sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Image Annotation Pipeline for Road Damage Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    args = parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize pipeline
        pipeline = ImageAnnotationPipeline(args)
        
        # Run interactive CLI
        config = pipeline.run_interactive_cli()
        
        # Ask user for confirmation before proceeding
        proceed = input("\nReady to start processing. Continue? (y/n) [y]: ").strip().lower()
        if not proceed or proceed.startswith('y'):
            pipeline.run()
        else:
            print("Pipeline cancelled.")
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
