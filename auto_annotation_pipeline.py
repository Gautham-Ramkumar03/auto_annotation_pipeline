#!/usr/bin/env python3
"""
Road Damage Auto Annotation Pipeline

This script creates a complete pipeline for road damage detection dataset creation:
1. Processes videos to extract frames and apply augmentations
2. Runs a YOLOv8 model to automatically annotate the extracted frames
3. Organizes the annotated images into a proper dataset structure

Usage:
    python auto_annotation_pipeline.py

The script provides an interactive CLI for configuring both the video processing
and annotation stages of the pipeline.
"""

import os
import sys
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path
import tempfile
import warnings
from typing import Dict, Any, Optional

# Import the component modules
from video_processor import VideoProcessor
from auto_annotate import AutoAnnotator, parse_args as annotate_parse_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('pipeline')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class AutoAnnotationPipeline:
    """
    Main class for the auto annotation pipeline, combining video processing
    and automatic annotation into a single workflow.
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
            'video_processing': {},
            'annotation': {}
        }
        
        # Set up temp and output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"road_damage_temp_{timestamp}")
        self.output_dir = os.path.join(os.getcwd(), "dataset")
        
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
        print("\n=== Road Damage Auto Annotation Pipeline ===")
        print("This tool will process videos and automatically annotate road damages.")
        
        # === Video Processing Configuration ===
        print("\n=== Video Processing Configuration ===")
        
        # Input folder
        self.config['video_processing']['input_folder'] = self._prompt_input_folder()
        
        # Output image format
        self.config['video_processing']['output_format'] = self._prompt_output_format()
        
        # Augmentation level
        self.config['video_processing']['augmentation_level'] = self._prompt_augmentation_level()
        
        # === Annotation Configuration ===
        print("\n=== Annotation Configuration ===")
        
        # Model selection
        self.config['annotation']['model_path'] = self._prompt_model_path()
        
        # Annotation format
        self.config['annotation']['format'] = self._prompt_annotation_format()
        
        # Dataset splitting
        split_option = self._prompt_split_option()
        if split_option == 'no_split':
            self.config['annotation']['no_split'] = True
            self.config['annotation']['split'] = None
        else:
            self.config['annotation']['no_split'] = False
            self.config['annotation']['split'] = self._prompt_split_ratios()
            
        print("\n=== Configuration Summary ===")
        self._print_config_summary()
        
        return self.config
    
    def _prompt_input_folder(self):
        """Prompt user for folder path containing video files."""
        while True:
            folder_path = input("Enter the folder path containing video files [./videos]: ").strip()
            if not folder_path:
                folder_path = "./videos"
            if os.path.isdir(folder_path):
                return folder_path
            print(f"The path '{folder_path}' is not a valid directory. Please try again.")

    def _prompt_output_format(self):
        """Prompt user to select output image format."""
        print("\nSelect output image format:")
        formats = ['png', 'jpg', 'jpeg']
        for i, fmt in enumerate(formats, 1):
            print(f"{i}. {fmt.upper()}")
            
        while True:
            try:
                choice = input(f"Enter your choice (1-{len(formats)}) [2]: ").strip()
                if not choice:
                    return 'jpg'  # Default
                choice = int(choice)
                if 1 <= choice <= len(formats):
                    return formats[choice-1]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def _prompt_augmentation_level(self):
        """Prompt user for augmentation level."""
        print("\nAugmentation levels:")
        print("0. No augmentation")
        print("1. Basic augmentation (simple transforms)")
        print("2. Standard augmentation (recommended)")
        print("3. Advanced augmentation (comprehensive)")
        print("4. Road Damage Minimal (focused on crack detection)")
        
        while True:
            choice = input("Select augmentation level (0-4) [2]: ").strip()
            if not choice:
                return 2  # Default
            try:
                level = int(choice)
                if 0 <= level <= 4:
                    return level
                print("Invalid level. Please enter a number between 0 and 4.")
            except ValueError:
                print("Please enter a valid number.")
    
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
    
    def _print_config_summary(self):
        """Print summary of the selected configuration."""
        print("\nVideo Processing:")
        print(f"- Input folder: {self.config['video_processing']['input_folder']}")
        print(f"- Output format: {self.config['video_processing']['output_format']}")
        print(f"- Augmentation level: {self.config['video_processing']['augmentation_level']}")
        
        print("\nAnnotation:")
        print(f"- Model: {self.config['annotation']['model_path']}")
        print(f"- Format: {self.config['annotation']['format']}")
        
        if self.config['annotation']['no_split']:
            print("- Dataset split: No split (all images for training)")
        else:
            print(f"- Dataset split: {self.config['annotation']['split']}")
    
    def run_pipeline(self):
        """Run the complete annotation pipeline."""
        logger.info("Starting Auto Annotation Pipeline")
        
        # Create temporary directory
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Step 1: Process videos
        self._run_video_processing()
        
        # Step 2: Annotate images
        self._run_annotation()
        
        # Step 3: Clean up
        self._cleanup()
        
        logger.info(f"Pipeline completed successfully! Dataset saved to: {self.output_dir}")
        print(f"\nPipeline completed! Your annotated dataset is available at: {self.output_dir}")
    
    def _run_video_processing(self):
        """Run the video processing stage."""
        logger.info("Starting video processing stage")
        
        # Initialize VideoProcessor
        processor = VideoProcessor()
        
        # Use the temp_dir as the output directory for the video processor
        # This is done by monkey-patching the dataset_dir attribute
        processor.dataset_dir = self.temp_dir
        
        # Extract frames from videos
        video_input_folder = self.config['video_processing']['input_folder']
        output_format = self.config['video_processing']['output_format']
        aug_level = self.config['video_processing']['augmentation_level']
        
        # Find all videos in the input folder
        video_files = []
        for root, _, files in os.walk(video_input_folder):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in processor.supported_video_formats):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            logger.error(f"No video files found in {video_input_folder}")
            sys.exit(1)
            
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Extract frames
        total_frames = 0
        for video_path in video_files:
            extracted = processor.extract_frames(
                video_path, 
                self.temp_dir,
                output_format
            )
            total_frames += extracted
            
        logger.info(f"Extracted {total_frames} frames from videos")
        
        # Apply augmentations if enabled
        if aug_level > 0:
            logger.info(f"Applying augmentations (level {aug_level})...")
            
            all_images = [f for f in os.listdir(self.temp_dir) if f.endswith(f'.{output_format}')]
            aug_count = 0
            
            for img_file in all_images:
                img_path = os.path.join(self.temp_dir, img_file)
                aug_result = processor.apply_augmentations(
                    img_path, 
                    self.temp_dir, 
                    output_format, 
                    aug_level
                )
                aug_count += aug_result
                
            logger.info(f"Created {aug_count} augmented images")
        
        logger.info("Video processing stage completed")
    
    def _run_annotation(self):
        """Run the annotation stage."""
        logger.info("Starting annotation stage")
        
        # Create arguments for AutoAnnotator
        args = annotate_parse_args()
        
        # Override with our configuration
        args.input_dir = self.temp_dir
        args.output_dir = self.output_dir
        args.model = self.config['annotation']['model_path']
        args.format = self.config['annotation']['format']
        
        if self.config['annotation']['no_split']:
            args.no_split = True
        else:
            args.no_split = False
            args.split = self.config['annotation']['split']
        
        # Initialize and run AutoAnnotator
        try:
            annotator = AutoAnnotator(args)
            annotator.process_images()
            logger.info(f"Annotation completed successfully")
        except Exception as e:
            logger.error(f"Annotation failed: {str(e)}")
            sys.exit(1)
    
    def _cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory: {str(e)}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Auto Annotation Pipeline for Road Damage Detection',
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
        pipeline = AutoAnnotationPipeline(args)
        
        # Run interactive CLI
        config = pipeline.run_interactive_cli()
        
        # Ask user for confirmation before proceeding
        proceed = input("\nReady to start processing. Continue? (y/n) [y]: ").strip().lower()
        if not proceed or proceed.startswith('y'):
            pipeline.run_pipeline()
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
