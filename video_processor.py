import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import concurrent.futures
import argparse
import warnings
from functools import partial
import sys
import subprocess

# Suppress albumentations update warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class VideoProcessor:
    def __init__(self):
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        self.supported_image_formats = {
            'png': '.png',
            'jpg': '.jpg', 
            'jpeg': '.jpeg'
        }
        self.dataset_dir = None
        self.frames_extracted = 0
        
        # Configure OpenCV for optimal CPU performance
        cv2.setUseOptimized(True)
        cv2.setNumThreads(os.cpu_count())
        print(f"Using OpenCV {cv2.__version__} with optimizations enabled")
        print(f"CPU threads available: {os.cpu_count()}")
        
        # Check and install required packages
        self.check_and_install_packages()
        
        # Import libraries that might not be installed initially
        try:
            import imgaug.augmenters as iaa
            self.imgaug_available = True
            print("Using imgaug for additional augmentations")
        except ImportError:
            self.imgaug_available = False
            print("imgaug not available - some augmentations may be limited")
        
        try:
            import skimage
            self.skimage_available = True
            print("Using scikit-image for additional image processing")
        except ImportError:
            self.skimage_available = False
            print("scikit-image not available - some augmentations may be limited")
        
        # Print albumentations version
        print(f"Using Albumentations version: {A.__version__}")
        
    def check_and_install_packages(self):
        """Check and install required packages if needed"""
        required_packages = {
            'imgaug': 'imgaug',
            'skimage': 'scikit-image'  # Correct package name is scikit-image, not skimage
        }
        
        for import_name, pip_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"Found {import_name} package")
            except ImportError:
                print(f"{import_name} not found. Trying to install {pip_name}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                    print(f"Installed {pip_name}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install {pip_name}: {e}")
                    print(f"You may need to manually install {pip_name} package.")
                    if import_name == 'skimage':
                        print("For scikit-image, use: pip install scikit-image")
                except Exception as e:
                    print(f"Unexpected error installing {pip_name}: {e}")
    
    def get_input_folder(self):
        """Prompt user for folder path containing video files"""
        while True:
            folder_path = input("Enter the folder path containing video files: ").strip()
            if os.path.isdir(folder_path):
                return folder_path
            print(f"The path '{folder_path}' is not a valid directory. Please try again.")
    
    def get_output_format(self):
        """Prompt user to select output image format"""
        print("\nSelect output image format:")
        for i, fmt in enumerate(self.supported_image_formats.keys(), 1):
            print(f"{i}. {fmt.upper()}")
            
        while True:
            try:
                choice = int(input("Enter your choice (number): "))
                if 1 <= choice <= len(self.supported_image_formats):
                    return list(self.supported_image_formats.keys())[choice-1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def get_dataset_split(self):
        """Prompt user to select dataset splitting method"""
        print("\nSelect dataset splitting method:")
        print("A. Use all frames for training")
        print("B. Custom split percentage")
        print("C. Default split (70% train, 20% test, 10% validation)")
        
        while True:
            choice = input("Enter your choice (A/B/C): ").strip().upper()
            if choice == 'A':
                return {'train': 100, 'test': 0, 'val': 0}
            elif choice == 'B':
                train_pct = float(input("Enter percentage for training (0-100): "))
                test_pct = float(input("Enter percentage for testing (0-100): "))
                val_pct = 100 - (train_pct + test_pct)
                print(f"Validation set will be {val_pct}%")
                if train_pct + test_pct > 100 or train_pct < 0 or test_pct < 0:
                    print("Invalid percentages. They must sum to 100 or less and be positive.")
                    continue
                return {'train': train_pct, 'test': test_pct, 'val': val_pct}
            elif choice == 'C':
                return {'train': 70, 'test': 20, 'val': 10}
            else:
                print("Invalid choice. Please enter A, B, or C.")
                
    def get_augmentation_choice(self):
        """Ask if the user wants to enable augmentation"""
        print("\nAugmentation options:")
        print("0. No augmentation")
        print("1. Basic augmentation (simple transforms)")
        print("2. Standard augmentation (recommended)")
        print("3. Advanced augmentation (comprehensive)")
        print("4. Road Damage Minimal (focused on crack detection)")
        
        while True:
            try:
                choice = int(input("Select augmentation level (0-4): ").strip())
                if 0 <= choice <= 4:
                    return choice
                print("Invalid choice. Please enter a number between 0 and 4.")
            except ValueError:
                print("Please enter a valid number.")
    
    def setup_dataset_structure(self):
        """Create the dataset directory structure"""
        # Change to img_data folder instead of dataset
        self.dataset_dir = os.path.join(os.getcwd(), 'img_data')
        if os.path.exists(self.dataset_dir):
            choice = input(f"The directory '{self.dataset_dir}' already exists. Overwrite? (y/n): ").strip().lower()
            if choice.startswith('y'):
                shutil.rmtree(self.dataset_dir)
            else:
                print("Operation cancelled.")
                return False
                
        # Create single output directory
        os.makedirs(self.dataset_dir, exist_ok=True)
            
        return True
    
    def extract_frames(self, video_path, output_dir, output_format, sample_interval=30):
        """Extract frames using intelligent sampling based on scene changes and intervals"""
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error opening video file {video_path}")
            return 0
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = Path(video_path).stem
        
        # Scene change detection parameters
        prev_frame = None
        diff_threshold = 35.0  # Threshold for scene change detection
        
        # Counter for extracted frames
        extracted_count = 0
        
        # Initialize progress bar
        pbar = tqdm(total=frame_count, desc=f"Processing {video_name}")
        
        frame_idx = 0
        
        # Use a more efficient batch processing approach
        batch_size = 5  # Process 5 frames at once to reduce loop overhead
        frames_batch = []
        indices_batch = []
        
        while True:
            # Read a batch of frames
            for _ in range(batch_size):
                success, frame = video.read()
                if not success:
                    break
                frames_batch.append(frame)
                indices_batch.append(frame_idx)
                frame_idx += 1
            
            if not frames_batch:
                break
                
            # Update progress bar
            pbar.update(len(frames_batch))
            
            # Process the batch
            for i, frame in enumerate(frames_batch):
                idx = indices_batch[i]
                extract_this_frame = False
                
                # Interval-based extraction
                if idx % sample_interval == 0:
                    extract_this_frame = True
                    
                # Scene change detection
                if prev_frame is not None:
                    # Convert frames to grayscale for comparison
                    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate frame difference
                    frame_diff = cv2.absdiff(gray_curr, gray_prev)
                    mean_diff = np.mean(frame_diff)
                    
                    # If significant change detected
                    if mean_diff > diff_threshold:
                        extract_this_frame = True
                
                # Save frame if it should be extracted
                if extract_this_frame:
                    frame_filename = f"{video_name}_frame_{idx}.{output_format}"
                    output_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(output_path, frame)
                    extracted_count += 1
                
                # Update previous frame
                prev_frame = frame.copy()
                
            # Clear the batch
            frames_batch.clear()
            indices_batch.clear()
        
        # Close video and progress bar
        video.release()
        pbar.close()
        
        print(f"Extracted {extracted_count} frames from {video_name}")
        return extracted_count
        
    def build_augmentation_pipeline(self, level):
        """Build augmentation pipeline based on selected level"""
        if level == 0:
            return []
            
        # Basic augmentations (Level 1)
        basic_augs = [
            ("h_flip", A.HorizontalFlip(p=0.5)),
            ("rotate_simple", A.Rotate(limit=10, p=0.5)),
            ("brightness", A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.5))
        ]
        
        if level == 1:
            return basic_augs

        # Road Damage Minimal augmentations (Level 4)
        if level == 4:
            road_damage_augs = [
                # Essential augmentations for road damage detection
                ("h_flip", A.HorizontalFlip(p=0.5)),
                ("rotate", A.Rotate(limit=10, p=0.5)),  # Conservative -10° to +10°
                ("brightness", A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0, p=0.5))  # -15% to +15%
            ]
            return road_damage_augs
            
        # Standard augmentations (Level 2) - with compatibility fixes
        standard_augs = basic_augs.copy()
        
        # Add guaranteed compatible transforms
        standard_transforms = [
            ("rotate", A.Rotate(limit=15, p=0.7)),
            # Use A.Affine instead of A.Shift for translation/scaling
            ("affine_translate", A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, p=0.7)),
            ("affine_scale", A.Affine(scale=(0.8, 1.2), p=0.7)),
            # Color/Intensity adjustments
            ("brightness_contrast", A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7)),
            ("hue_sat_val", A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=25, p=0.5)),
        ]
        
        for name, transform in standard_transforms:
            try:
                standard_augs.append((name, transform))
            except Exception as e:
                print(f"Skipping {name} due to: {e}")
        
        if level == 2:
            return standard_augs
            
        # Advanced augmentations (Level 3) - with robust fallbacks
        advanced_augs = standard_augs.copy()
        
        # Try adding advanced transforms with fallbacks
        self.safely_add_transform(advanced_augs, "perspective", 
                               lambda: A.Perspective(scale=(0.05, 0.2), p=0.4),
                               lambda: A.Affine(shear=(-10, 10), p=0.3))  # Fallback to Affine
        
        self.safely_add_transform(advanced_augs, "elastic", 
                              lambda: A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                              lambda: A.Affine(scale=(0.9, 1.1), p=0.3))  # Fallback
        
        self.safely_add_transform(advanced_augs, "grid_distort", 
                               lambda: A.GridDistortion(distort_limit=0.1, p=0.3),
                               lambda: A.Affine(scale=(0.9, 1.1), rotate=(-5, 5), p=0.3))  # Fallback
        
        # Weather effects with fallbacks
        try:
            # Try to add RandomRain
            advanced_augs.append(("rain", A.RandomRain(
                slant_lower=-10, slant_upper=10, 
                drop_length=20, blur_value=3, p=0.3
            )))
        except AttributeError:
            # Fallback for rain effect using simple operations
            advanced_augs.append(("simulated_rain", 
                               A.Lambda(image=lambda x, **kwargs: self.simulate_rain(x))))
        
        try:
            # Try to add RandomFog
            advanced_augs.append(("fog", A.RandomFog(
                fog_coef_lower=0.1, fog_coef_upper=0.25, p=0.3
            )))
        except AttributeError:
            # Fallback for fog using simple brightness/contrast
            advanced_augs.append(("simulated_fog", 
                               A.Compose([
                                   A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=-0.1, p=1.0),
                                   A.GaussianBlur(blur_limit=(1, 3), p=1.0)
                               ], p=0.3)))
        
        try:
            # Try to add RandomShadow
            advanced_augs.append(("shadow", A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1), p=0.3
            )))
        except AttributeError:
            # No fallback for shadow as it's complex
            pass
            
        # Add remaining transforms with compatibility checks
        self.safely_add_transform(advanced_augs, "cutout",
                               lambda: A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                               lambda: A.Cutout(num_holes=4, max_h_size=32, max_w_size=32, p=0.3))
        
        self.safely_add_transform(advanced_augs, "blur", 
                               lambda: A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                               None)
        
        self.safely_add_transform(advanced_augs, "noise", 
                               lambda: A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                               None)
                               
        self.safely_add_transform(advanced_augs, "contrast_adaptive",
                               lambda: A.CLAHE(clip_limit=4.0, p=0.3),
                               lambda: A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.3, p=0.3))
                               
        return advanced_augs
    
    def safely_add_transform(self, pipeline, name, primary_transform, fallback_transform=None):
        """Safely add a transform to the pipeline with fallback"""
        try:
            # Try to add primary transform
            if callable(primary_transform):
                transform = primary_transform()
            else:
                transform = primary_transform
            pipeline.append((name, transform))
        except Exception as e:
            if fallback_transform:
                try:
                    # Try fallback
                    if callable(fallback_transform):
                        fallback = fallback_transform()
                    else:
                        fallback = fallback_transform
                    pipeline.append((f"{name}_fallback", fallback))
                    print(f"Using fallback for {name}")
                except Exception as e2:
                    print(f"Both primary and fallback for {name} failed: {e2}")
            else:
                print(f"Skipping {name} transform: {e}")
    
    def simulate_rain(self, image):
        """Custom rain effect implementation"""
        try:
            # Create rain streaks
            h, w = image.shape[:2]
            rain_layer = np.zeros_like(image)
            
            # Generate random rain streaks
            for _ in range(50):  # Number of rain drops
                x1, y1 = random.randint(0, w), random.randint(0, h)
                length = random.randint(5, 15)
                angle = random.uniform(-0.2, 0.2)
                thickness = random.randint(1, 2)
                
                # Calculate end point
                x2 = int(x1 + length * np.sin(angle))
                y2 = int(y1 + length * np.cos(angle))
                
                # Draw the streak
                cv2.line(rain_layer, (x1, y1), (x2, y2), (255, 255, 255), thickness)
            
            # Blur the rain layer
            rain_layer = cv2.GaussianBlur(rain_layer, (5, 5), 0)
            
            # Blend with original image
            alpha = 0.7
            result = cv2.addWeighted(image, alpha, rain_layer, 1 - alpha, 0)
            return result
        except Exception as e:
            print(f"Error in simulate_rain: {e}")
            return image  # Return original image if simulation fails
    
    def apply_augmentations(self, image_path, output_dir, output_format, aug_level=2):
        """Apply various augmentations to an image and save the results"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Couldn't read image {image_path}")
            return 0
        
        # Get augmentation pipeline based on level
        try:    
            augmentations = self.build_augmentation_pipeline(aug_level)
        except Exception as e:
            print(f"Error building augmentation pipeline: {e}")
            print("Falling back to basic augmentations")
            # Fallback to very basic augmentations that should work with any version
            augmentations = [
                ("h_flip", A.HorizontalFlip(p=0.5)),
                ("rotate", A.Rotate(limit=15, p=0.5))
            ]
            
        if not augmentations:
            return 0  # No augmentation requested
        
        base_filename = Path(image_path).stem
        aug_count = 0
        
        # Special handling for Road Damage Minimal (level 4)
        if aug_level == 4:
            # Generate exactly 2 augmented versions as requested
            transforms = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=0.7),
                    A.Rotate(limit=10, p=0.7),
                ], p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0, p=0.5)
            ])
            
            # Generate first augmentation version
            aug1 = transforms(image=image)['image']
            aug1_filename = f"{base_filename}_aug1.{output_format}"
            aug1_path = os.path.join(output_dir, aug1_filename)
            cv2.imwrite(aug1_path, aug1)
            aug_count += 1
            
            # Generate second augmentation version - ensure it's different
            aug2 = transforms(image=image)['image']
            aug2_filename = f"{base_filename}_aug2.{output_format}"
            aug2_path = os.path.join(output_dir, aug2_filename)
            cv2.imwrite(aug2_path, aug2)
            aug_count += 1
            
            # Resize to 640x640 maintaining aspect ratio
            original_h, original_w = image.shape[:2]
            aspect = original_w / original_h
            
            if aspect > 1:  # wider than tall
                new_w = 640
                new_h = int(640 / aspect)
            else:  # taller than wide
                new_h = 640
                new_w = int(640 * aspect)
                
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create 640x640 canvas and paste the resized image at center
            canvas = np.zeros((640, 640, 3), dtype=np.uint8)
            offset_x = (640 - new_w) // 2
            offset_y = (640 - new_h) // 2
            canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
            
            # Save the resized original
            resized_filename = f"{base_filename}_resized.{output_format}"
            resized_path = os.path.join(output_dir, resized_filename)
            cv2.imwrite(resized_path, canvas)
            aug_count += 1
            
            return aug_count
            
        # For all other augmentation levels, use the existing logic
        
        # Track successfully processed augmentations
        successful_augs = []
        img_height, img_width = image.shape[:2]
        
        # Apply each albumentations transform
        for aug_name, aug_transform in augmentations:
            try:
                augmented = aug_transform(image=image)['image']
                
                # Ensure the image is valid
                if augmented is None or augmented.size == 0 or augmented.shape[0] == 0 or augmented.shape[1] == 0:
                    print(f"Invalid image from {aug_name}, skipping")
                    continue
                
                aug_filename = f"{base_filename}_{aug_name}.{output_format}"
                aug_path = os.path.join(output_dir, aug_filename)
                cv2.imwrite(aug_path, augmented)
                aug_count += 1
                successful_augs.append(augmented)  # Keep track of successful augmentations
            except Exception as e:
                print(f"Error applying {aug_name} augmentation: {str(e)}")
        
        # Apply imgaug augmentations if available and we're in advanced mode
        if aug_level >= 3 and self.imgaug_available:
            try:
                import imgaug.augmenters as iaa
                
                # Create specialized imgaug augmenters
                imgaug_pipeline = [
                    ("road_curve", iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                    ("occlusion", iaa.Cutout(nb_iterations=1, size=0.15, squared=False)),
                    ("contrast_enhance", iaa.ContrastNormalization((0.8, 1.2))),
                    ("sharpen", iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.8, 1.2)))
                ]
                
                # Apply each imgaug augmenter
                for aug_name, augmenter in imgaug_pipeline:
                    try:
                        # Convert to imgaug format
                        augmented = augmenter.augment_image(image)
                        
                        aug_filename = f"{base_filename}_iaa_{aug_name}.{output_format}"
                        aug_path = os.path.join(output_dir, aug_filename)
                        cv2.imwrite(aug_path, augmented)
                        aug_count += 1
                        successful_augs.append(augmented)
                    except Exception as e:
                        print(f"Error applying imgaug {aug_name}: {str(e)}")
            except ImportError:
                print("Skipping imgaug augmentations - module not available")
        
        # Apply manual time-of-day variations - these are less likely to fail
        tod_augmentations = self.apply_time_of_day_variations(image, base_filename, output_dir, output_format)
        aug_count += tod_augmentations
        
        # Apply manual seasonal variations (if we have enough images)
        if len(successful_augs) > 0 and aug_level >= 3:
            seasonal_count = self.apply_seasonal_variations(
                successful_augs[0] if len(successful_augs) > 0 else image,
                base_filename, output_dir, output_format
            )
            aug_count += seasonal_count
        
        # Apply special augmentations that require multiple images (only for advanced level)
        if aug_level == 3:
            # Need to ensure we have enough valid images before applying mosaic/cutmix
            all_images = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                          if f.endswith(f'.{output_format}')]
            
            # Only proceed if we have enough images
            if len(all_images) >= 4:
                try:
                    # Implement mosaic augmentation with careful error handling
                    mosaic_img = self.create_mosaic([random.choice(all_images) for _ in range(4)], 
                                                target_size=(img_height, img_width))
                    if mosaic_img is not None:
                        mosaic_filename = f"{base_filename}_mosaic.{output_format}"
                        mosaic_path = os.path.join(output_dir, mosaic_filename)
                        cv2.imwrite(mosaic_path, mosaic_img)
                        aug_count += 1
                except Exception as e:
                    print(f"Error in mosaic augmentation: {e}")
                
                # Try to apply mixup and cutmix with better error handling
                if len(all_images) >= 2:
                    try:
                        second_img_path = random.choice(all_images)
                        second_img = cv2.imread(second_img_path)
                        
                        # Ensure second image is valid and same size as first
                        if second_img is not None and second_img.shape[:2] == image.shape[:2]:
                            # MixUp
                            alpha = random.uniform(0.3, 0.7)
                            mixup_img = cv2.addWeighted(image, alpha, second_img, 1-alpha, 0)
                            mixup_filename = f"{base_filename}_mixup.{output_format}"
                            mixup_path = os.path.join(output_dir, mixup_filename)
                            cv2.imwrite(mixup_path, mixup_img)
                            aug_count += 1
                            
                            # CutMix with careful region selection
                            try:
                                cutmix_img = image.copy()
                                h, w = cutmix_img.shape[:2]
                                cut_w = int(w * random.uniform(0.3, 0.5))
                                cut_h = int(h * random.uniform(0.3, 0.5))
                                cx = random.randint(0, w - cut_w)
                                cy = random.randint(0, h - cut_h)
                                
                                # Apply cutmix by replacing the region
                                cutmix_img[cy:cy+cut_h, cx:cx+cut_w] = second_img[cy:cy+cut_h, cx:cx+cut_w]
                                cutmix_filename = f"{base_filename}_cutmix.{output_format}"
                                cutmix_path = os.path.join(output_dir, cutmix_filename)
                                cv2.imwrite(cutmix_path, cutmix_img)
                                aug_count += 1
                            except Exception as e:
                                print(f"Error in cutmix operation: {e}")
                    except Exception as e:
                        print(f"Error in mixup augmentation: {e}")
        
        return aug_count
    
    def apply_time_of_day_variations(self, image, base_filename, output_dir, output_format):
        """Apply time of day variations with improved robustness"""
        count = 0
        
        try:
            # Morning - slightly cool/blue tint
            morning = image.copy()
            morning = cv2.convertScaleAbs(morning, alpha=1.1, beta=10)  # Brighter
            morning[:,:,0] = np.clip(morning[:,:,0] * 1.1, 0, 255)  # More blue
            morning_path = os.path.join(output_dir, f"{base_filename}_morning.{output_format}")
            cv2.imwrite(morning_path, morning)
            count += 1
        except Exception as e:
            print(f"Error creating morning variation: {e}")
        
        try:
            # Afternoon - slightly warm/yellow
            afternoon = image.copy()
            afternoon[:,:,2] = np.clip(afternoon[:,:,2] * 1.1, 0, 255)  # More red
            afternoon[:,:,1] = np.clip(afternoon[:,:,1] * 1.05, 0, 255)  # Slightly more green
            afternoon_path = os.path.join(output_dir, f"{base_filename}_afternoon.{output_format}")
            cv2.imwrite(afternoon_path, afternoon)
            count += 1
        except Exception as e:
            print(f"Error creating afternoon variation: {e}")
        
        try:
            # Evening - orange/golden hour
            evening = image.copy()
            evening[:,:,2] = np.clip(evening[:,:,2] * 1.2, 0, 255)  # More red
            evening[:,:,1] = np.clip(evening[:,:,1] * 1.1, 0, 255)  # More green
            evening[:,:,0] = np.clip(evening[:,:,0] * 0.8, 0, 255)  # Less blue
            evening = cv2.convertScaleAbs(evening, alpha=0.9, beta=0)  # Slightly darker
            evening_path = os.path.join(output_dir, f"{base_filename}_evening.{output_format}")
            cv2.imwrite(evening_path, evening)
            count += 1
        except Exception as e:
            print(f"Error creating evening variation: {e}")
        
        try:
            # Night - dark with blue tint
            night = image.copy()
            night = cv2.convertScaleAbs(night, alpha=0.6, beta=-20)  # Much darker
            night[:,:,0] = np.clip(night[:,:,0] * 1.1, 0, 255)  # More blue
            night_path = os.path.join(output_dir, f"{base_filename}_night.{output_format}")
            cv2.imwrite(night_path, night)
            count += 1
        except Exception as e:
            print(f"Error creating night variation: {e}")
            
        return count
    
    def apply_seasonal_variations(self, image, base_filename, output_dir, output_format):
        """Apply seasonal variations to the images"""
        count = 0
        
        try:
            # Summer - warmer, brighter
            summer = image.copy()
            summer[:,:,2] = np.clip(summer[:,:,2] * 1.1, 0, 255)  # More red
            summer[:,:,1] = np.clip(summer[:,:,1] * 1.1, 0, 255)  # More green
            summer = cv2.convertScaleAbs(summer, alpha=1.1, beta=5)  # Brighter
            summer_path = os.path.join(output_dir, f"{base_filename}_summer.{output_format}")
            cv2.imwrite(summer_path, summer)
            count += 1
        except Exception as e:
            print(f"Error creating summer variation: {e}")
        
        try:
            # Winter - cooler, bluer
            winter = image.copy()
            winter[:,:,0] = np.clip(winter[:,:,0] * 1.2, 0, 255)  # More blue
            winter[:,:,1] = np.clip(winter[:,:,1] * 0.9, 0, 255)  # Less green
            winter[:,:,2] = np.clip(winter[:,:,2] * 0.9, 0, 255)  # Less red
            winter = cv2.convertScaleAbs(winter, alpha=1.05, beta=10)  # Slightly brighter (snow effect)
            winter_path = os.path.join(output_dir, f"{base_filename}_winter.{output_format}")
            cv2.imwrite(winter_path, winter)
            count += 1
        except Exception as e:
            print(f"Error creating winter variation: {e}")
        
        try:
            # Autumn - orange/yellow tint
            autumn = image.copy()
            autumn[:,:,2] = np.clip(autumn[:,:,2] * 1.15, 0, 255)  # More red
            autumn[:,:,1] = np.clip(autumn[:,:,1] * 1.05, 0, 255)  # More green
            autumn[:,:,0] = np.clip(autumn[:,:,0] * 0.85, 0, 255)  # Less blue
            autumn_path = os.path.join(output_dir, f"{base_filename}_autumn.{output_format}")
            cv2.imwrite(autumn_path, autumn)
            count += 1
        except Exception as e:
            print(f"Error creating autumn variation: {e}")
        
        try:
            # Spring - slightly green tint
            spring = image.copy()
            spring[:,:,1] = np.clip(spring[:,:,1] * 1.1, 0, 255)  # More green
            spring = cv2.convertScaleAbs(spring, alpha=1.05, beta=5)  # Slightly brighter
            spring_path = os.path.join(output_dir, f"{base_filename}_spring.{output_format}")
            cv2.imwrite(spring_path, spring)
            count += 1
        except Exception as e:
            print(f"Error creating spring variation: {e}")
            
        return count
    
    def create_mosaic(self, img_paths, target_size=(640, 640)):
        """Create a mosaic from 4 images with improved error handling"""
        try:
            # Load and validate images
            imgs = []
            for path in img_paths[:4]:
                try:
                    img = cv2.imread(path)
                    if img is not None and img.size > 0 and len(img.shape) == 3:
                        imgs.append(img)
                except Exception as e:
                    print(f"Error loading image for mosaic: {e}")
            
            if len(imgs) < 4:
                print(f"Not enough valid images for mosaic (found {len(imgs)})")
                return None
            
            # Get dimensions from the first image if target size is not specified
            h, w = imgs[0].shape[:2]
            mosaic_width = target_size[1] if target_size else w*2
            mosaic_height = target_size[0] if target_size else h*2
                
            # Create empty mosaic with proper dimensions
            mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
            
            # Calculate half dimensions
            h_half = mosaic_height // 2
            w_half = mosaic_width // 2
            
            # Resize and place images
            for i, img in enumerate(imgs):
                # Resize each image to fit its quadrant
                if i == 0:  # Top-left
                    resized = cv2.resize(img, (w_half, h_half))
                    mosaic[:h_half, :w_half] = resized
                elif i == 1:  # Top-right
                    resized = cv2.resize(img, (w_half, h_half))
                    mosaic[:h_half, w_half:] = resized
                elif i == 2:  # Bottom-left
                    resized = cv2.resize(img, (w_half, h_half))
                    mosaic[h_half:, :w_half] = resized
                else:  # Bottom-right
                    resized = cv2.resize(img, (w_half, h_half))
                    mosaic[h_half:, w_half:] = resized
                
            return mosaic
            
        except Exception as e:
            print(f"Error creating mosaic: {str(e)}")
            return None
    
    def distribute_images(self, temp_dir, output_format):
        """Move all images to a single output folder"""
        all_images = [f for f in os.listdir(temp_dir) if f.endswith(f'.{output_format}')]
        
        # Move images to output directory
        for img in all_images:
            src = os.path.join(temp_dir, img)
            dst = os.path.join(self.dataset_dir, img)
            shutil.copy2(src, dst)
                
        return len(all_images)
        
    def process_videos(self):
        """Main function to process videos and create dataset"""
        # Try importing now in case installation succeeded
        try:
            import imgaug.augmenters as iaa
            self.imgaug_available = True
        except ImportError:
            pass
            
        try:
            import skimage
            self.skimage_available = True
        except ImportError:
            pass
            
        # Get user inputs
        input_folder = self.get_input_folder()
        output_format = self.get_output_format()
        
        # Remove dataset split choice and always use 100% for single folder
        
        # Get augmentation level instead of just yes/no
        aug_level = self.get_augmentation_choice()
        enable_augmentation = aug_level > 0
        
        # Setup dataset structure
        if not self.setup_dataset_structure():
            return
            
        # Create temporary directory for extracted frames
        temp_dir = os.path.join(self.dataset_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Find all video files
        video_files = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in self.supported_video_formats):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            print(f"No video files found in {input_folder}")
            return
            
        print(f"\nFound {len(video_files)} video files to process")
        
        # Process each video with optimized parallelism
        total_frames = 0
        max_workers = min(os.cpu_count(), len(video_files))
        
        if max_workers > 1:
            print(f"Processing videos in parallel using {max_workers} CPU workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for video_path in video_files:
                    futures.append(
                        executor.submit(self.extract_frames, video_path, temp_dir, output_format)
                    )
                    
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Videos processed"):
                    total_frames += future.result()
        else:
            # Sequential processing for a single video
            for video_path in video_files:
                extracted = self.extract_frames(
                    video_path, 
                    temp_dir, 
                    output_format
                )
                total_frames += extracted
                
        print(f"\nTotal frames extracted: {total_frames}")
        
        # Apply augmentations if enabled with optimized parallelism
        if enable_augmentation:
            print(f"\nApplying augmentations (level {aug_level})...")
            
            # Special message for Road Damage Minimal option
            if aug_level == 4:
                print("\nUsing minimal road damage augmentation settings:")
                print("- Horizontal flip (preserves road orientation)")
                print("- Conservative rotation (-10° to +10°)")
                print("- Moderate brightness adjustment (-15% to +15%)")
                print("- Resizing to 640x640 while maintaining aspect ratio")
                print("Generating 2 augmented versions per original image")
            
            aug_count = 0
            
            all_images = [f for f in os.listdir(temp_dir) if f.endswith(f'.{output_format}')]
            
            # For safety, use ThreadPoolExecutor instead of ProcessPoolExecutor for augmentations
            # This avoids pickling issues with the augmentation objects across processes
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # Create a partial function with fixed arguments
                augment_func = partial(self.apply_augmentations, output_dir=temp_dir, output_format=output_format, aug_level=aug_level)
                
                # Map the function across all images - more efficient than submitting individual tasks
                image_paths = [os.path.join(temp_dir, img) for img in all_images]
                
                # Process in chunks to reduce overhead
                chunk_size = max(1, len(all_images) // (os.cpu_count() * 2))
                results = list(tqdm(
                    executor.map(augment_func, image_paths),  # Remove chunksize for ThreadPoolExecutor
                    total=len(image_paths),
                    desc="Augmenting"
                ))
                
                aug_count = sum(results)
                
                # Apply time-of-day simulations for advanced mode (level 3)
                if aug_level == 3:
                    print("Applying time-of-day simulations...")
                    # Fix: Use the correct method name
                    tod_func = partial(self.apply_time_of_day_variations, output_dir=temp_dir, output_format=output_format)
                    tod_results = list(tqdm(
                        executor.map(tod_func, image_paths),  # Remove chunksize for ThreadPoolExecutor
                        total=len(image_paths),
                        desc="Time of day simulation"
                    ))
                    tod_count = sum(tod_results)
                    aug_count += tod_count
                    print(f"Created {tod_count} time-of-day variations")
                    
            print(f"Created {aug_count} augmented images")
            total_frames += aug_count
        
        # Distribute images to single folder
        print("\nMoving images to output folder...")
        total_images = self.distribute_images(temp_dir, output_format)
        
        # Remove temporary directory
        shutil.rmtree(temp_dir)
        
        # Print summary
        print("\nDataset creation complete!")
        print(f"Total images: {total_frames}")
        print(f"All images saved to {self.dataset_dir}")

if __name__ == "__main__":
    print("Auto Annotation Pipeline - Video Processor")
    print("------------------------------------------")
    print("A tool to create ML-ready datasets from videos for road damage detection")
    try:
        processor = VideoProcessor()
        processor.process_videos()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf you're having package installation issues, try manually installing these first:")
        print("pip install imgaug scikit-image")
