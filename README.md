# Auto Annotation Pipeline

An automated pipeline for processing videos into annotated datasets for road damage detection and machine learning applications.

## Overview

The Auto Annotation Pipeline is a comprehensive tool designed to streamline the creation of annotated datasets for road damage detection. It combines video processing, frame extraction, image augmentation, and automatic annotation using YOLOv8 to generate high-quality labeled datasets for training machine learning models.

## Features

- **Video Processing**
  - Intelligent frame extraction with scene change detection
  - Multi-format video support (mp4, avi, mov, mkv, flv, wmv)
  - Multi-threaded processing for optimal performance
  
- **Image Augmentation**
  - Multiple augmentation levels (Basic, Standard, Advanced, Road Damage Minimal)
  - Time-of-day variations (morning, afternoon, evening, night)
  - Seasonal variations (summer, winter, autumn, spring)
  - Advanced augmentation techniques (mosaic, mixup, cutmix)
  
- **Automatic Annotation**
  - YOLOv8-powered object detection for road damage
  - Customizable confidence and IoU thresholds
  - Support for single-class or multi-class annotations
  
- **Dataset Management**
  - Multiple annotation formats (YOLO, COCO, Pascal VOC)
  - Configurable train/validation/test splits
  - Dataset merging capabilities
  - Automatic creation of dataset.yaml for YOLOv8 training

## Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for faster processing)
- CUDA and cuDNN (for GPU acceleration)
- Conda or Miniconda (for environment management)

## Installation

### Using Conda

1. Clone the repository:
   ```bash
   git clone https://github.com/Gautham-Ramkumar03/Auto_Annotation_Pipeline.git
   cd Auto_Annotation_Pipeline
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate auto_annotation_pipeline
   ```

3. Verify installation:
   ```bash
   python -c "from ultralytics import YOLO; print(f'YOLOv8 installation successful: {YOLO}')"
   ```

### Manual Installation

If you prefer not to use Conda, you can install the required packages manually:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the main dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install ultralytics opencv-python albumentations tqdm
   ```

## Usage

### Complete Pipeline (Recommended)

For most users, the recommended approach is to use the end-to-end pipeline script, which seamlessly integrates all components into a single workflow:

```bash
python auto_annotation_pipeline.py
```

This script provides:
- **Streamlined workflow** - handles the entire process from video processing to dataset creation
- **User-friendly interface** - interactive prompts guide you through all configuration options
- **Optimized pipeline** - ensures each stage's output integrates correctly with the next
- **Automatic resource management** - handles temporary files and directory structures
- **Error handling** - provides clear feedback and troubleshooting guidance

The pipeline guides you through:
1. Selecting video input folder
2. Configuring output format and augmentation settings
3. Selecting model and annotation format
4. Setting dataset split options

After configuration, the script:
1. Processes videos to extract frames
2. Applies selected augmentation techniques
3. Runs YOLOv8 to detect and annotate road damage
4. Organizes everything into a properly structured dataset ready for training

This all-in-one approach eliminates the need to manually coordinate between different scripts and ensures compatibility between steps.

### Individual Components

#### Video Processing

Extract frames from videos and apply augmentations:

```bash
python video_processor.py
```

The script will interactively prompt for:
- Input folder containing videos
- Output image format
- Augmentation level
- Dataset splitting method

#### Auto Annotation

Annotate existing images with a YOLOv8 model:

```bash
python auto_annotate.py --input-dir img_data --output-dir dataset --model models/best.pt --format yolo
```

Options:
- `--input-dir`: Directory containing images (default: img_data)
- `--output-dir`: Directory to save dataset (default: dataset)
- `--model`: Path to YOLOv8 model file (default: best.pt)
- `--format`: Annotation format (yolo, coco, voc)
- `--conf-threshold`: Confidence threshold (default: 0.25)
- `--iou-threshold`: IoU threshold for NMS (default: 0.7)
- `--batch-size`: Processing batch size (default: 8)
- `--single-class`: Treat all detections as a single class
- `--no-split`: Don't split dataset (all images in train folder)
- `--split`: Custom split ratios as "train,val,test" (default: "70,20,10")

#### Merge Datasets

Merge two annotated datasets:

```bash
python merge_datasets.py --base_dataset dataset --additional_dataset new_dataset --output_dataset merged_dataset --final_format yolo
```

Options:
- `--base_dataset`: Path to the primary dataset
- `--additional_dataset`: Path to the secondary dataset
- `--output_dataset`: Path for the merged dataset
- `--final_format`: Desired annotation format (yolo, coco, voc)

## Augmentation Levels

The pipeline offers multiple augmentation levels to enhance your dataset:

1. **Basic (Level 1)**
   - Horizontal flipping
   - Simple rotation (±10°)
   - Brightness adjustment

2. **Standard (Level 2)** - Recommended
   - All basic augmentations
   - Enhanced rotation (±15°)
   - Translation and scaling
   - Brightness, contrast, hue, saturation adjustments

3. **Advanced (Level 3)**
   - All standard augmentations
   - Perspective, elastic and grid distortions
   - Weather effects (rain, fog, shadow)
   - Cutout, blur, noise
   - Advanced techniques (mosaic, mixup, cutmix)

4. **Road Damage Minimal (Level 4)**
   - Conservative rotation (±10°)
   - Moderate brightness adjustment (±15%)
   - Resizing to 640x640 maintaining aspect ratio
   - Limited to 2 augmented versions per original image

## Directory Structure

After running the pipeline, the following directory structure is generated:

```
dataset/
├── train/
│   ├── images/       # Training images
│   └── labels/       # Annotations (format depends on selection)
├── val/
│   ├── images/       # Validation images
│   └── labels/       # Annotations (format depends on selection)
├── test/
│   ├── images/       # Test images
│   └── labels/       # Annotations (format depends on selection)
└── dataset.yaml      # YOLOv8 dataset configuration
```

## Annotation Formats

The pipeline supports three annotation formats:

1. **YOLO** (default)
   - One .txt file per image
   - Each line: `class_id center_x center_y width height` (normalized 0-1)
   - Compatible with YOLOv8 training

2. **COCO**
   - Single JSON file with all annotations
   - Contains image info, bounding boxes, and categories
   - Compatible with many frameworks

3. **Pascal VOC**
   - One XML file per image
   - Contains detailed annotation with bounding boxes
   - Widely supported format

## Example Workflow

1. **Prepare Videos**
   - Collect road condition videos
   - Organize them in a folder (e.g., `videos/`)

2. **Run the Pipeline**
   ```bash
   python auto_annotation_pipeline.py
   ```
   - Select the video folder
   - Choose augmentation level 2 (standard)
   - Select YOLOv8 model (or use the default)
   - Choose YOLO annotation format
   - Set dataset split (e.g., 70% train, 20% validation, 10% test)

3. **Train a Model with the Generated Dataset**
   ```bash
   yolo train data=dataset/dataset.yaml model=yolov8n.pt epochs=100
   ```

## Troubleshooting

### Video Processing Issues

- **Error**: "No video files found in input directory"
  - **Solution**: Check that your videos have supported extensions (.mp4, .avi, .mov, .mkv, .flv, .wmv)

- **Error**: "Failed to load video file"
  - **Solution**: Verify the video is not corrupted by opening it with a media player

### Annotation Issues

- **Error**: "No images found in directory"
  - **Solution**: Check that your images have supported extensions (.jpg, .jpeg, .png)

- **Error**: "Failed to load model"
  - **Solution**: Verify the model path and ensure the model file exists

- **Error**: "CUDA out of memory"
  - **Solution**: Decrease the batch size using the `--batch-size` option

## Contributing

Contributions to the Auto Annotation Pipeline are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Implement your changes
4. Run tests to ensure functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) team for their object detection framework
- The [Albumentations](https://albumentations.ai/) library for image augmentation capabilities
