# Human Body Segmentation Project

This project implements a hierarchical semantic segmentation pipeline using a custom deep learning model (`HierarchicalDeepLabV3`). The project includes scripts for training the model, evaluating its performance, and a Streamlit application for visualizing segmentation results.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Interactive Visualization](#interactive-visualization)
- [Project Structure](#project-structure)
- [Human Body Class Hierarchy](#human-body-class-hierarchy)
- [Example Results](#example-results)

## Overview

This project aims to develop a hierarchical segmentation model that can identify multiple levels of semantic categories from an image. It uses a modified version of the DeepLabV3 architecture to perform segmentation at three different hierarchical levels. The project provides:
- **Training Script**: To train the model on your dataset.
- **Evaluation Script**: To evaluate the model's performance on a validation set.
- **Interactive App**: A Streamlit-based app to visualize segmentation results.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/olegblokhin11/HumanBodySegmentation.git
cd human-body-segmentation
pip install -r requirements.txt
```

### Install Dependencies
The project requires specific versions of packages listed in the requirements.txt file. Install these dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
### Training
To train the segmentation model, use the `train.py` script. Make sure to set up the appropriate configuration file (`.yml`) before starting the training.

```bash
python train.py
```

**Training Configuration:**
- **Backbone**: Choose the backbone model for the architecture (e.g., ResNet50).
- **Learning Rate**: Specify the learning rate and scheduler in the `.yml` config file.
- **Epochs**: Adjust the number of epochs for training.
- **Batch Size**: Customize the batch size as needed.
- **And More**: Other parameters, such as momentum, weight decay, image augmentation settings, can also be adjusted in the configuration file.

### Evaluation
Evaluate the trained model using the `evaluate.py` script. This will load weights from a checkpoint and run the model on the validation dataset, providing metrics such as Pixel Accuracy, Class Accuracy, Mean IoU.
```bash
python evaluate.py
```

**Output Metrics:** The evaluation script displays metrics for each hierarchical level, such as:
- **Pixel Accuracy**: Measures the percentage of correctly classified pixels.
- **Class Pixel Accuracy**: Calculates the average accuracy per class, **excluding the background class** to focus on segmentation performance for non-background classes.
- **Mean IoU**: Intersection over Union averaged across all classes, **excluding the background class** for a focused evaluation on meaningful regions.

**Current Best Model Metrics:** The following metrics were achieved by the current best model on the validation dataset:
| Level | Pixel Accuracy | Class Pixel Accuracy | Mean IoU |
| :---      |    :---:   |         :---:        |   :---:  |
| Level 0   | 0.964      | 0.935                | 0.868    |
| Level 1   | 0.956      | 0.829                | 0.732    |
| Level 2   | 0.929      | 0.721                | 0.593    |

You can download the best model checkpoint from [Google Drive](https://drive.google.com/file/d/1Bo2IQ5gkCfM9fzLZFOaNjoZlD6nPuX6u/view?usp=sharing) and use it for further evaluation or inference.

## Interactive Visualization
To run the web app and visualize segmentation results, use the `app.py` script. This allows users to upload images and see segmented outputs for each hierarchical level.
```bash
streamlit run app.py
```
To specify a custom port (e.g., 8501), use the `--server.port` option:
```bash
streamlit run app.py --server.port 8501
```
**Accessing the App in a Browser**: After running the command, open your browser and navigate to:
```bash
http://localhost:8501
```

**Features:**
- Upload an image and get segmented outputs for different levels.
- View color-coded segmentation maps with a legend.
- Compare results easily across different hierarchical levels.
