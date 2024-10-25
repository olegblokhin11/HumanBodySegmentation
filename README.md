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
