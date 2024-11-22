# CVPR Video Diffusion

This repository contains code for core models, and evaluation and common pre-processing utility tools for **Video Diffusion** experiments. It includes large video files and resources for demonstrating and evaluating the performance of diffusion-based models for video synthesis.

As demo video requires lfs to download due to storage limit of github, I also add a google drive link to download the sample demo video here (no identity and account meta data is included in the video, to follow the anonymous rules strictly.): [Download Anonymous Demo Video](https://drive.google.com/file/d/1BzpSGjx3U_Un3T3To4xW7xchKFeUIF1d/view?usp=sharing)

## Repository Structure

- **PanoDiffusion/**  
  Contains code and resources specific to panoramic video diffusion models.

- **VideoDiffusion/**  
  Core implementation and experiments for video diffusion-based techniques.

- **demo/**  
  Contains demonstration video files showcasing the output of diffusion models.

- **eval/**  
  Evaluation scripts and tools for measuring the performance of generated videos.

- **utils/**  
  Utility scripts for preprocessing and postprocessing video data.

## Features

- Large-scale video diffusion implementations.
- Panoramic video generation using diffusion models.
- Evaluation tools for metrics such as FVD and mTSED.
- Pretrained models and demo outputs.

## Requirements

- Python 3.9
- NumPy
- PyTorch 2.0
- Git LFS (for managing large files)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alexandor91/CVPRVideoDiffusion.git
   cd CVPRVideoDiffusion

