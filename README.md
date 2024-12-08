# ScanInsight: Brain Tumor Detection & Classification

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Authors: [Zain Ali](https://github.com/zainnobody), [Hallie Kinsey](https://github.com/halliekinsey), and [Liam Richardson](https://github.com/oliwansd)


## Overview

**ScanInsight** is a prototype computer vision tool designed to assist medical professionals in the detection and classification of brain tumors from MRI scans. It leverages advanced YOLO-based object detection models (YOLOv8 and YOLOv11) to identify tumor regions and provide bounding boxes for three distinct tumor classes. The system aims to expedite the diagnostic process and improve accuracy by providing an intuitive interface that allows users—particularly clinical staff—to visualize model predictions.

This project is backed by comprehensive documentation and presentation materials:
- The **written report** (Team_01_FinalReport.pdf) outlining methodology, results, and conclusions is available [here](https://docs.google.com/document/d/1Wc9xgoGkn9yrXyPmxzmy7rC9WnQwS3EanVA5cSMT1SQ/edit?usp=sharing).
- This **full code base** (including the final notebook) is hosted on [GitHub](https://github.com/zainnobody/AAI-521-Final). The main analysis and training pipeline can be found in the [ScanInsight.ipynb notebook](https://github.com/zainnobody/AAI-521-Final/blob/main/ScanInsight.ipynb).
- A **team presentation video** (Team_01_Video_Presentation.mp4) summarizing the project’s approach and findings is available [here](https://drive.google.com/file/d/1AhfSrMsqmyj2uoKmN0KADvaH49bWOU6U/view?usp=sharing), accompanied by [presentation slides](https://docs.google.com/presentation/d/1PwXq2o4taYhlvK3_erAVULdhh83SbpTKYSoR2wHVVuk/edit?usp=sharing).

This repository includes:

- **Data and Results**: Preprocessed MRI image datasets, annotations, metrics, and results from YOLO model training runs.
- **Models**: Code and checkpoints for YOLOv8 and YOLOv11 models fine-tuned for brain tumor detection.
- **Notebooks & Scripts**: Jupyter notebooks for data exploration, training, evaluation, and visualization, as well as a Python UI script for a standalone demonstration.
- **Documentation**: This README and additional project documentation for workflow understanding, extensibility, and reproducibility.

*Disclaimer:* ScanInsight is a prototype and not a production-level clinical tool. Always refer to qualified medical professionals for clinical decisions.

## Features

- **Automatic Tumor Detection**: Uses YOLO-based models to detect and classify brain tumor types in MRI scans.
- **Data Exploration**: The `ScanInsight.ipynb` notebook provides thorough EDA (Exploratory Data Analysis), including visualization of slices, bounding boxes, and metrics.
- **Model Training & Evaluation**: Scripts to train YOLOv8 and YOLOv11 models, visualize performance curves (Precision-Recall, F1), and generate confusion matrices.
- **User Interface (UI)**: A standalone Python UI application (`ScanInsight-UI.py`) that allows users to:
  - Load trained models.
  - Select images or entire volumes for inference.
  - Visualize predictions and bounding boxes.
  - Save comments and notes about predictions for later review.
  
## Repository Structure

```
AAI-521-Final
|-- README.md                  # This README file
|-- ScanInsight.ipynb          # Main Jupyter notebook for EDA, training, and evaluation
|-- ScanInsight-UI.py          # Standalone UI application script
|-- model_paths.txt            # File containing references to trained model checkpoints
|-- comments/
|   `-- volume_8_slice_41_..._comments.json  # Example comment file
|-- external-research/
|   `-- yolo11-research.webp   # Reference image or research snippet
|-- results/
|   |-- yolov11n_brain_tumor_detection_v1/
|   |   |-- detect_yolo11/     # YOLOv11 generated metrics, figures, weights, etc.
|   |   |-- metrics.pkl        # Pickled metrics dictionary from YOLOv11 run
|   |   `-- yolov11n_brain_tumor_detection_v1.pt  # Trained YOLOv11 model
|   `-- yolov8n_brain_tumor_detection_v1/
|       |-- detect_yolo8/      # YOLOv8 generated metrics, figures, weights, etc.
|       `-- yolov8n_brain_tumor_detection_v1.pt   # Trained YOLOv8 model
|
|-- data/
|   |-- (Dataset structure with images and labels)
|   |-- brain_tumor_detection_path_data.csv
|   `-- expanded_brain_tumor_annotations.csv
|
`-- README.md (this file)
```

**Key Files:**

- **ScanInsight.ipynb**:  
  - Performs dataset exploration, preprocessing, and visualization.
  - Trains YOLO models (YOLOv8 and YOLOv11) and evaluates their performance.
  - Visualizes metrics (precision, recall, mAP, confusion matrices).
  
- **ScanInsight-UI.py**:  
  - A standalone Python script implementing a Tkinter-based GUI.
  - Users can load a model from `model_paths.txt`, select MRI images or image folders, and view model predictions.
  - Allows saving of per-image comments in JSON.

- **model_paths.txt**:  
  - Maps model names (e.g., `YOLOv8n`, `YOLOv11n`) to their corresponding `.pt` checkpoint files.
  - The UI script uses this file to let users choose which model to load.

- **results**:  
  - Contains model training results, including metrics, graphs (F1, PR, Precision, Recall curves), confusion matrices, and final model weights.
  - Organized by model version and run identifier.

- **data**:  
  - Includes the MRI datasets and labels used for training and evaluation.
  - CSV files (`brain_tumor_detection_path_data.csv`, `expanded_brain_tumor_annotations.csv`) provide metadata and label expansions for easier analysis.

## Installation & Dependencies

**Prerequisites:**

- Python 3.8+  
- Jupyter Notebook (for running `.ipynb` files)  
- PyTorch, torchvision  
- ultralytics (for YOLO models)  
- OpenCV, Pillow, NumPy, Matplotlib, Pandas, Seaborn  
- Tkinter (included with most Python installations) for the UI

**Installation Steps:**

1. Clone the repository:
   ```bash
   git clone https://github.com/zainnobody/AAI-521-Final.git
   cd AAI-521-Final
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install the above-mentioned packages manually.)*

3. (Optional) Launch Jupyter Notebook to explore `ScanInsight.ipynb`:
   ```bash
   jupyter notebook ScanInsight.ipynb
   ```

4. Run the UI:
   ```bash
   python ScanInsight-UI.py
   ```

## Usage

- **Exploring Data & Training Models**:  
  Open `ScanInsight.ipynb` in Jupyter Notebook. Follow the notebook cells to:
  - Perform EDA on the MRI dataset.
  - Train YOLOv8 and YOLOv11 models.
  - Generate evaluation metrics and curves.

- **UI Dashboard**:  
  Run `ScanInsight-UI.py` with Python. In the GUI:
  - Accept the disclaimer.
  - Load a model from the dropdown (e.g., YOLOv11n).
  - Choose an image or folder of images (volume).
  - View predictions, bounding boxes, and save comments.

## Model Performance

- **YOLOv8**: Initial tests resulted in ~55% mAP@50 with ~51.6% recall and ~64.6% precision.
- **YOLOv11**: Final chosen model, demonstrating stable performance across training, validation, and test sets (~52% mAP@50), with improved speed and efficiency. Although slightly lower precision/recall than YOLOv8, YOLOv11’s architecture is more advanced, making it potentially more scalable.

For more details, refer to the plots, metrics summaries, and confusion matrices saved under `results/`.

## Limitations & Future Work

- **Not a Clinical Tool**: This is a research prototype and must not be used for clinical decision-making.
- **Data Imbalance**: Future work may involve data augmentation, class weighting, or more extensive model tuning to handle underrepresented tumor classes better.
- **Model Improvements**: Attempting longer training periods, using more advanced YOLO architectures, or integrating additional domain knowledge may enhance performance.

## References

- [Medical Image Dataset for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection/data)
- Ultralytics YOLOv8 & YOLOv11 Documentation  
- Relevant academic and industry literature on object detection and imbalanced classification (cited within the `ScanInsight.ipynb`).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
