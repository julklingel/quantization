# COCO ResNet Project

This project implements three different ResNet models for the COCO dataset: classification, object detection, and image segmentation. The goal is to explore various quantization methods. MLFlow is utilized for experiment tracking.
## Project Structure

```
quantization
├── data
│   └── coco
│       ├── annotations
│       ├── train2017
│       └── val2017
├── models
│   ├── classification
│   │   └── resnet_classification.py
│   ├── detection
│   │   └── resnet_detection.py
│   └── segmentation
│       └── resnet_segmentation.py
├── notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── quantization_experiments.ipynb
├── scripts
│   ├── train_classification.py
│   ├── train_detection.py
│   ├── train_segmentation.py
│   └── quantization_methods.py
├── tests
│   ├── test_classification.py
│   ├── test_detection.py
│   ├── test_segmentation.py
│   └── test_quantization.py
├── mlflow_tracking.py
├── main.py
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/julklingel/quantization.git
   cd quantization
   ```

2. Install the required dependencies using conda (or mini-conda):
   ```
   conda env create -f environment.yml -n {your env name}
   ```

   ```
   conda activate {your env name}
   ```



3. Download the COCO dataset and place it in the `data/coco` directory.

## Usage

- When selecting a notebook dont forget to set the intepreter in the right corner to your conda env. 
- To explore the dataset, run the `data_exploration.ipynb` notebook.
- For training models, use the `model_training.ipynb` notebook.
- To experiment with quantization methods, refer to the `quantization_experiments.ipynb` notebook.
- The training scripts in the `scripts` directory can be executed directly for training specific models.

## MLflow Tracking

MLflow is used for tracking experiments. Ensure that the MLflow server is running before executing training scripts or notebooks. You can start the server with:
```
mlflow ui
```

## Goals

- Implement and evaluate ResNet models for different tasks on the COCO dataset.
- Experiment with quantization techniques to optimize model performance.
- Utilize MLflow for effective experiment tracking and comparison.