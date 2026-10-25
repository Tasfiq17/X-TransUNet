# X-TransUNet

**A Privacy-Preserving, Explainable Federated Learning Framework for Chest X-Ray Lung Segmentation**

## Overview

X-TransUNet is a federated deep learning framework for automated lung segmentation from chest X-ray images.

The project explores collaborative model training across distributed datasets without requiring participating institutions to exchange their raw training images.

It combines a TransUNet-based segmentation architecture with federated learning and explainable AI techniques to investigate segmentation performance and model interpretability.

## Key Features

- Federated learning simulation using Flower
- TransUNet-based lung segmentation
- Transformer-based feature representation
- Attention-map visualization
- Grad-CAM++ explainability analysis
- Dice coefficient and Intersection over Union (IoU) evaluation
- Comparison with centralized training
- Ablation experiments without the transformer component
- External evaluation using the JSRT dataset

## Implementation

The implementation is provided as a Google Colab notebook.

**Main file:** `X_TransUNet_Research.ipynb`

The project uses Python and PyTorch, together with Flower, OpenCV, Albumentations, and other supporting libraries.

## Getting Started

1. Open the notebook in Google Colab.
2. Install the required dependencies using the installation cells.
3. Mount Google Drive.
4. Prepare the datasets according to the paths specified in the notebook.
5. Run preprocessing and model initialization.
6. Execute the federated training experiments.
7. Evaluate the trained model and generate the visualizations.

Some evaluation cells require previously saved checkpoints and must be executed after training.

## Experimental Evaluation

The notebook includes federated and centralized experiments, segmentation performance evaluation, ablation studies, and explainability visualizations.

The reported experimental findings and detailed methodology are described in the associated research manuscript.

## Data Availability

Datasets are not redistributed through this repository. Users must obtain the required datasets from their original providers and comply with the applicable licensing and access conditions.

## Model Checkpoints

The notebook references saved model checkpoints for evaluating trained models. These checkpoints are not included unless separately provided in the repository or its releases.

## Associated Publication

**X-TransUNet: A Privacy-Preserving, Explainable Federated Learning Framework for Chest X-Ray Lung Segmentation**

Journal submission.

## Citation

If you use this implementation, please cite the associated research publication once its bibliographic details are available.

## License

See the repository's LICENSE file for applicable terms.
