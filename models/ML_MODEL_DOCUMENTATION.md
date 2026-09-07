# ML Model Documentation

## 1. Project Overview

TerraSpectra is a hyperspectral crop disease forecasting system designed to detect crop stress and disease using hyperspectral data.

The machine learning pipeline uses 3D Convolutional Neural Networks (3D-CNN) and Vision Transformers (ViT) to extract spatial and spectral features from hyperspectral image patches.

## 2. Dataset

The model uses preprocessed hyperspectral spectral patches.

- Number of samples: 78,400
- Patch size: 5 × 5
- Spectral bands: 20
- Number of classes: 4

Original patch shape:

`(78400, 5, 5, 20)`

Model input tensor shape:

`(78400, 1, 20, 5, 5)`

The dataset was divided into training and testing sets using an 80:20 split.

- Training samples: 62,720
- Testing/validation samples: 15,680

## 3. 3D-CNN Model

A 3D Convolutional Neural Network was implemented using PyTorch.

The 3D-CNN processes both spatial and spectral information from hyperspectral data.

Main components include:

- Conv3D layers
- Batch Normalization
- ReLU activation
- Pooling layers
- Dropout
- Fully connected classification layer

The CNN was tested using sample hyperspectral tensors to verify the forward pass and output dimensions.

## 4. Vision Transformer (ViT)

A lightweight Vision Transformer baseline was used to extract feature representations from hyperspectral patches.

The ViT helps capture relationships between different spatial and spectral features using attention mechanisms.

## 5. Hybrid CNN + ViT Model

A hybrid architecture was developed by combining CNN and ViT features.

The CNN extracts local spatial-spectral features, while the Vision Transformer captures global feature relationships.

The extracted features are combined and passed to the final classifier.

The model produces predictions for four crop disease/stress classes.

## 6. Training

The Hybrid CNN + ViT model was trained using the prepared spectral patches.

Training was performed for 5 epochs.

Example training results:

- Epoch 1 Accuracy: approximately 52%
- Epoch 2 Accuracy: approximately 54%
- Epoch 3 Accuracy: approximately 55%
- Epoch 4 Accuracy: approximately 57%
- Epoch 5 Accuracy: approximately 58%

The final trained model was saved as:

`models/final_hybrid_model.pth`

## 7. Model Evaluation

The trained Hybrid CNN + ViT model was evaluated on the test dataset.

Final validation result:

- Total validation predictions: 15,680
- Correct predictions: 9,248
- Validation Accuracy: 58.98%

The prediction count matched the number of validation samples, so prediction validation passed successfully.

## 8. CNN vs Hybrid Comparison

The standalone CNN model and Hybrid CNN + ViT model were compared.

The Hybrid model achieved slightly better performance than the standalone CNN.

Approximate results:

| Model | Accuracy |
|---|---:|
| 3D-CNN | ~57% |
| Hybrid CNN + ViT | ~58% |

This indicates that combining CNN spatial-spectral features with ViT global features can improve classification performance.

## 9. Tile Testing

The model was also tested on individual raster tiles.

Raster tiling was used to divide larger hyperspectral data into smaller tiles suitable for model processing.

The tile prediction test completed successfully.

## 10. Batch Inference

Batch inference was implemented to process multiple tiles together.

This improves inference efficiency compared with processing every tile individually.

The batch inference test completed successfully.

## 11. Prediction Validation

Predictions generated from raster tiles were validated against the expected output structure.

The number of predictions matched the number of input samples.

Result:

`Prediction validation PASSED!`

## 12. Model Performance and Error Testing

Model performance was tested using sample data and prediction results.

The testing process checked:

- Number of predictions
- Correct and incorrect predictions
- Model accuracy
- Prediction consistency

These tests helped identify model errors and verify that the inference pipeline was working correctly.

## 13. Inference Optimization

Inference was optimized using PyTorch inference mode.

`torch.inference_mode()` was used during prediction because gradients are not required during inference.

Batch processing was also used to improve prediction efficiency.

The optimized inference pipeline was tested successfully.

## 14. Technologies Used

- Python
- PyTorch
- NumPy
- H5Py
- Scikit-learn
- 3D-CNN
- Vision Transformer (ViT)
- CNN + ViT Hybrid Model

## 15. Conclusion

The TerraSpectra ML pipeline successfully implements hyperspectral crop disease classification using deep learning.

The Hybrid CNN + ViT model achieved approximately 58.98% validation accuracy on the prepared dataset.

The model was tested for individual tile prediction, batch inference, prediction validation, model performance, and optimized inference.

The final trained model and ML experiments are maintained in the `m2-m1` branch.