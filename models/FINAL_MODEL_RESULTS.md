# Final Model Results

## Project
TerraSpectra - Hyperspectral Crop Disease Forecasting

## Final Model
The final model is a Hybrid CNN + Vision Transformer (ViT) architecture.

The CNN component extracts spatial-spectral features from hyperspectral patches, while the Vision Transformer captures global feature relationships.

## Dataset

- Total samples: 78,400
- Patch size: 5 × 5
- Spectral bands: 20
- Number of classes: 4
- Training samples: 62,720
- Validation/Test samples: 15,680

Model input format:

`(N, 1, 20, 5, 5)`

## Final Training

The final Hybrid CNN + ViT model was trained for 5 epochs using the prepared hyperspectral spectral patches.

The trained model was saved as:

`models/final_hybrid_model.pth`

## Final Results

The final model was validated on 15,680 samples.

- Correct predictions: 9,248
- Total predictions: 15,680
- Validation accuracy: 58.98%

Prediction validation result:

`Prediction validation PASSED!`

## Model Comparison

| Model | Approx. Accuracy |
|---|---:|
| 3D-CNN | ~57% |
| Hybrid CNN + ViT | ~58% |

The Hybrid CNN + ViT model performed slightly better than the standalone 3D-CNN.

## Inference Verification

The model was tested for:

- Individual tile prediction
- Batch inference
- Tile prediction validation
- Model performance and error testing
- Optimized inference

All implemented inference tests completed successfully.

## Final Verification

The final model file was saved successfully and the prediction pipeline was verified using the validation dataset.

The number of predictions matched the number of validation samples.

## Conclusion

The TerraSpectra machine learning pipeline successfully trained and validated a Hybrid CNN + ViT model for hyperspectral crop disease classification.

The final model and results are maintained in the `m2-m1` branch.