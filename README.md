# Enhancing land use and land cover classification through comparative analysis of deep learning architectures <br>
Our study focuses on Land Use Classification by: <br>
- Training, custom-built CNNs from scratch <br>
- Applying  transfer learning to fine-tune pre-trained networks such as AlexNet and ResNet for Land Use Classification <br>
- Extensive analysis, such as, number of misclassified images per label, prediction accuracy per label, showing misclassified images for each label, etc.

## Authors
- [Md. Sakib Bin Alam](https://github.com/SakibBinAlam)
- Anjana
- [Aiman Lameesa](https://github.com/aimanlameesa)


### Dataset
- **Source**: [EuroSAT](https://github.com/phelber/EuroSAT)
- **Total images**: 27,000 geo-referenced Sentinel-2 satellite images
- **Image size**: 64 × 64 pixels, 3-channel RGB
- **Number of classes**: 10 land use / land cover (LULC) categories:
  AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

### Preprocessing Pipeline
1. **Loading**: Images are loaded via `torchvision.datasets.ImageFolder` from the extracted EuroSAT dataset directory.
2. **Transforms** (applied in order):
   - `transforms.ToTensor()` — converts PIL images to PyTorch tensors and scales pixel values from [0, 255] to [0.0, 1.0].
   - `transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))` — normalises each channel to the range [−1.0, 1.0].

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Maximum epochs | 20 |

### Early Stopping Criterion
Training employs early stopping to prevent overfitting:
- **Monitored metric**: average validation loss per epoch
- **Improvement threshold (min_delta)**: **0.001** — the validation loss must decrease by at least 0.001 relative to the best observed validation loss to be considered an improvement.
- **Patience**: **5** — training is halted if no improvement (exceeding the threshold) is observed for 5 consecutive epochs.

### Model Architectures
Three custom CNN architectures of increasing depth are evaluated:

**Model 3 — deeperEuroCNN**
- Conv2d(3 → 64, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2×2)
- Conv2d(64 → 128, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2×2)
- Conv2d(128 → 256, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2×2)
- Flatten → Dropout(0.2) → Linear(256×6×6 → 200) → ReLU → Dropout(0.2) → Linear(200 → 100) → ReLU → Dropout(0.2) → Linear(100 → 75) → ReLU → Dropout(0.2) → Linear(75 → 10)

### Transfer Learning Models

#### Common Configuration

| Parameter | Value |
|-----------|-------|
| Data split | 70% train / 20% validation / 10% test (pre-split) |
| Image size | 224 × 224 (resized) |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Maximum epochs | 20 |


### Software Environment
- Python 3.x
- PyTorch (with CUDA support)
- torchvision
- matplotlib, seaborn, pandas, numpy
- GPU: NVIDIA RTX A6000

## References
[1]  S. Basu, S. Ganguly, S. Mukhopadhyay, R. DiBiano, M. Karki, and R. Nemani, “Deepsat: a learning framework for satellite 
      imagery,” in Proceedings of the 23rd SIGSPATIAL international conference on advances in geographic information systems, 2015, 
      pp. 1–10.

[2] R. Naushad, T. Kaur, and E. Ghaderpour, “Deep transfer learning for land use and land cover classification: A comparative study,” 
     Sensors, vol. 21, no. 23, p. 8083, 2021.

[3] P. Kowaleczko et al., “MuS2: A Benchmark for Sentinel-2 Multi-Image Super-Resolution,” arXiv preprint arXiv:2210.02745, 2022.
