# Enhancing land use and land cover classification through comparative analysis of deep learning architectures <br>
This mansucript is curreenlty under review in the journal Ecological Informatics. <br>

Our study focuses on Land Use Classification by: <br>
- Training, custom-built CNNs from scratch <br>
- Applying  transfer learning to fine-tune pre-trained networks such as AlexNet and ResNet for Land Use Classification <br>
- Extensive analysis, such as, number of misclassified images per label, prediction accuracy per label, showing misclassified images for each label, etc.

This repository contains the part of CNN implementation. Transfer learning implementation for our project is not uploaded yet.

## Dataset
https://github.com/phelber/EuroSAT

## Authors
- [Md. Sakib Bin Alam](https://github.com/SakibBinAlam)
- Anjana
- [Aiman Lameesa](https://github.com/aimanlameesa)

## Reproducibility Documentation

### Random Seed
All experiments use a **global random seed of 42** to ensure reproducibility. The seed is set across all relevant libraries and backends:
- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed(42)` and `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `os.environ['PYTHONHASHSEED'] = '42'`

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
3. **Data splitting** (using `torch.utils.data.random_split`):
   - First split: 80% training (21,600 images) / 20% test (5,400 images)
   - Second split (on training set): 80% training (17,280 images) / 20% validation (4,320 images)
   - Effective split: **64% train / 16% validation / 20% test**

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Maximum epochs | 20 |
| DataLoader workers | 2 |
| Pin memory | True |
| Shuffle (train/val) | True |

### Early Stopping Criterion
Training employs early stopping to prevent overfitting:
- **Monitored metric**: average validation loss per epoch
- **Improvement threshold (min_delta)**: **0.001** — the validation loss must decrease by at least 0.001 relative to the best observed validation loss to be considered an improvement.
- **Patience**: **5** — training is halted if no improvement (exceeding the threshold) is observed for 5 consecutive epochs.
- **Best model checkpointing**: the model state with the lowest validation loss is saved to disk (`bestmodel1.pt`).

### Model Architectures
Three custom CNN architectures of increasing depth are evaluated:

**Model 1 — EuroCNN (Base)**
- Conv2d(3 → 128, kernel 3×3) → ReLU → MaxPool2d(2×2) → Flatten → Dropout(0.2) → Linear(128×31×31 → 10)

**Model 2 — deepEuroCNN**
- Conv2d(3 → 128, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2×2)
- Conv2d(128 → 256, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2×2)
- Flatten → Linear(256×14×14 → 100) → ReLU → Dropout(0.2) → Linear(100 → 50) → ReLU → Dropout(0.2) → Linear(50 → 10)

**Model 3 — deeperEuroCNN**
- Conv2d(3 → 64, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2×2)
- Conv2d(64 → 128, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2×2)
- Conv2d(128 → 256, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2×2)
- Flatten → Dropout(0.2) → Linear(256×6×6 → 200) → ReLU → Dropout(0.2) → Linear(200 → 100) → ReLU → Dropout(0.2) → Linear(100 → 75) → ReLU → Dropout(0.2) → Linear(75 → 10)

### Transfer Learning Models (Best Configurations)

The transfer learning experiments use a **separate, pre-split** version of the EuroSAT dataset and are trained on Google Colab (NVIDIA A100-SXM4-40GB, CUDA 11.2).

#### Common Configuration

| Parameter | Value |
|-----------|-------|
| Data split | 70% train / 20% validation / 10% test (pre-split) |
| Image size | 224 × 224 (resized) |
| Normalisation | ImageNet statistics — mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] |
| Batch size | 64 (training/validation), 4 (test) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Maximum epochs | 20 |

**Training transforms** (applied in order):
1. `Resize(224, 224)` → `RandomHorizontalFlip()` → `RandomRotation(20)` → `ToTensor()` → `Normalize(mean, std)`

**Validation/Test transforms**:
1. `Resize(224, 224)` → `ToTensor()` → `Normalize(mean, std)`

#### AlexNet Model 3 (Version 3)

- **Base model**: `torchvision.models.alexnet` with pre-trained weights (`AlexNet_Weights.IMAGENET1K_V1`)
- **Fine-tuning strategy**: All parameters trainable except BatchNorm2d layers (frozen)
- **Custom classifier head** (replacing the default AlexNet classifier):
  - Dropout(0.5) → Linear(9216 → 4096) → ReLU
  - Dropout(0.5) → Linear(4096 → 2048) → ReLU
  - Linear(2048 → 1024) → ReLU
  - Dropout(0.5) → Linear(1024 → 512) → ReLU
  - Dropout(0.5) → Linear(512 → 256) → ReLU
  - Dropout(0.5) → Linear(256 → 10)
- **Total trainable parameters**: 51,370,058

#### ResNet50 Model 3 (Version 3)

- **Base model**: `torchvision.models.resnet50` with pre-trained weights (`ResNet50_Weights.IMAGENET1K_V2`)
- **Fine-tuning strategy**: All parameters trainable except BatchNorm2d layers (frozen)
- **Custom FC head** (replacing `model.fc`):
  - Linear(2048 → 512) → ReLU → Dropout(0.25)
  - Linear(512 → 256) → ReLU → Dropout(0.5)
  - Linear(256 → 10)
- **Total trainable parameters**: 17,689,994

### Software Environment
- Python 3.x
- PyTorch (with CUDA support)
- torchvision
- matplotlib, seaborn, pandas, numpy
- GPU: NVIDIA RTX A6000 (CUDA 12.2, Driver 535.183.01)

## References
[1]  S. Basu, S. Ganguly, S. Mukhopadhyay, R. DiBiano, M. Karki, and R. Nemani, “Deepsat: a learning framework for satellite 
      imagery,” in Proceedings of the 23rd SIGSPATIAL international conference on advances in geographic information systems, 2015, 
      pp. 1–10.

[2] R. Naushad, T. Kaur, and E. Ghaderpour, “Deep transfer learning for land use and land cover classification: A comparative study,” 
     Sensors, vol. 21, no. 23, p. 8083, 2021.

[3] P. Kowaleczko et al., “MuS2: A Benchmark for Sentinel-2 Multi-Image Super-Resolution,” arXiv preprint arXiv:2210.02745, 2022.
