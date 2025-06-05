# FlavourClassification

```c++
████████╗██████╗  █████╗ ███╗   ██╗███████╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗██████╗       
╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔══██╗      
   ██║   ██████╔╝███████║██╔██╗ ██║███████╗█████╗  ██║   ██║██████╔╝██╔████╔██║█████╗  ██████╔╝      
   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ██╔══██╗      
   ██║   ██║  ██║██║  ██║██║ ╚████║███████║██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗██║  ██║      
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝      
                                                                                                     
███████╗ ██████╗ ██████╗                                                                             
██╔════╝██╔═══██╗██╔══██╗                                                                            
█████╗  ██║   ██║██████╔╝                                                                            
██╔══╝  ██║   ██║██╔══██╗                                                                            
██║     ╚██████╔╝██║  ██║                                                                            
╚═╝      ╚═════╝ ╚═╝  ╚═╝                                                                            
                                                                                                     
███╗   ██╗███████╗██╗   ██╗████████╗██████╗ ██╗███╗   ██╗ ██████╗                                    
████╗  ██║██╔════╝██║   ██║╚══██╔══╝██╔══██╗██║████╗  ██║██╔═══██╗                                   
██╔██╗ ██║█████╗  ██║   ██║   ██║   ██████╔╝██║██╔██╗ ██║██║   ██║                                   
██║╚██╗██║██╔══╝  ██║   ██║   ██║   ██╔══██╗██║██║╚██╗██║██║   ██║                                   
██║ ╚████║███████╗╚██████╔╝   ██║   ██║  ██║██║██║ ╚████║╚██████╔╝                                   
╚═╝  ╚═══╝╚══════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝                                    
                                                                                                     
███████╗██╗      █████╗ ██╗   ██╗ ██████╗ ██╗   ██╗██████╗                                           
██╔════╝██║     ██╔══██╗██║   ██║██╔═══██╗██║   ██║██╔══██╗                                          
█████╗  ██║     ███████║██║   ██║██║   ██║██║   ██║██████╔╝                                          
██╔══╝  ██║     ██╔══██║╚██╗ ██╔╝██║   ██║██║   ██║██╔══██╗                                          
██║     ███████╗██║  ██║ ╚████╔╝ ╚██████╔╝╚██████╔╝██║  ██║                                          
╚═╝     ╚══════╝╚═╝  ╚═╝  ╚═══╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝                                          
                                                                                                     
 ██████╗██╗      █████╗ ███████╗███████╗██╗███████╗██╗ ██████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔════╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
██║     ██║     ███████║███████╗███████╗██║█████╗  ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
██║     ██║     ██╔══██║╚════██║╚════██║██║██╔══╝  ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
╚██████╗███████╗██║  ██║███████║███████║██║██║     ██║╚██████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
 ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                                                                                     
```
A modular PyTorch framework for classifying neutrino flavours from IceCube simulation data using Transformer encoder models.

> ### For the data extraction and transformation process, see the [IcePack](https://github.com/KUcyans/IcePack/tree/main)

## 🗂 Directory Overview

### `VernaDataSocket/` (DataSockets)

Handles dataset logic and preparation.

- **MonoFlavourDataset.py** – Loads individual flavour datasets.
- **MultiFlavourDataset.py** – Merges multiple MonoFlavour datasets with optional noise.
- **MultiFlavourDataModule.py** – PyTorch Lightning `LightningDataModule` to prepare loaders for training/validation.
- **NoiseDataset.py** – Generates or loads noise-only data.
- **PseudoNormaliser.py** – Applies feature scaling or pseudo-normalisation strategies.

---

### `Model/`

Defines the Transformer encoder and its building blocks.

- **FlavourClassificationTransformerEncoder.py** – Main model class implementing flavour classification logic.
- **EncoderBlock.py** – Defines a single Transformer encoder block.
- **BuildingBlocks/** – Core attention and projection layers:
  - `ALiBiAttention.py`, `T5Attention.py`, `XFormersAttention.py`, `InnocentAttention.py` – Variants of attention mechanisms.
  - `MultiHeadAttention.py`, `ScaledDotProductAttention.py` – Base attention formulations.
  - `FFN.py`, `OutputProjection.py`, `Pooling.py`, `LayerNormalisation.py` – Standard Transformer layers.

---

### `Enum/`

Defines configuration enums used throughout training and inference.

- **AttentionType.py** – Enum for choosing the attention mechanism.
- **ClassificationMode.py** – Enum to toggle between different output modes (e.g., νₑ, ν_μ, ν_τ only).
- **EnergyRange.py** – Bin categories based on event energy.
- **Flavour.py** – Flavour labels and representations.
- **LossType.py** – Supported loss functions (e.g., CE, focal loss).
- **LrDecayMode.py** – Learning rate schedulers.
- **PositionalEncodingType.py** – Positional encoding strategies (e.g., sinusoidal, rotary).

---

### `TrainingUtils/`

Custom utilities to assist training.

- **EquinoxDecayingAsymmetricSinusoidal.py** – Exotic LR decay scheduler.
- **KatsuraCosineAnnealingWarmupRestarts.py** – Warmup+Cosine LR scheduler.
- **LocalMinimumCheckpoint.py**, **MidEpochCheckPoint.py** – Callback extensions for smarter checkpointing.

---

### `config/`

- **config.json** – Specifies all hyperparameters, model settings, training options, and data paths. Used by both `train.py` and `predict.py`.

---

### Root Scripts

- **train.py** – Launches training using the model, dataset, and config.
- **predict.py** – Runs inference on a dataset using a trained model checkpoint.
- **InferenceUtil.py** – Utilities for performing predictions and aggregating outputs.

---



---

## 🚀 Quick Start

### Training

```bash
python train.py --config config/config.json
```

### Inference

```bash
python predict.py --checkpoint path/to/model.ckpt --config config/config.json
```

---
## License

[![Apache Licensed](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)  

---
You find this project useful?
Please consider giving it a star on GitHub!
and cite it in your publications as:

```bibtex
@software{,
  title = {Transformer for Flavour Classification},
  author = {Cyan Jo},
  url = {https://github.com/KUcyans/FlavourClassification},
  version = {0.1.0},
  date = {2025-06-05},
  license = {Apache-2.0},
  type = {software},s
}
```