# Continual Learning for Video Game Anomaly Detection

This repository presents a continual learning framework for unsupervised video anomaly detection using Atari gameplay as a testbed. The project combines ConvLSTM-based video modeling with state-of-the-art continual learning techniques to detect diverse anomalies while minimizing catastrophic forgetting across multiple environments.

## Overview

- **Goal:** Detect visual anomalies across multiple Atari games using a continual learning setup.
- **Core Techniques:**
  - ConvLSTM-based temporal modeling
  - Experience replay with importance sampling
  - Elastic Weight Consolidation (EWC)
  - Knowledge distillation between tasks
- **Anomaly Types:**
  - Block
  - Flicker
  - Freeze
  - Freeze Skip
  - Split Horizontal
  - Split Vertical

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/vishalbanwari26/continual-vad.git
cd continual-vad
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare the Dataset

Place the Atari VAD data(download - https://www.kaggle.com/datasets/benedictwilkinsai/atari-anomaly-dataset-aad/data) in the following structure:

```
viper_rl_data/datasets/atari/AAD/
├── clean/
│   ├── BeamRiderNoFrameskip-v4/
│   ├── SpaceInvadersNoFrameskip-v4/
│   └── SeaquestNoFrameskip-v4/
└── anomaly/
    ├── BeamRiderNoFrameskip-v4/
    ├── SpaceInvadersNoFrameskip-v4/
    └── SeaquestNoFrameskip-v4/
```

Each folder should contain `.hdf5` or `.h5` gameplay sequences and a `meta.json` file for anomaly annotations.

### Run the Pipeline

```bash
python main.py
```

This will train the model sequentially on BeamRider, SpaceInvaders, and Seaquest, and evaluate anomaly detection after each stage.

## Results Comparison

### Baseline (No Continual Learning)

| Game       | Block | Flicker | Freeze | Freeze Skip | Split H | Split V |
|------------|-------|---------|--------|--------------|---------|---------|
| BeamRider  | 0.82  | 0.82    | 0.80   | 0.79         | 0.84    | 0.84    |
| SpaceInv   | 0.83  | 0.47    | 0.47   | 0.45         | 0.71    | 0.61    |
| Seaquest   | 0.78  | 0.86    | 0.48   | 0.49         | 0.79    | 0.79    |
| After Seaquest (BeamRider) | 0.79 | 0.22 | 0.35 | 0.37 | 0.71 | 0.69 |

### Continual Learning (Replay + EWC + KD)

After Seaquest - 

| Game       | Block | Flicker | Freeze | Freeze Skip | Split H | Split V |
|------------|-------|---------|--------|--------------|---------|---------|
| BeamRider  | 0.73  | 0.75    | 0.55   | 0.57         | 0.68    | 0.64    |
| SpaceInv   | 0.72  | 0.89    | 0.40   | 0.38         | 0.64    | 0.52    |
| Seaquest   | 0.91  | 0.97    | 0.36   | 0.38         | 0.90    | 0.91    |

These results demonstrate the effectiveness of continual learning in mitigating catastrophic forgetting. Structural anomalies like Block and Split types are consistently detected well, while temporal anomalies like Freeze and Freeze Skip remain more challenging. Flicker detection shows significant improvement under the continual setup, highlighting the benefits of temporal modeling across games.


## Methodology Highlights

- **Reconstruction & Temporal Errors:** Combines MSE reconstruction loss and frame-to-frame motion consistency.
- **Replay Buffer:** Importance sampling based on anomaly score deviation.
- **EWC:** Fisher information matrix scaled by 700, decayed with factor 0.7.
- **Knowledge Distillation:** Feature and output-level alignment with frozen teacher model.

## Future Work

- Scale to more diverse game environments
- Improve detection of motion-based anomalies like freeze/freeze-skip
- Explore transformer-based temporal modeling
- Integrate generative replay for efficient memory usage
- Apply to real-world industrial anomaly datasets

## License

MIT License  
© 2025 Vishal Banwari
