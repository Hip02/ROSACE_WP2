# Report: Code Organization and Future Directions

## Part 1: Code Organization Upgrades

The current codebase has good foundations but can be improved for scalability, reproducibility, and ease of maintenance.

### 1. Training Loop Abstraction
**Current State:** The training loop is manually implemented in the `Network` class (`DeepLearning/networks/model.py`). This mixes model definition, training logic, validation, and visualization.
**Suggestion:** Adopt a framework like **PyTorch Lightning** or **Hugging Face Accelerate**.
- **Why?** It standardizes the training loop, handles device placement (CPU/GPU/MPS) automatically, simplifies checkpointing, and makes multi-GPU training trivial.
- **Action:** Refactor `Network` into a `LightningModule`. Move the training step, validation step, and optimizer configuration into specific methods.

### 2. Configuration Management
**Current State:** You have moved to YAML files, which is excellent. However, `main.py` manually parses this and initializes classes.
**Suggestion:** Use **Hydra** or **OmegaConf** for more advanced configuration.
- **Why?** Hydra allows for hierarchical configuration, overriding parameters from the command line easily (e.g., `python main.py model.lr=0.01`), and automatic logging of the configuration used for each run.
- **Action:** Replace `argparse` + `yaml` with a Hydra decorator in `main.py`.

### 3. Experiment Tracking
**Current State:** Loss curves are saved as PNGs using `matplotlib`. Metrics are printed to stdout.
**Suggestion:** Integrate **Weights & Biases (wandb)** or **TensorBoard**.
- **Why?** These tools provide interactive dashboards to visualize loss curves, compare experiments, view generated images/videos in real-time, and track hyperparameters.
- **Action:** Add `wandb.init()` in your training script and `wandb.log({"loss": loss})` in the training loop.

### 4. Data Pipeline
**Current State:** `LascoC2ImagesDataset` handles logic like temporal neighbor search and specific filename parsing.
**Suggestion:** Decouple data preparation from the dataset class.
- **Why?** The dataset class is complex. Pre-calculating neighbor indices or creating a manifest file (CSV/JSON) listing `{image_t, image_t-1, image_t+1, label}` during a preprocessing step would make the Dataset class much simpler and faster.
- **Action:** Create a `prepare_dataset.py` script that generates a "manifest" CSV file with all file paths and targets pre-computed. The Dataset then just reads this row-by-row.

### 5. Testing
**Current State:** Custom `sanity_checkers` scripts.
**Suggestion:** Implement unit tests using `pytest`.
- **Why?** Automated testing ensures that refactoring (like the one we just did) doesn't break existing functionality.
- **Action:** Create a `tests/` folder. Write tests for:
    - `Dataset.__getitem__` (check shapes, types).
    - `Model.forward` (check input/output shapes).
    - `Loss.forward` (check if it returns a scalar and gradients flow).

## Part 2: New Project Ideas

Based on the solar physics context (CME detection in LASCO C2 images), here are some advanced ideas:

### 1. Physics-Informed Deep Learning
- **Idea:** Incorporate physical constraints into the loss function beyond simple segmentation.
- **Details:** If you have access to mass or kinetic energy estimates (which seem to be in your CSV), add a regression head to predict these physical quantities. Constrain the segmentation mask to be consistent with the estimated mass (e.g., sum of pixel intensities $\propto$ mass).

### 2. Temporal Models (Video/Sequence Processing)
- **Idea:** Move from "Stack of 5 frames" to a true sequence model.
- **Details:**
    - **ConvLSTM / ConvGRU:** Replace the UNet encoder with a Recurrent Convolutional Network to explicitly model the time evolution of the CME.
    - **Video Swin Transformer:** Use state-of-the-art 3D Transformer backbones designed for video understanding.
- **Benefit:** Better handling of CME kinematics and noise suppression (stars/comets move differently than CMEs).

### 3. Multi-View / Multi-Instrument Fusion
- **Idea:** If you can obtain data from other viewpoints (e.g., STEREO A/B) or other instruments (e.g., LASCO C3), fuse this data.
- **Details:** Train a model that takes simultaneous observations to reconstruct the 3D structure of the CME (e.g., Neural Radiance Fields - NeRF for Solar atmosphere).

### 4. Self-Supervised Pre-training
- **Idea:** Labeled CME data might be scarce or noisy.
- **Details:** Train an encoder (like a Masked Autoencoder - MAE) on *all* available LASCO images (years of data) to reconstruct masked out patches. Then fine-tune this pre-trained encoder on your labeled CME dataset.
- **Benefit:** The model learns "what the sun/corona looks like" from massive unlabeled data, likely improving segmentation performance on the smaller labeled set.

### 5. Uncertainty Estimation
- **Idea:** Predict not just the mask, but the uncertainty of the prediction.
- **Details:** Use Bayesian Neural Networks (e.g., Monte Carlo Dropout) or an ensemble of models.
- **Benefit:** Crucial for space weather forecasting to know when the model is unsure (e.g., during complex events or high noise).

### 6. Real-time Event Detection on Edge
- **Idea:** Optimize the model for low-latency inference.
- **Details:** Use techniques like Knowledge Distillation or quantization (to int8) to make the model run fast on smaller hardware.
- **Goal:** A continuously running "alert system" that processes the latest available image immediately.
