# Cat Vs Dog Classifier

A deep learning project to classify images of cats and dogs using Convolutional Neural Networks (CNNs) models and compare the performance of different architectures(Alexnet, VGG16, ResNet18) on the binary classification task.

# Project Structure

CatVsDog/
│
├── data/                  # Dataset: train, test, val
│   ├── train/
│   ├── test/
│   └── val/
│
├── models/                # Model architectures
│   ├── alexnet.py
│   ├── vgg16.py
│   └── resnet.py
│
├── notebooks/             # Jupyter notebooks for EDA, experiments and analysis
│   ├── CatVsDog.ipynb
│   └── analysis.ipynb     # Analysis of results with accuracy and loss curves and their conclusion
│
├── results/               # Training logs, saved models, TensorBoard logs
│   ├── alexnet/
│   ├── resnet18/
│   ├── vgg16/
│   └── runs/
│
├── utils/                 # Helper scripts
│   ├── dataset.py
│   └── train_eval.py
│
├── main.py                # Entry point for training and evaluation
└── README.md              # Project documentation

# Models Used
- AlexNet
- VGG16
- ResNet18
All models are implemented from scratch or modified using torchvision.models

# Dataset
The dataset is a binary classification problem where we have images of cats and dogs. The dataset is split into train, test, and validation sets. It is available in .jpg format in the `data/` directory.

# Training and Evaluation
Training and evaluation are handled via train_eval.py .
Each model's performance is logged in results/ .

# Metrics:
- Accuracy
- Loss

# Visualization
Plots for:
- Training and validation accuracy
- Training and validation loss
See notebooks/analysis.ipynb for full insights

## How to Run
 1. **Clone the repo**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd CatVsDog
```
 2. **Create virtual environment**
    ```bash
    pythom -m venv .venv
    .venv\Scripts\activate    #Windows
```
 3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
```
 4. **Run the pipeline**
    python main.py -m model_name -e epochs -bs batch_size -lr learning_rate -e early_stopping

    # Example: 
    ```bash
    python main.py -m resnet18 -e 15 -bs 16 -lr 0.0001 -e 3
   ```
# Requirements
- Python 3.12
- PyTorch 
- Torchvision
- Matplotlib
- tensorboard

# TensorBoard: for tracking training and validation metrics
  Launch with:
  ```bash
  tensorboard --logdir=results/runs
  ```
  Open in browser: http://localhost:6006/
    
# Hyperparameters
- Batch size: 16
- Learning rate: 0.0001
- Epochs: 25
- early stopping patience: 3

# Performance Comparision Table
| Model     | Accuracy | Parameters | Epochs | Time per Epoch |
|-----------|----------|------------|--------|----------------|
| AlexNet   | 89.5%    | 61M        | 10     | 3.5 min        |
| VGG16     | 90.1%    | 138M       | 15     | 5 min          |
| ResNet18  | 92.5%    | 11M        | 7      | 3.2 min        |

# Future Work
- Investigate the effect of different optimizers (e.g., Adam, SGD) on model performance
- Explore the impact of data augmentation on model accuracy
- Consider using transfer learning for improved performance
- Build a web interface for users to input their own images and receive predictions
