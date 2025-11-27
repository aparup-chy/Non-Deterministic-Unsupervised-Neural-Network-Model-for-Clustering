# 🧠 Non-Deterministic Unsupervised Neural Network Model for Clustering

A comprehensive implementation of a Variational Autoencoder (VAE) for unsupervised clustering of handwritten digits from the MNIST dataset, demonstrating uncertainty quantification and probabilistic latent representations.

**Course:** CSE425 - Neural Networks  
**Instructor:** Mr. Moin Mostakim

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Objectives](#key-objectives)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Evaluation Metrics](#evaluation-metrics)
- [References](#references)

---

## 🎯 Project Overview

This project implements a **Variational Autoencoder (VAE)** to perform unsupervised clustering on the MNIST handwritten digit dataset. Unlike deterministic autoencoders, VAEs introduce stochasticity to the latent space, enabling:

- **Uncertainty quantification** in data representations
- **Robust cluster assignments** across multiple initializations
- **Probabilistic modeling** of the latent space
- **Comparison** with deterministic baseline methods

The model learns meaningful latent representations of handwritten digits without using labels during training, demonstrating the power of unsupervised learning for clustering tasks.

---

## 🔑 Key Objectives

1. **Select**: Unsupervised clustering application using MNIST digits
2. **Design**: Variational Autoencoder neural architecture with probabilistic latent space
3. **Implement**: Train and evaluate VAE on MNIST dataset
4. **Quantify**: 
   - Clustering performance metrics (ARI, NMI, Silhouette)
   - Uncertainty through multi-run evaluation
   - Stability across different random seeds
5. **Compare**: VAE performance vs. deterministic autoencoders

---

## 🔬 Methodology

### Model Architecture: Variational Autoencoder (VAE)

The VAE consists of three main components:

#### 1. **Encoder Network**
- Maps input `x` to latent distribution parameters: `μ(x)` and `log σ²(x)`
- Architecture:
  - Input: 784 dimensions (28×28 flattened MNIST image)
  - Hidden Layer: 400 units with ReLU activation
  - Output: 32-dimensional latent space (mean and log-variance)

#### 2. **Reparameterization Trick**
```
z = μ(x) + σ(x) · ε,  where ε ~ N(0,1)
```
This enables backpropagation through the stochastic sampling process.

#### 3. **Decoder Network**
- Reconstructs image `x̂` from latent code `z`
- Architecture:
  - Input: 32-dimensional latent code
  - Hidden Layer: 400 units with ReLU activation
  - Output: 784 dimensions with Sigmoid activation

### Loss Function

The VAE optimizes the Evidence Lower Bound (ELBO):

```
Loss = Reconstruction Loss + β·KL Divergence
```

Where:
- **Reconstruction Loss**: Binary cross-entropy between input and reconstruction
- **KL Divergence**: Regularization term encouraging latent distribution to match prior N(0,1)
- **β**: Weighting parameter (β = 1.0 in this implementation)

---

## 🏗️ Architecture

### Network Specifications

```python
VAE Architecture:
├── Encoder
│   ├── Linear(784 → 400) + ReLU
│   ├── Linear(400 → 32)  [μ branch]
│   └── Linear(400 → 32)  [log σ² branch]
├── Latent Space (32 dimensions)
└── Decoder
    ├── Linear(32 → 400) + ReLU
    └── Linear(400 → 784) + Sigmoid
```

### Key Features
- **Latent Dimension**: 32
- **Hidden Dimension**: 400
- **Input/Output Dimension**: 784
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output)
- **Optimizer**: Adam with learning rate 1e-3

---

## 📦 Installation & Setup

### Prerequisites
```bash
Python 3.7+
CUDA (optional, for GPU acceleration)
```

### Required Libraries
```bash
pip install torch torchvision datasets scikit-learn matplotlib seaborn numpy
```

### Dataset
The MNIST dataset is automatically downloaded via the Hugging Face `datasets` library:
```python
from datasets import load_dataset
mnist = load_dataset("ylecun/mnist")
```

---

## 🚀 Usage

### 1. Basic Training

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Prepare data
train_data = torch.stack([transform(example['image']) for example in mnist['train']])
dataset = TensorDataset(train_data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Initialize and train model
model = VAE(input_dim=784, latent_dim=32)
history = train_vae(model, dataloader, epochs=10, lr=1e-3, kl_weight=1.0)
```

### 2. Generate Embeddings

```python
model.eval()
with torch.no_grad():
    embeddings = model.encoder(train_data.to(device)).cpu()
```

### 3. Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)
```

### 4. Multi-Run Evaluation

```python
# Evaluate stability across multiple seeds
results, summary = multi_run_evaluate(
    dataset_tensor=train_data,
    labels_tensor=labels,
    runs=3,
    epochs=10,
    batch_size=128
)

print("Multi-run Summary:")
print(f"ARI: {summary['ARI_mean']:.3f} ± {summary['ARI_std']:.3f}")
print(f"NMI: {summary['NMI_mean']:.3f} ± {summary['NMI_std']:.3f}")
```

### 5. Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
            c=cluster_labels, cmap='tab10', s=5)
plt.colorbar()
plt.title("t-SNE Visualization of VAE Latent Space")
plt.show()
```

---

## 📊 Results

### Quantitative Metrics (Mean over 3 runs)

| Metric | Value |
|--------|-------|
| **Silhouette Score** | ~0.42 |
| **Davies-Bouldin Index** | ~0.65 |
| **Calinski-Harabasz Index** | ~247,349 |
| **Adjusted Rand Index (ARI)** | ~0.47-0.48 |
| **Normalized Mutual Info (NMI)** | ~0.55 |

### Multi-Run Stability Analysis

The model demonstrates high stability across different initializations:
- **Low standard deviation** in ARI and NMI scores
- **Consistent cluster assignments** across runs
- **Robust to random initialization**

### Qualitative Results

#### t-SNE Visualization
- Clear separation of digit clusters in 2D projection
- Well-formed cluster boundaries
- Minimal overlap between different digit classes

#### Reconstruction Quality
- Preserved main digit features
- Some stochastic variation (expected from VAE)
- Smooth interpolation in latent space

### Comparison with Baseline
VAE demonstrates superior:
- **Stability**: More consistent results across runs
- **Uncertainty Quantification**: Captures data ambiguity
- **Latent Space Quality**: Better-structured representations

---

## 🔧 Technical Details

### Hyperparameters

```python
HYPERPARAMETERS = {
    'input_dim': 784,           # Flattened 28×28 MNIST images
    'hidden_dim': 400,          # Encoder/Decoder hidden layer size
    'latent_dim': 32,           # Latent space dimensionality
    'batch_size': 128,          # Training batch size
    'learning_rate': 1e-3,      # Adam optimizer learning rate
    'epochs': 10-20,            # Training epochs
    'kl_weight': 1.0,           # KL divergence weight (β)
    'num_clusters': 10          # Number of digit classes
}
```

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy + KL Divergence
- **Device**: GPU (if available) or CPU
- **Random Seeds**: 42, 43, 44 (for multi-run evaluation)

### Data Preprocessing

```python
transform = transforms.Compose([
    transforms.ToTensor(),                    # Convert PIL to Tensor
    transforms.Normalize((0.5,), (0.5,)),     # Normalize to [-1, 1]
    transforms.Lambda(lambda x: x.view(-1))   # Flatten to 784
])
```

---

## 📈 Evaluation Metrics

### Clustering Metrics

1. **Silhouette Score** (0.42)
   - Measures cluster cohesion and separation
   - Range: [-1, 1], higher is better
   - Interpretation: Good cluster structure

2. **Davies-Bouldin Index** (0.65)
   - Ratio of within-cluster to between-cluster distances
   - Lower values indicate better clustering
   - Interpretation: Well-separated clusters

3. **Calinski-Harabasz Index** (247,349)
   - Ratio of between-cluster to within-cluster dispersion
   - Higher values indicate better-defined clusters

4. **Adjusted Rand Index** (0.47-0.48)
   - Measures agreement with true labels
   - Range: [-1, 1], adjusted for chance
   - Interpretation: Moderate agreement

5. **Normalized Mutual Information** (0.55)
   - Information-theoretic measure of clustering quality
   - Range: [0, 1], higher is better

### Uncertainty Quantification

- **Multi-run standard deviations**: Low variance across seeds
- **Sample-wise reconstruction variability**: Consistent with latent noise
- **Probabilistic predictions**: Confidence intervals via sampling

---

## 🎓 Discussion

### Key Findings

1. **Non-deterministic nature** of VAE provides robust latent representations
2. **Consistent clustering** across multiple random initializations
3. **Better uncertainty estimation** compared to deterministic autoencoders
4. **Well-structured latent space** enables meaningful interpolation

### Advantages of VAE Approach

- ✅ **Uncertainty quantification** through probabilistic modeling
- ✅ **Robust to initialization** due to regularized latent space
- ✅ **Interpretable latent representations** via KL divergence
- ✅ **Generative capability** for sampling new data

### Limitations

- ⚠️ Some digit classes show partial overlap (e.g., '3' vs '5')
- ⚠️ Performance sensitive to latent dimension and batch size
- ⚠️ Computational overhead compared to simple autoencoders

---

## 🔮 Future Work

1. **Advanced Architectures**
   - β-VAE for disentangled representations
   - Conditional VAE for controlled generation
   - VQ-VAE for discrete latent spaces

2. **Larger Latent Spaces**
   - Explore higher-dimensional embeddings
   - Compare clustering quality vs. dimension

3. **Improved Clustering Objectives**
   - Deep Embedding for Clustering (DEC)
   - Joint optimization of reconstruction and clustering

4. **Better Uncertainty Metrics**
   - Predictive uncertainty quantification
   - Epistemic vs. aleatoric uncertainty

---

## 📚 References

1. **Kingma, D. P., & Welling, M. (2013)**  
   *Auto-Encoding Variational Bayes*  
   arXiv preprint arXiv:1312.6114  
   [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

2. **Higgins, I., et al. (2016)**  
   *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*  
   International Conference on Learning Representations (ICLR)  
   [https://openreview.net/forum?id=Sy2fzU9gl](https://openreview.net/forum?id=Sy2fzU9gl)

3. **Xie, J., Girshick, R., & Farhadi, A. (2016)**  
   *Unsupervised Deep Embedding for Clustering Analysis*  
   Proceedings of the 33rd International Conference on Machine Learning (ICML)  
   [https://arxiv.org/abs/1511.06335](https://arxiv.org/abs/1511.06335)

---

## 📄 License

This project is submitted as coursework for CSE425 - Neural Networks at BRAC University.

---

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun and collaborators
- **Instructor**: Mr. Moin Mostakim for guidance and support
- **PyTorch Community**: For excellent documentation and tools
- **Hugging Face**: For the `datasets` library

---

## 📧 Contact

**Aparup Chowdhury**  
Student ID: 22101229  
Section: 03  
Course: CSE425 - Neural Networks

For questions or discussions about this project, please contact through the university portal.

---

*This project demonstrates the application of advanced unsupervised learning techniques for discovering structure in high-dimensional data, with emphasis on uncertainty quantification and robust clustering.*
