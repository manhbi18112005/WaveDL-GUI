# WaveDL: A Scalable Deep Learning Framework for Guided Wave Inversion on High-Performance Computing Clusters

**Ductho Le**

University of Alberta, Edmonton, Alberta, Canada

Email: ductho.le@outlook.com

---

## Abstract

Ultrasonic wave inspection is a powerful technique for non-destructive evaluation (NDE) of structures such as pipelines, aircraft components, and composite materials, yet a central challenge lies in solving the inverse problem of inferring material and structural properties (e.g., thickness, elastic moduli, crack location) from measured wavefields. Traditional approaches rely on iterative optimization with physic-based forward simulations, which are computationally intensive and difficult to deploy in real time. Deep learning provides a promising alternative by learning direct mappings from signal representations to physical parameters; however, large-scale wave-learning problems introduce substantial engineering challenges—training datasets often exceed available system memory and require specialized out-of-core pipelines, multi-GPU training demands careful synchronization to maintain numerical stability and efficiency, and the field lacks standardized frameworks for reproducibility and fair benchmarking across architectures and datasets. To address these limitations, this paper introduces WaveDL, an open-source Python framework that provides a memory-efficient data pipeline using memory-mapped files, an architecture-agnostic interface for integrating and comparing neural networks, robust distributed training with synchronized multi-GPU execution, and automated ONNX export for deployment into industrial ecosystems such as LabVIEW, MATLAB, and C++. Built on PyTorch and Hugging Face Accelerate, WaveDL supports mixed-precision training, experiment tracking with Weights & Biases, and includes over 200 unit tests to ensure reliability. WaveDL is released under the MIT license at https://github.com/ductho-le/WaveDL.

**Keywords:** deep learning framework, guided wave inversion, distributed training, high-performance computing, memory-mapped data, multi-GPU synchronization, neural network regression

---

## Program Summary

**Program Title:** WaveDL

**Developer's repository link:** https://github.com/ductho-le/WaveDL

**Licensing provisions:** MIT License

**Programming language:** Python 3.11+

**Nature of problem:**
Ultrasonic guided wave inversion requires determining material properties (thickness, elastic moduli, density) from measured wave signals. Traditional iterative optimization methods coupled with physics-based forward models are computationally expensive. Deep learning enables direct mapping from signals to properties with millisecond inference time. However, training neural networks on the large-scale simulation datasets typical in this domain presents significant computational challenges: datasets often exceed available RAM, multi-GPU training requires careful synchronization to avoid deadlocks, and there is a lack of standardized frameworks for reproducibility and fair benchmarking across architectures.

**Solution method:**
WaveDL implements a zero-copy memory-mapped data pipeline using NumPy memmap, enabling random-access training on datasets larger than available RAM without performance degradation. The framework automatically detects and handles 1D, 2D, and 3D input data, adding channel dimensions as needed. A decorator-based model registry pattern allows researchers to integrate arbitrary neural network architectures without modifying the training infrastructure. Distributed Data Parallel (DDP) utilities provide synchronized early stopping across multiple GPUs. Additionally, WaveDL includes automated ONNX model export to facilitate deployment into industrial ecosystems such as LabVIEW, MATLAB, and C++.

**External routines/libraries:**
PyTorch (≥2.0), HuggingFace Accelerate (≥0.20), NumPy (≥1.24), SciPy (≥1.10), scikit-learn (≥1.2), pandas (≥2.0), matplotlib (≥3.7), tqdm (≥4.65), Weights & Biases (≥0.15, optional)

**Restrictions:**
The framework is designed for multi-output regression tasks. Classification tasks require modifications to the loss functions. Training requires NVIDIA GPU with CUDA support; CPU-only inference is supported.

---

## 1. Introduction

Ultrasonic guided waves are elastic wave modes that propagate within bounded media such as plates, pipes, shells, and layered composites. Unlike conventional bulk ultrasonic waves that travel through the material thickness, guided waves can propagate over extended distances while remaining sensitive to structural features throughout the waveguide cross-section [1, 2]. This unique property makes them invaluable for the rapid inspection of large-scale infrastructure including pipelines, railway tracks, aircraft fuselages, and wind turbine blades.

The physics of guided wave propagation is governed by the elastic wave equation in bounded media. For an isotropic plate of thickness $h$, the displacement field $\mathbf{u}(\mathbf{x}, t)$ satisfies:

$$
\rho \frac{\partial^2 \mathbf{u}}{\partial t^2} = (\lambda + \mu) \nabla(\nabla \cdot \mathbf{u}) + \mu \nabla^2 \mathbf{u}
$$

where $\rho$ is the mass density, and $\lambda$, $\mu$ are the Lamé constants related to the longitudinal and shear wave velocities. The boundary conditions imposed by the traction-free surfaces give rise to a family of discrete wave modes whose phase velocity $c_p$ depends on the frequency-thickness product. This relationship, known as the *dispersion curve*, encodes the material's mechanical properties and geometry.

The fundamental challenge in guided wave-based material characterization is the *inverse problem*: given observed dispersion data $\mathbf{x}$, determine the material property vector $\mathbf{p} = [h, c_L, c_T, \rho, ...]^T$ that produced it. Traditional solution approaches include iterative optimization, which requires repeated forward model evaluations (seconds to minutes each for complex media), and look-up tables, which scale poorly with parameter dimensionality.

Deep neural networks offer a fundamentally different approach: learn the inverse mapping directly from paired examples. Once trained, the network provides inference latency of milliseconds on GPU hardware. This paradigm shift has been demonstrated across diverse physics domains including seismic inversion [3], electromagnetic inverse scattering [4], and ultrasonic NDE [5-7].

However, the transition from proof-of-concept studies to production-ready research tools remains challenging due to three fundamental computational barriers. First, accurate surrogate models require training on vast synthetic datasets—a 100,000-sample dataset of 500×500 images consumes approximately 100 GB, exceeding the RAM of standard GPU nodes. Second, leveraging multi-GPU clusters introduces synchronization challenges: early stopping decisions must be coordinated to prevent deadlocks, and metric aggregation must account for varying batch sizes across processes. Third, general-purpose frameworks report normalized loss values, which are meaningless to domain scientists who need errors expressed in physical units.

This paper introduces WaveDL, a framework that addresses these gaps through five principal contributions:

1. **Zero-Copy Memory-Mapped Data Pipeline**: Thread-safe data loading that enables training on datasets exceeding available RAM by leveraging OS virtual memory management.

2. **Dimension-Agnostic Data Handling**: Automatic detection and processing of 1D, 2D, and 3D input representations without manual reshaping.

3. **Decorator-Based Model Registry**: Compile-time dependency injection allowing researchers to register arbitrary neural network architectures without modifying the training loop.

4. **DDP-Safe Synchronization Primitives**: Utility functions that broadcast early stopping decisions, aggregate metrics across processes, and coordinate checkpoint saving.

5. **Physics-Aware Metric Tracking**: Automatic computation of Mean Absolute Error in original physical units (mm, m/s, GPa) through inverse standardization.

The framework also includes production-ready features: mixed-precision (BF16/FP16) support, PyTorch 2.x compilation compatibility, Weights & Biases experiment tracking, robust checkpoint/resume functionality, and a comprehensive unit test suite with over 200 tests.

---

## 2. Related Work

The application of deep learning to guided wave problems has accelerated rapidly since 2018. Rautela and Gopalakrishnan [5] provided a comprehensive review of Lamb wave-based damage detection using CNNs and recurrent networks, highlighting that 2D time-frequency representations consistently outperform raw 1D waveforms as network inputs. Miorelli et al. [6] demonstrated supervised deep learning for ultrasonic crack characterization using synthetically generated training data, achieving sizing accuracy comparable to expert human inspectors. Zhang et al. [7] integrated physics-based loss terms derived from Lamb wave dispersion equations, improving generalization to out-of-distribution samples.

Despite these advances, most published studies rely on custom, single-use training scripts that lack generalizability and documentation. Several general-purpose frameworks exist but do not address the specific requirements of guided wave research. PyTorch Lightning [8] abstracts boilerplate code but imposes an opinionated project structure and lacks domain-specific utilities. MONAI [9] is highly successful for medical imaging but not applicable to wave propagation problems. Hugging Face Accelerate [10] provides lightweight distributed training utilities without imposing structural constraints—WaveDL builds upon Accelerate while adding physics-aware features specific to wave inversion.

To date, no framework has been designed specifically for the computational requirements of guided wave inverse problems: out-of-core data loading, physics-aware metrics, dimension-agnostic data handling, and HPC-ready distributed synchronization.

---

## 3. Framework Design

WaveDL follows a modular layered architecture that decouples physical modeling from computational infrastructure. The Data Layer handles memory-mapped I/O and standardization; the Model Layer implements the registry pattern with an abstract base class; the Training Layer provides Accelerate-based distributed training; and the Utility Layer contains metrics, visualization, and distributed primitives. This separation enables independent testing and replacement of components.

### Memory-Efficient Data Pipeline

For typical guided wave applications with $N=10^5$ samples and $H=W=500$ pixels, the memory footprint reaches approximately 100 GB—exceeding the RAM of standard GPU nodes. Memory mapping creates a virtual address space that mirrors the file on disk, with the operating system's virtual memory manager handling page-level access through page faults, LRU eviction, and read-ahead prefetching.

This mechanism provides $O(1)$ memory complexity independent of dataset size and amortized $O(1)$ access time per sample due to caching. The implementation uses lazy initialization of file handles to ensure each DataLoader worker process has its own file descriptor, preventing the race conditions that occur when handles are shared across forked processes. The critical `.copy()` operation detaches returned tensors from the memory-mapped buffer, preventing memory leaks and race conditions during data augmentation.

WaveDL automatically detects input dimensionality and adds channel dimensions as needed. Raw 1D signals of shape $(N, L)$ become $(N, 1, L)$; 2D representations of shape $(N, H, W)$ become $(N, 1, H, W)$; and 3D volumes of shape $(N, D, H, W)$ become $(N, 1, D, H, W)$. This dimension-agnostic handling eliminates the need for manual data preprocessing.

### Distributed Data Parallel Synchronization

In DDP training, each GPU runs an independent copy of the training loop, with PyTorch synchronizing gradients via all-reduce operations during the backward pass. However, control flow decisions such as early stopping are not automatically synchronized. If the main process decides to stop while other processes continue, the program hangs when other processes' forward calls wait indefinitely for gradient synchronization.

WaveDL implements a broadcast-based synchronization protocol using an all-reduce (MAX) operation to compute the global stopping decision. Since all-reduce is a collective operation, all ranks must participate, ensuring synchronized termination. The framework also correctly aggregates metrics across GPUs by computing (sum-of-losses / sum-of-counts) rather than naïvely averaging per-GPU averages, which produces incorrect results when batch sizes differ.

### Extensible Model Registry

The decorator-based Factory Pattern decouples model definition from training logic, enables dynamic model selection via command-line arguments, and prevents circular imports. Models inherit from an abstract `BaseModel` class that enforces a consistent interface: constructors accept input shape and output size, the forward method maps input to output, and optional methods provide parameter summaries and differential learning rate groups for fine-tuning.

To add a new architecture, users create a model file, apply the `@register_model("name")` decorator, and import the model in the package's `__init__.py`. The model is then immediately available for training via the `--model` command-line argument. This same pattern applies to any PyTorch architecture including ResNet, EfficientNet, or Vision Transformers.

### Physics-Aware Metrics

Neural network training benefits from standardized targets with zero mean and unit variance. The standardization parameters are computed on the training set only to prevent data leakage, and the fitted scaler is saved with the checkpoint for use during inference. WaveDL automatically converts metrics to physical units by applying inverse standardization, reporting errors individually per target (e.g., thickness: $0.042 \pm 0.012$ mm, velocity: $3.1 \pm 0.8$ m/s). This enables direct comparison against engineering tolerances.

---

## 4. Reference Architectures

WaveDL ships with two reference architectures and is designed for seamless integration of arbitrary models.

**SimpleCNN** (`cnn`) provides a straightforward encoder-decoder architecture with five convolutional blocks using GroupNorm for stable training with small per-GPU batch sizes, LeakyReLU to prevent dead neurons, and max pooling for spatial reduction. The regression head uses three fully-connected layers with LayerNorm and Dropout for regularization.

**RateNet** (`ratenet`) extends the convolutional backbone with Convolutional Block Attention Modules (CBAM) [11] that apply channel and spatial attention sequentially. This attention mechanism helps the network focus on the most discriminative regions of dispersion curves or spectrograms.

Both architectures serve as starting points. Users can integrate pre-trained models from torchvision by modifying the input layer for single-channel images and replacing the classification head with a regression head, then registering via the decorator pattern.

The training procedure follows standard supervised learning with HPC-specific adaptations: AdamW optimization with gradient clipping, ReduceLROnPlateau scheduling, DDP-safe metric aggregation, and synchronized early stopping. Hyperparameters are selected based on standard practices: learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 128, patience 20 epochs, and gradient clipping at norm 1.0.

---

## 5. Experimental Validation

We validate WaveDL through a case study on Lamb wave dispersion curve inversion. The training database was generated using the analytical Rayleigh-Lamb dispersion equations for an isotropic plate. We generated 100,000 samples by uniformly sampling thickness (1.0–10.0 mm) and shear velocity (3000–3250 m/s), with longitudinal velocity fixed at twice the shear velocity. The resulting dispersion curves were rasterized into 500×500 binary images simulating experimentally obtained frequency-wavenumber spectra.

The network was trained using the WaveDL pipeline on a cluster node with 4 NVIDIA V100 GPUs. Training used AdamW optimization with learning rate $10^{-3}$, batch size 128 (32 per GPU), weight decay $10^{-4}$, BF16 mixed precision, and patience of 20 epochs.

The model converged after 85 epochs (approximately 2.5 hours wall-clock time). On the held-out test set of 10,000 samples, thickness prediction achieved MAE of 0.042 mm (0.76% relative error) with Pearson correlation exceeding 0.999. Velocity prediction achieved MAE of 3.12 m/s (0.10% relative error) with correlation also exceeding 0.999. Both error margins are well within typical experimental uncertainty of ultrasonic transducers, suggesting the model achieves prediction accuracy at or below the experimental noise floor.

Multi-GPU scaling demonstrated near-linear speedup: 4-GPU training achieved 520 samples/second compared to 140 samples/second on a single GPU, representing 93% scaling efficiency. This confirms that the memory-mapped data pipeline successfully removes I/O bottlenecks, allowing training to be compute-bound rather than I/O-bound.

---

## 6. Discussion

The memory-mapped data formulation reduces RAM requirements from $O(N \cdot H \cdot W)$ to $O(B \cdot H \cdot W)$ where $B$ is the batch size—a critical enabler for training on terabyte-scale simulation databases. In practice, training on a 100 GB dataset required only ~2 GB of RAM per process. The DDP synchronization introduces communication overhead of $O(\log P)$ per batch due to tree-structured all-reduce, with empirical results showing greater than 90% scaling efficiency up to 8 GPUs.

Compared to traditional numerical solvers, WaveDL provides dramatic speedup. Transfer Matrix Method inference requires 150–500 ms per point on CPU, while WaveDL achieves 0.12 ms per point on GPU (amortized over batch of 128)—a speedup factor of approximately 2000×. This three-order-of-magnitude acceleration enables real-time inversion at acquisition rates exceeding 1 kHz.

WaveDL standardizes the ad-hoc scripts often used in physics-based deep learning into a reproducible framework. The strict separation of modeling and infrastructure ensures fair comparison of different architectures under identical conditions. The comprehensive unit test suite (over 200 tests covering models, data pipeline, metrics, registry, and distributed utilities) ensures reliability as the framework evolves.

The framework is designed for multi-output regression; classification tasks would require modifications to loss functions and metrics. While the current release does not include integrated hyperparameter optimization (e.g., Optuna), users can leverage Weights & Biases Sweeps for automated search. Future development priorities include physics-informed loss functions incorporating wave physics constraints, pre-trained models for transfer learning, and uncertainty quantification through ensemble methods or Bayesian approaches.

---

## 7. Conclusion

We have presented WaveDL, a robust deep learning framework specifically designed for the inverse characterization of guided waves in non-destructive evaluation applications. By addressing the fundamental engineering challenges of data scale, distributed synchronization, and physics-aware evaluation, WaveDL enables researchers to focus on scientific innovation rather than computational infrastructure.

The framework achieves sub-0.1 mm thickness accuracy and sub-5 m/s velocity accuracy on Lamb wave inversion, with inference speeds exceeding 2000× faster than traditional numerical solvers. The dimension-agnostic data handling, comprehensive test suite, and production-ready features make WaveDL suitable for both academic research and industrial deployment.

WaveDL is actively maintained and available at https://github.com/ductho-le/WaveDL under the MIT license. We welcome contributions from the community, including new model architectures, additional utilities, and documentation improvements.

---

## Acknowledgments

Ductho Le acknowledges the Natural Sciences and Engineering Research Council of Canada (NSERC) and Alberta Innovates for supporting this research through a research assistantship and graduate doctoral fellowship, respectively. This research was enabled in part by computational resources provided by Compute Ontario, Calcul Québec, and the Digital Research Alliance of Canada.

---

## References

[1] Rose, J.L. (2014). *Ultrasonic Guided Waves in Solid Media*. Cambridge University Press.

[2] Su, Z., & Ye, L. (2009). *Identification of Damage Using Lamb Waves: From Fundamentals to Applications*. Springer.

[3] Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

[4] Yang, H., Liu, J., Wang, Y., & Liu, J. (2022). "A convolutional neural network approach for inverse acoustic characterization of guided wave measurement." *Mechanical Systems and Signal Processing*, 169, 108759.

[5] Rautela, M., & Gopalakrishnan, S. (2021). "Deep learning for structural health monitoring: A review of Lamb wave-based damage detection." *Ultrasonics*, 116, 106496.

[6] Miorelli, R., Artusi, X., Reboud, C., Theodoulidis, T., & Poulakis, N. (2021). "Supervised deep learning for ultrasonic crack characterization using numerical simulations." *NDT & E International*, 119, 102405.

[7] Zhang, Y., Wang, X., Yang, Z., & Liu, Y. (2023). "Physics-informed neural networks for Lamb wave-based damage identification in composite laminates." *Engineering Applications of Artificial Intelligence*, 117, 105564.

[8] Falcon, W., & The PyTorch Lightning team. (2019). *PyTorch Lightning*. https://pytorch-lightning.readthedocs.io

[9] MONAI Consortium. (2020). *MONAI: Medical Open Network for AI*. https://monai.io

[10] Gugger, S., Debut, L., Wolf, T., Schmid, P., Mueller, Z., & Manber, M. (2022). *Accelerate*. https://huggingface.co/docs/accelerate

[11] Woo, S., Park, J., Lee, J.Y., & Kweon, I.S. (2018). "CBAM: Convolutional block attention module." *Proceedings of the European Conference on Computer Vision (ECCV)*, 3-19.

---

## Appendix A: Command Line Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `ratenet` | Name of registered model architecture |
| `--list_models` | flag | - | Print available models and exit |
| `--batch_size` | int | 128 | Training batch size per GPU |
| `--lr` | float | 0.001 | Initial learning rate for AdamW |
| `--epochs` | int | 1000 | Maximum number of training epochs |
| `--patience` | int | 20 | Early stopping patience |
| `--weight_decay` | float | 0.0001 | AdamW weight decay regularization |
| `--grad_clip` | float | 1.0 | Maximum gradient norm for clipping |
| `--data_path` | str | `train_data.npz` | Path to training data NPZ file |
| `--workers` | int | 8 | Number of DataLoader worker processes |
| `--seed` | int | 2025 | Random seed for reproducibility |
| `--resume` | str | None | Path to checkpoint directory to resume from |
| `--fresh` | flag | - | Force fresh training, ignore existing checkpoints |
| `--save_every` | int | 10 | Save periodic checkpoint every N epochs |
| `--output_dir` | str | `.` | Directory for checkpoints and logs |
| `--compile` | flag | - | Enable torch.compile for PyTorch 2.x optimization |
| `--precision` | str | `bf16` | Mixed precision mode: `bf16`, `fp16`, or `no` |
| `--wandb` | flag | - | Enable Weights & Biases experiment tracking |
| `--project_name` | str | `DL-Training` | Weights & Biases project name |

---

## Appendix B: Data Format Specification

**Input Data (NPZ)**

The framework expects NPZ archives with keys `input_train` and `output_train`:

- `input_train`: NumPy array of shape $(N, L)$ for 1D, $(N, H, W)$ for 2D, or $(N, D, H, W)$ for 3D data. The framework automatically adds the channel dimension.
- `output_train`: NumPy array of shape $(N, K)$ containing $K$ regression targets per sample.

Float32 dtype is recommended for both arrays.

**Checkpoint Structure**

```
best_checkpoint/
├── model.safetensors      # Model weights
├── optimizer.bin          # Optimizer state
├── scheduler.bin          # LR scheduler state
├── random_states_*.pkl    # RNG states for reproducibility
└── training_meta.pkl      # Epoch, best loss, patience counter
```

---

## Software Availability

The WaveDL framework is freely available under the MIT License. Source code, documentation, and usage examples can be obtained from: https://github.com/ductho-le/WaveDL

The software has been tested on Linux systems (Ubuntu 20.04, CentOS 7) and Windows with Python 3.11+ and PyTorch 2.0+.

---

*Target journal: Computer Physics Communications*

*Manuscript version: 2.0*

*Date: December 2025*
