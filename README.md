# GINN Encoder + 3D Spherical Quantum-Mechanical Decoder for Bond Dissociation Energy (BDE)

An end-to-end, fully differentiable physics-informed deep learning architecture that predicts **Bond Dissociation Energies (BDE)** by explicitly solving a 3D spherical Quantum-Mechanical (Schrödinger) model for every molecule in the dataset.

Unlike standard black-box machine learning models, this framework integrates physical laws directly into the neural network's forward pass, enforcing continuous spatial normalization, quantum mechanical constraints, and physical potential energy surfaces.

---

## 📌 Architecture Overview

The system processes molecular input through five interconnected phases in a continuous feedforward loop:
---

## 🛠 Key Components

### 1. Data Processing & Parsing (`Phase 0`)
* **Custom Recursive-Descent Formula Parser:** Converts NIST-style free-text bond labels (e.g., `CH3-C(C6H5)CN(CH3)`, `H-2H or H-D`, `Br-CO-C6H5`) into fragment compositions and bond orders.
* **Programmatic Element Feature Lookup:** Features for elements up to $Z=98$ are dynamically retrieved using the `mendeleev` library (covering electronegativity, covalent radii, ionization energy, electron affinity, and series one-hot encodings).
* **Graph Representation:** Molecules are converted into 2-node molecular graphs where node features represent fragment compositions and edge features carry bond order, electronegativity differences ($\Delta \chi$), reduced mass ($\mu$), and covalent radii sums.

### 2. GINN Encoder + Transformer (`Phase 1`)
* **Graph Isomorphism Network (GIN):** Utilizes sum-aggregation (Xu et al., 2019) to ensure injective multiset mapping over fragment graphs.
* **Self-Attention Transformer:** Applies multi-head self-attention across nodes to output a contextual embedding vector $S$.

### 3. Morse Potential Energy Surface Decoder (`Phase 2`)
* Maps contextual vectors to physically bounded parameters of the Morse potential energy curve:
  * **Well Depth ($D_e$):** Predicted within $[5, 2005]\text{ kJ/mol}$.
  * **Potential Width ($a$):** Predicted within $[0.3, 4.0]\text{ \AA}^{-1}$.
  * **Equilibrium Distance ($R_e$):** Extracted directly from experimental ground truth.

### 4. 3D Spherical Quantum-Mechanical Solver (`Phase 3`)
* **Radial Solver:** Diagnoses the centrifugal-corrected radial Time-Independent Schrödinger Equation (TISE) via exact finite-difference matrix diagonalization (`torch.linalg.eigh`) across angular momentum channels $l \in \{0, 1, 2, 3\}$.
* **Angular Solver:** Evaluates complex Spherical Harmonics $Y_l^m(\theta, \phi)$ with Condon-Shortley phase using Associated Legendre Polynomials generated via explicit Rodrigues' formula chains.
* **LCAO Wavefunction:** Combines radial and angular channels via learned orbital character distributions $p(l,m)$ to construct atomic wavefunctions $\psi_{\text{atom}, i}$, which are linearly combined into a normalized molecular orbital wavefunction $\psi_{\text{molecule}}$.

### 5. Prediction & Spatial Geometry Extraction (`Phases 4 & 5`)
* **Energy Prediction:** Computes zero-point vibrational energy ($E_{\text{vib,mol}}$) from radial eigenvalues and predicts bond strength:
  $$\hat{\text{BDE}} = D_e - E_{\text{vib,mol}}$$
* **Unsupervised Spatial Diagnostics:** Applies a temperature-scaled softmax over the 5D probability density grid $P(r, \theta, \phi, t) = |\Psi|^2$. Argmax extraction determines spatial orientation vectors to infer bond angles.

---

## 🔬 Potential Chemistry Applications

1. **Physics-Informed QSAR/QSPR Modeling:** Offers interpretable predictions where every intermediate latent parameter corresponds to a real physical quantity ($D_e, a, R_{0,l}, E_{0,l}, p(l,m)$).
2. **Automated Thermochemical Screening:** Accelerates reaction enthalpy and bond strength screening without relying on heavy ab initio electronic structure calculations (such as DFT or Coupled Cluster).
3. **Surrogate Modeling for Quantum Chemistry:** Serves as a differentiable surrogate model for radial Schrödinger equation solving and orbital hybridization estimation.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed along with the required dependencies:

```bash
pip install torch torch-geometric mendeleev numpy pandas matplotlib
