# Comprehensive Methodology: T-TFGN Architecture and Formulation

---

## 1. Mathematical Formulation

Let patient $p$ have hospital admission history up to retrospective prediction timestamp $t_{\text{pred}}$.
The upcoming laboratory investigation occurs at $t_{\text{lab}} \ge t_{\text{pred}} + \Delta t_{\text{lead}}$, where $\Delta t_{\text{lead}} \ge 30\text{ minutes}$.

We seek to learn a parameterized function $f_\theta$ mapping pre-result multimodal observations to the Clinical Actionability Score (CAS):
$$\text{CAS} = \sigma\left(f_\theta(\mathbf{x}_{\text{lab}}, \Delta t_k, k, \mathbf{x}_{\text{context}})\right) \in [0, 1]$$

where:
- $\mathbf{x}_{\text{lab}} \in \mathbb{R}^{D_l}$ represents longitudinal trajectory features across lookback window $[t_{\text{pred}} - 48\text{h}, t_{\text{pred}}]$.
- $\Delta t_k \ge 0$ is the elapsed duration since the most recent observation for target analyte $k$.
- $k \in \{0, 1, 2, 3, 4\}$ denotes target analyte identity ($\text{Glucose}, \text{Creatinine}, \text{Potassium}, \text{Sodium}, \text{Hemoglobin}$).
- $\mathbf{x}_{\text{context}} \in \mathbb{R}^{D_c}$ comprises baseline demographics, chronic comorbidities, active pre-prediction medication orders, and vital sign trends.

---

## 2. Architectural Components

```
                   Input Features
        ┌───────────────────────────────────┐
        │  Longitudinal Lab   Target Index  │
        │    Trajectory x_lab      k        │
        └─────────────────┬─────────────────┘
                          │
            [Time-Decay Recency Layer]
            lambda_k = sigmoid(gamma_k * dt + 1)
            h_decayed = lambda_k * x_lab
                          │
              [Analyte Embedding Layer]
              e_k = Embedding(k)
                          │
             [Laboratory Encoder MLP]
             h_lab = MLP([h_decayed, e_k])
                          │
       ┌──────────────────┴──────────────────┐
       │                                     │
       │                   ┌─────────────────┴─────────────────┐
       │                   │      Clinical Context Vector      │
       │                   │             x_context             │
       │                   └─────────────────┬─────────────────┘
       │                                     │
       │                         [Context Encoder MLP]
       │                         h_context = MLP(x_context)
       │                                     │
       ├───────────────────┬─────────────────┤
       │                   │                 │
       │         [Bilinear Tensor Fusion]    │
       │         h_tensor = sum_r (U_r h_lab) * (V_r h_ctx)
       │                   │                 │
       └───────────────────┼─────────────────┘
                           │
             [Dynamic Adaptive Gate]
             gate = sigmoid(W_g [h_lab, h_ctx, h_tens] + b)
             h_fused = gate * W_l(h_lab) + (1-gate) * W_c(h_ctx) + W_t(h_tens)
                           │
              [Actionability Head MLP]
              z = Linear(ReLU(Linear(h_fused)))
                           │
                 [Sigmoid Output Head]
                 CAS = sigmoid(z) in [0, 1]
```

### A. Analyte-Specific Continuous Time Decay
$$\lambda_k = \sigma(\gamma_k \cdot \Delta t_k + 1.0)$$
$$\mathbf{h}_{\text{decayed}} = \lambda_k \odot \mathbf{x}_{\text{lab}}$$
where $\gamma_k$ is an analyte-specific learnable parameter initialized to $-0.05$. When $\gamma_k < 0$, larger elapsed intervals $\Delta t_k$ smoothly attenuate the historical observation's contribution.

### B. Low-Rank Bilinear Tensor Fusion
$$\mathbf{h}_{\text{tensor}} = \sum_{r=1}^R (\mathbf{U}_r \mathbf{h}_{\text{lab}}) \odot (\mathbf{V}_r \mathbf{h}_{\text{context}}) + \mathbf{b}$$
Decomposes 3D cross-modal outer-product interactions into rank-$R$ projection factors, capturing high-order multiplicative synergies (e.g. potassium slope $\times$ active furosemide) without parameter explosion.

### C. Patient-Specific Adaptive Gating
$$\mathbf{g} = \sigma\left(\mathbf{W}_g [\mathbf{h}_{\text{lab}}, \mathbf{h}_{\text{context}}, \mathbf{h}_{\text{tensor}}] + \mathbf{b}_g\right)$$
$$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{W}_l \mathbf{h}_{\text{lab}} + (1 - \mathbf{g}) \odot \mathbf{W}_c \mathbf{h}_{\text{context}} + \mathbf{W}_t \mathbf{h}_{\text{tensor}}$$
Enables dynamic convex modulation: when acute laboratory instability dominates, $\mathbf{g} \to 1$; when chronic disease severity and active medications dictate care, $\mathbf{g} \to 0$.
