# Graph Neural Network for Network Intrusion Detection with Cross-Dataset Generalization, Adversarial Robustness, and Zero-Day Detection

**Target Venue:** IEEE Access
**Status:** DRAFT — results sections marked with `[RESULTS PLACEHOLDER]`

---

## Abstract

Network intrusion detection systems (NIDS) based on graph neural networks (GNNs) have shown promising results on individual benchmark datasets, yet three critical gaps persist: most studies omit cross-dataset generalization testing, few evaluate robustness under adversarial perturbation, and even fewer address zero-day (novel attack) detection within a unified pipeline. This paper presents a staged GNN-based NIDS that combines Time2Vec temporal encoding, Edge-augmented Graph Attention Network v2 (E-GATv2) encoding, adversarially-regularized masked autoencoder (MAE) pretraining, conditional variational autoencoder (CVAE) minority-class augmentation, focal-loss classification with per-class threshold calibration, and prototypical few-shot learning for zero-day detection — all within a strictly time-respecting pipeline that prevents chronological leakage. We train on NF3-CSE-CIC-IDS2018 and NF3-UNSW-NB15 under a unified 11-class behavioral taxonomy, and evaluate in-domain, cross-dataset (NF3-ToN-IoT, NF3-BoT-IoT), and out-of-schema (CIC-DDoS2019, CIC-Darknet2020) without fine-tuning. Adversarial robustness is assessed via projected gradient descent (PGD) attacks at perturbation magnitudes up to ε=0.05. A multi-signal explainability analysis combining native attention weights with SHAP feature attribution validates architectural decisions and provides per-class feature importance. `[RESULTS PLACEHOLDER: headline in-domain macro-F1, cross-dataset macro-F1, adversarial robustness at ε=0.05, zero-day detection AUC]`

**Keywords:** Network intrusion detection, graph neural networks, adversarial robustness, few-shot learning, zero-day detection, cross-dataset generalization, explainable AI

---

## 1. Introduction

Network intrusion detection remains a critical component of cybersecurity infrastructure as attack surfaces expand with the proliferation of IoT devices, cloud services, and increasingly sophisticated threats. Machine learning-based NIDS have demonstrated substantial improvements over signature-based approaches, particularly in detecting previously unseen attack variants. However, the translation of these research advances to operational settings is hindered by several persistent gaps.

First, the overwhelming majority of GNN-based NIDS studies evaluate exclusively on in-domain test splits drawn from the same dataset distribution as training data [CITATION NEEDED: survey of cross-dataset evaluation in NIDS literature]. Real-world deployment requires generalization to network environments, traffic patterns, and attack behaviors not represented in training — a condition that single-dataset evaluation fundamentally cannot assess.

Second, adversarial robustness in graph-based NIDS remains underexplored. While adversarial attacks against network intrusion detection models are well-documented in the image and tabular domains, the intersection of adversarial perturbation and graph-structured flow data has received limited attention [CITATION NEEDED: ARGANIDS, problem-space structural adversarial attacks paper]. A model that achieves high in-domain accuracy but collapses under small adversarial perturbations provides a false sense of security.

Third, zero-day attack detection — recognizing attack behaviors that belong to no known training class — is typically treated as a separate research thread from standard NIDS classification. Pipelines that address binary detection, multiclass classification, and novelty detection within a single architecture are rare, yet operational NIDS must perform all three functions simultaneously.

This paper addresses these gaps through a staged graph neural network architecture with the following contributions:

1. **A unified staged pipeline** combining Time2Vec temporal encoding, E-GATv2 graph attention with edge-augmented message passing, adversarially-regularized MAE self-supervised pretraining, CVAE-based minority class augmentation, two-stage (binary → multiclass) classification with focal loss and per-class threshold calibration, and prototypical few-shot learning for zero-day detection. While individual components draw on established techniques, their integration into a single time-respecting graph NIDS pipeline with adversarial training at multiple stages is, to our knowledge, novel.

2. **Rigorous cross-dataset blind evaluation** on four external datasets — two in-schema (NF3-ToN-IoT, NF3-BoT-IoS) and two out-of-schema (CIC-DDoS2019, CIC-Darknet2020) — with zero fine-tuning. This addresses the generalizability question that most published NIDS evaluations leave unanswered.

3. **Adversarial robustness characterization** through PGD attacks at multiple perturbation magnitudes (ε = 0 to 0.05) during both training and evaluation, producing a robustness curve rather than a single-point measurement.

4. **Multi-signal explainability** combining native E-GATv2 attention weights with SHAP feature attribution, cross-validated across both signals, and explicitly tied to architectural decisions (per-class threshold calibration), grounding the interpretability claim in two independent evidence sources rather than one.

We do not claim a fundamentally new neural architecture. Rather, our contribution is the rigorous integration and evaluation of complementary techniques within a single leakage-free pipeline, tested under conditions — cross-dataset generalization, adversarial perturbation, and zero-day detection — that approximate the challenges of real deployment more closely than standard single-dataset benchmarks.

The remainder of this paper is organized as follows. Section 2 reviews related work across four thematic areas. Section 3 describes the datasets and our unified 11-class behavioral taxonomy. Section 4 details the nine-stage methodology. Section 5 presents the experimental setup. `[RESULTS PLACEHOLDER: Section 6 (Results), Section 7 (Ablation), Section 8 (Explainability)]` Section 9 discusses limitations, and Section 10 concludes.

---

## 2. Related Work

We organize related work into four thematic areas: temporal graph methods for NIDS, adversarial robustness in GNN-based intrusion detection, few-shot and zero-day detection, and cross-dataset evaluation practices.

### 2.1 Temporal Graph Neural Networks for NIDS

Graph-based approaches to network intrusion detection model flows as edges connecting host endpoints, enabling the model to capture structural attack patterns — such as distributed denial-of-service fan-in structures or lateral movement paths — that tabular feature-based methods miss. TE-G-SAGE [CITATION NEEDED: TE-G-SAGE — temporal edge graph SAGE, verify authors/venue/year] introduced time-respecting graph construction with chronological train/val/test splits, demonstrating that random splitting causes look-ahead leakage that artificially inflates reported metrics. Our pipeline adopts this temporal discipline and extends it with Time2Vec encoding [CITATION NEEDED: Time2Vec paper — verify authors/venue] for learnable periodic temporal features, replacing the random-walk-based temporal encoding (CTDNE) used in earlier graph NIDS work.

E-GATv2 (Edge-augmented Graph Attention Network v2) extends the GATv2 attention mechanism [CITATION NEEDED: GATv2 paper — Brody et al., verify venue/year] by injecting edge features directly into attention score computation at every layer, rather than only at the aggregation step. This is particularly relevant for NIDS: flow-level features such as packet counts, TCP flags, and inter-arrival times carry discriminative signal that should influence which neighbors the model attends to, not just what information is aggregated from them. Prior work including StrGNN [CITATION NEEDED: StrGNN — verify authors/venue/year] and PPT-GNN [CITATION NEEDED: PPT-GNN — verify authors/venue/year] has demonstrated the value of edge-feature-aware message passing for flow-based NIDS; our E-GATv2 implementation builds on this lineage with the dynamic attention mechanism of GATv2.

TGN (Temporal Graph Networks) [CITATION NEEDED: TGN paper — Rossi et al., verify venue/year] provides a general framework for continuous-time dynamic graphs, but its focus on node-level prediction tasks differs from our flow-classification setting, where edges themselves are the prediction targets.

### 2.2 Adversarial Robustness in GNN-based NIDS

Adversarial vulnerability in network intrusion detection has been demonstrated in multiple contexts. ARGANIDS [CITATION NEEDED: ARGANIDS — verify authors/venue/year] showed that GAN-generated adversarial flow perturbations can cause significant misclassification in MLP and CNN-based NIDS. The problem-space structural adversarial attacks paper [CITATION NEEDED: verify authors/venue/year] extended adversarial examples to graph-structured NIDS by perturbing both node features and graph structure. The IoT GNN+FGSM/DeepFool study [CITATION NEEDED: verify authors/venue/year] evaluated FGSM and DeepFool attacks against GNN-based IoT NIDS.

Our approach differs from prior adversarial NIDS work in three respects. First, we integrate adversarial training (PGD with ε=0.03 at training time) into a self-supervised MAE pretraining stage rather than applying it only during supervised fine-tuning — the encoder learns to reconstruct clean features from adversarially perturbed inputs, building robustness into the representation itself. Second, we evaluate adversarial robustness as a curve across multiple ε values (0 to 0.05) rather than a single attack strength, providing a more complete picture of model degradation. Third, we assess whether adversarial training for robustness trades off against clean-data accuracy and cross-dataset generalization, which the ablation study addresses directly.

### 2.3 Few-Shot and Zero-Day Detection

Zero-day attack detection — identifying attack behaviors not represented in training data — has been approached through anomaly detection, open-set recognition, and few-shot learning paradigms. Prototypical networks [CITATION NEEDED: Prototypical networks — Snell et al., NeurIPS 2017, verify] learn a metric space where classification is performed by computing distances to class prototypes, enabling generalization to novel classes from limited support examples. The prototypical capsule network [CITATION NEEDED: verify authors/venue/year] applied this concept to NIDS with capsule-network-based feature extraction.

Our prototypical stage differs from standard prototypical networks in two ways: we use attention-weighted prototype computation (where support samples are weighted by learned relevance rather than averaged uniformly), and we employ an ensemble of five episodic prototype sets at inference time to reduce support-sampling variance. CSCVAE-NID [CITATION NEEDED: verify authors/venue/year] addressed the related problem of class imbalance via variational autoencoders; we extend this idea with a conditional VAE that generates synthetic minority-class embeddings conditioned on class identity, targeting the specific classes (Generic, Shellcode/Worms, Backdoor, Infiltration) that remain small even after taxonomic merging.

MalMoE [CITATION NEEDED: verify authors/venue/year] and the cross-domain heterogeneous ensemble paper [CITATION NEEDED: verify authors/venue/year] explore multi-expert and ensemble approaches to the generalization problem, which conceptually parallels our staged pipeline where different model components specialize in different sub-tasks (binary detection, multiclass discrimination, novelty detection).

### 2.4 Cross-Dataset Evaluation in NIDS Research

A systematic review of recent GNN-based NIDS papers reveals that cross-dataset evaluation — testing a model trained on dataset A against dataset B with zero fine-tuning — is the exception rather than the rule [CITATION NEEDED: survey/meta-review reference]. Most studies report results on a single dataset (commonly CIC-IDS2017/2018, UNSW-NB15, or CIC-DDoS2019 in isolation). When multiple datasets are used, they are typically merged and shuffled before splitting, which tests distribution shift within a combined pool but does not test generalization to genuinely unseen network environments.

The REAL-IoT study [CITATION NEEDED: verify authors/venue/year] is a notable exception, evaluating cross-dataset performance on IoT-specific traffic. Our evaluation extends this principle: we test on two in-schema datasets (ToN-IoT, BoT-IoT) that share the NF3 feature format but represent different network environments and attack distributions, and on two out-of-schema datasets (CIC-DDoS2019, CIC-Darknet2020) that use entirely different feature schemas, requiring feature mapping. This four-dataset blind evaluation provides a more comprehensive assessment of generalization than any single cross-dataset comparison.

The standard NetFlow feature set by Sarhan et al. [CITATION NEEDED: Sarhan et al. NetFlow feature set paper — verify venue/year] provides the feature foundation for the NF3 datasets used in this study. We adopt the NF3 feature schema and extend it with temporal encoding at the flow level rather than the packet level — a limitation we discuss explicitly in Section 9.

---

## 3. Datasets and Unified Taxonomy

### 3.1 NF3 Dataset Suite

We use four datasets in the NF3 (NetFlow v3) format [CITATION NEEDED: NF3 dataset paper — verify authors, venue, year], which provides 53 standardized flow-level features extracted from raw packet captures. NF3 was chosen over the earlier NF-v2 format specifically for its eight inter-arrival time (IAT) features (SRC_TO_DST_IAT and DST_TO_SRC_IAT, each with min, max, avg, stddev), which capture burst and timing patterns critical for distinguishing attack behaviors such as DDoS flooding, brute-force login attempts, and botnet beaconing.

**Training and validation datasets:**
- **NF3-CSE-CIC-IDS2018:** `[tab01: total flows, benign/attack ratio]` flows collected in a controlled network environment with six attack families: Bot, DoS, DDoS, Infiltration, Brute Force, and Web Attack.
- **NF3-UNSW-NB15:** `[tab01: total flows, benign/attack ratio]` flows with nine attack families: Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms.

**In-schema blind test datasets (same NF3 feature format, zero fine-tuning):**
- **NF3-ToN-IoT:** `[tab01: total flows, benign/attack ratio]` flows from an IoT-specific network testbed.
- **NF3-BoT-IoT:** `[tab01: total flows, benign/attack ratio]` flows capturing botnet traffic patterns.

**Out-of-schema blind test datasets (different feature schema, requires feature mapping):**
- **CIC-DDoS2019:** Raw CIC-flow format. DDoS-specific traffic.
- **CIC-Darknet2020:** Raw CIC-flow format. Darknet/background radiation traffic.

The NF3 data was sourced from the UQ eSpace repository [CITATION NEEDED] and Kaggle mirrors. CIC datasets were sourced from the Canadian Institute for Cybersecurity's official repository [CITATION NEEDED].

### 3.2 Unified 11-Class Behavioral Taxonomy

Training jointly on two datasets with different attack label vocabularies requires a unified taxonomy. The raw label union spans 14 distinct attack labels (9 from UNSW-NB15, 6 from CIC-IDS2018, with DoS shared). We collapse these to 11 behavioral categories by merging labels that share the same attack mechanism, regardless of source dataset. Table 2 presents the full mapping.

This taxonomic unification is itself a methodological contribution: most cross-dataset NIDS studies either restrict evaluation to binary (benign/attack) classification or train separate models per dataset, avoiding the label-alignment problem entirely. By defining behaviorally-grounded merge criteria and reporting per-class results under the unified taxonomy, we enable joint training across structurally different datasets while maintaining fine-grained attack-type discrimination.

The unified classes and their constituent source labels are:

| Unified Class | Source Labels Merged | Behavioral Rationale |
|---|---|---|
| Benign | Benign (both datasets) | — |
| DoS/DDoS | UNSW DoS + CIC DoS + CIC DDoS | Volumetric/resource exhaustion; single-source vs. distributed distinction is topological, not behavioral |
| Reconnaissance | UNSW Reconnaissance + UNSW Analysis | Pre-attack probing; Analysis (port scans, spam, penetration probing) is behaviorally reconnaissance |
| Exploits | UNSW Exploits + UNSW Fuzzers | Malformed-input exploitation; both target vulnerabilities via crafted payloads |
| Backdoor | UNSW Backdoor | Standalone — C2/persistent-access behavior distinct from automated botnet traffic |
| Bot | CIC Bot | Automated botnet C2; periodic beaconing pattern distinct from manual backdoor access |
| Brute Force | CIC Brute Force | — |
| Web Attack | CIC Web Attack (XSS, SQLi as bundled in CIC labeling) | — |
| Infiltration | CIC Infiltration | — |
| Generic | UNSW Generic | Standalone — cipher/block-algorithm attacks structurally unlike any other class |
| Shellcode/Worms | UNSW Shellcode + UNSW Worms | Payload-delivery/self-propagation; merged as both are near-singleton classes with closest behavioral pairing |

**Known imbalance after unification:** Generic, Shellcode/Worms, Backdoor, and Infiltration are the four smallest classes post-merge. These are the primary targets for CVAE-based augmentation (Stage E, Section 4.5) and the primary classes to monitor in per-class recall analysis.

### 3.3 Feature Selection

The NF3 schema provides 53 flow-level features. We retain 44 as classifier input and exclude 9 fields for reasons of identity leakage, sparsity, or redundancy. Table 3 provides the complete feature schema with grouping. Key exclusion rationales:

- **IPV4_SRC_ADDR, IPV4_DST_ADDR:** Direct identity leakage — the model would memorize attacker IPs rather than learning behavior. Used exclusively as node identities for graph construction.
- **L4_SRC_PORT, L4_DST_PORT:** High-cardinality near-identity fields; documented exclusion practice in prior NF-based NIDS studies.
- **L7_PROTO:** Can be near-unique to a class, enabling protocol-identity shortcut learning.
- **FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS:** Start time is consumed exclusively by Time2Vec encoding (Stage B); end time is redundant given flow duration.
- **FTP_COMMAND_RET_CODE, DNS_QUERY_ID:** Near-constant/sparse or random identifiers with no behavioral signal.

Protocol-conditional fields (ICMP_TYPE, ICMP_IPV4_TYPE for ICMP flows; DNS_QUERY_TYPE, DNS_TTL_ANSWER for DNS flows) are filled with 0 for non-applicable flows. No separate missingness indicator column is added — prior work on this project identified that indicator columns caused dataset-identity leakage.

The final edge input to the encoder is **61-dimensional**: 44 kept raw features (z-score normalized) concatenated with 17 Time2Vec temporal encoding dimensions (Section 4.2).

### 3.4 Stated Limitation: Flow-Level Temporal Granularity

NF3 timestamps are flow-level (FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS), not packet-level. While the eight inter-arrival time features provide sub-flow timing statistics, the fine-grained packet-level temporal dynamics available in raw pcap-based approaches (e.g., millisecond-scale inter-packet intervals within a single flow) are lost. Our Time2Vec encoding operates on flow start times, which capture ordering and spacing between flows but not within-flow packet timing. This limitation is inherent to the NF3 format and is acknowledged as a threat to temporal validity — a packet-level implementation of the same architecture would likely extract richer temporal signal.

---

## 4. Methodology

Our architecture follows a nine-stage pipeline (Figure 1), with each stage consuming the output of preceding stages and no stage accessing data from later stages during training. This section describes each stage in sequence.

### 4.1 Stage A: Time-Respecting Split and Graph Construction

We partition each dataset chronologically by `FLOW_START_MILLISECONDS` into training (70%), validation (15%), and test (15%) sets. The chronological split prevents look-ahead leakage: a flow that occurs later in time cannot influence the model's prediction on an earlier flow during training.

Critically, we construct three **physically separate graphs** (G_train, G_val, G_test) rather than one graph with a train/val/test mask. This ensures that neighbor sampling for validation and test nodes cannot traverse edges from future time windows. Nodes are identified by hashed (IP, port) tuples; edges represent individual flows with 44 normalized features as edge attributes.

We further segment each graph into 120-second sliding windows, yielding lists of PyG `Data` objects per split. This windowed construction keeps memory bounded on GPU hardware and aligns naturally with the online inference setting where flows arrive continuously and are processed within a sliding temporal context window.

A StandardScaler is fit on E_train edge features only and applied frozen to validation and test splits. Split indices are persisted to disk as the single source of truth loaded by every downstream notebook.

### 4.2 Stage B: Time2Vec Temporal Encoding

We encode each flow's start time using a learnable Time2Vec embedding [CITATION NEEDED] rather than feeding the raw timestamp or relying on handcrafted temporal features. The Time2Vec function is:

φ(t) = [ω₀t + b₀, sin(ω₁t + b₁), sin(ω₂t + b₂), ..., sin(ω₁₆t + b₁₆)]

where ωᵢ and bᵢ are learnable parameters (k=16 periodic terms, 1 linear term, 17 dimensions total). The linear term captures global temporal trends (e.g., overall traffic volume growth over the capture period); the periodic terms capture recurring patterns at learned frequencies. We initialize ω from a log-uniform distribution spanning expected flow-duration timescales (milliseconds to minutes) and train jointly with the encoder.

Time normalization (min-max scaling of raw timestamps) is fit on E_train's time range only and applied frozen to validation and test, maintaining the same temporal leakage discipline as the feature scaler.

The 17-dim Time2Vec output is concatenated with the 44 z-score-normalized raw features, producing a **61-dimensional edge feature vector** that serves as input to the encoder.

### 4.3 Stage C: E-GATv2 Encoder

Our encoder is a 3-layer Edge-augmented Graph Attention Network v2 (E-GATv2). GATv2 [CITATION NEEDED: Brody et al.] improves upon the original GAT by computing attention scores dynamically rather than statically, allowing the ranking of attention scores to vary across nodes. We extend GATv2 with edge-augmented attention: at each layer, the 61-dim edge features are linearly projected and injected into the attention score computation, so that flow-level features (packet counts, TCP flags, IAT statistics) influence which neighbors the model attends to.

Architecture details:
- **Layers:** 3, with neighbor sampling fan-out [15, 10, 5] (layer 1 samples 15 neighbors, layer 2 samples 10, layer 3 samples 5).
- **Hidden dimension:** 256 per layer, 8 attention heads.
- **Activation:** ELU. **Regularization:** Dropout 0.3 (attention), 0.2 (feature). Residual connections with LayerNorm between layers.
- **Node initialization:** Learned 128-dim embedding table, random initialization, trained jointly with encoder.
- **Output per flow:** Concatenation of source node embedding (256-dim), destination node embedding (256-dim), and edge embedding (256-dim) = **768-dim flow representation**.

This 768-dim vector is the shared representation consumed by every downstream stage (D through H). The encoder is implemented using PyTorch Geometric's `GATv2Conv` with the `edge_dim=61` parameter for native edge-feature support.

### 4.4 Stage D: Adversarially-Regularized MAE Pretraining

We pretrain the encoder using a masked autoencoder (MAE) objective on benign traffic only. By training exclusively on benign flows, the encoder learns to reconstruct normal traffic patterns; deviations from these patterns (attack flows) produce higher reconstruction error, providing an implicit anomaly signal that complements the supervised stages.

**Masking:** 40% of the 61 edge features are randomly zeroed per batch (post-Time2Vec concatenation, so temporal features can also be masked).

**Adversarial regularization (FGSM):** Before masking, we apply a Fast Gradient Sign Method (FGSM) perturbation to the unmasked edge features at ε = 0.01–0.03 in normalized feature space. Perturbed features are clipped to stay within the per-feature [min, max] range observed in the training set. The encoder must reconstruct the clean (unperturbed) target despite receiving a perturbed and partially masked input. This adversarial step — inspired by ARGANIDS but applied at the self-supervised pretraining stage rather than during supervised fine-tuning — builds robustness into the representation before any task-specific head is attached.

**Decoder:** MLP (256 → 128 → 61), reconstructing the original 61-dim edge features at masked positions only.

**Training:** AdamW optimizer, lr=1e-3, weight decay=1e-5, cosine annealing schedule, 30 epochs, batch size 4096, fp16 mixed precision. Loss is MSE computed only on masked positions. Early stopping with patience=5 on validation reconstruction MSE.

The pretrained encoder weights are carried forward to Stage F; the decoder is discarded after pretraining.

### 4.5 Stage E: CVAE Minority-Class Augmentation

Class imbalance after taxonomic unification disproportionately affects four classes: Generic, Shellcode/Worms, Backdoor, and Infiltration. To mitigate this, we train a Conditional Variational Autoencoder (CVAE) that generates synthetic 768-dim flow embeddings for minority classes.

**Architecture:** The encoder maps (768-dim flow embedding + 11-dim one-hot class condition) through layers 256 → 128 → (μ, σ) with a 64-dim latent space. The decoder maps (64-dim latent + 11-dim condition) through 128 → 256 → 768, reconstructing the flow embedding.

**Training:** Loss = MSE reconstruction + β·KL divergence, with β=0.5 to prevent posterior collapse. Adam optimizer, lr=5e-4, batch size 512, 50 epochs. Trained only on classes below the median class count.

**Generation:** Synthetic embeddings are generated until each minority class reaches approximately 40% of the majority class count — not full balance, to avoid synthetic-dominated training in Stage G. All synthetic samples are tagged `is_synthetic=True` and injected exclusively into Stage G's training pool at a 1:1 ratio with real minority samples. Synthetic embeddings are never present in validation or test splits.

### 4.6 Stage F: Binary Classification (Stage-1 Head)

The first classification stage performs binary benign/attack detection using a head MLP (768 → 256 → 64 → 2) on top of the pretrained encoder.

**Two-phase training:** Phase A (5 epochs) freezes the encoder and trains only the head (lr=1e-3). Phase B (15 epochs) unfreezes the encoder for joint fine-tuning (lr=1e-5 encoder, 1e-4 head). This progressive unfreezing prevents the randomly initialized head from producing large gradients that destabilize the pretrained encoder weights.

**Loss:** Focal loss (γ=2, α = inverse class frequency) to focus training on hard-to-classify examples, which disproportionately include attack variants that resemble benign traffic.

**Class balancing:** Per-epoch undersampling of benign flows to a 2:1 benign-to-attack ratio, resampled each epoch (not a static pre-sampled pool).

**Adversarial training:** Projected Gradient Descent (PGD) with ε=0.03, α=0.01, 7 steps, applied to 30% of each batch. PGD adversarial examples are generated on-the-fly during training; the model is trained on both clean and adversarially perturbed examples.

**Decision threshold:** After training, the binary decision threshold is calibrated on the validation set to achieve attack-class recall ≥ 0.995. This favors recall over precision — the multiclass Stage G is designed to resolve false positives.

### 4.7 Stage G: Multiclass Classification (Stage-2 Head)

Flows classified as "attack" by Stage F are passed to the multiclass head (MLP 768 → 256 → 11) for fine-grained attack-type classification among the 11 unified classes.

**Training pool:** Real minority-class samples + Stage E synthetic embeddings (1:1 ratio) + per-epoch undersampled majority classes. The encoder continues fine-tuning from the Stage F state (not reset) at lr=1e-5.

**Loss:** Focal loss (γ=2) with per-class α weights computed via effective-number-of-samples reweighting, which handles extreme class imbalance more robustly than plain inverse frequency.

**Adversarial training:** Same PGD configuration as Stage F (ε=0.03, α=0.01, 7 steps).

**Per-class threshold calibration:** Rather than using a single global argmax threshold, we calibrate a decision threshold per class via grid search (0.1–0.9, step 0.05) on the validation set, maximizing per-class F1. This addresses the finding from prior work that globally tuned thresholds disadvantage classes with overlapping feature distributions — a finding our XAI analysis (Section 8) explicitly investigates by checking whether classes needing separate thresholds share top discriminative features.

### 4.8 Stage H: Prototypical Few-Shot Network

Flows with Stage G confidence below a threshold τ₂, or flagged by independent novelty detection, are routed to the prototypical few-shot network for zero-day assessment.

**Episodic training:** 5-way, 5-shot, 15 query samples per class, 200 episodes per epoch, 30 epochs. Each episode samples 5 classes, 5 support examples per class, and 15 query examples per class — all drawn exclusively from G_train.

**Attention-weighted prototypes:** Rather than computing a plain mean of support embeddings, we learn a small MLP (768 → 1) that scores each support sample by its relevance to the class prototype. A softmax over scores produces attention weights; the prototype is the weighted sum. This prevents outlier support samples (e.g., mislabeled or ambiguous flows) from distorting the prototype.

**Distance metric:** Cosine similarity between query embedding and prototype.

**Novelty detection:** At inference, if the maximum cosine similarity to any known-class prototype falls below a threshold τ, the flow is flagged as "novel/zero-day" rather than forced into the closest known class. τ is tuned via leave-one-class-out validation: one class is held out during training, and τ is set to maximize novelty detection F1 on the held-out class. This is repeated for all 11 classes, producing both a global τ and per-class τ values.

**Inference voting:** To reduce variance from support set sampling, we ensemble 5 different prototype sets (different random support draws), with the final classification determined by majority vote.

### 4.9 Inference Pipeline

At deployment time, the inference path for an incoming flow is:

1. Extract 44 NF3 features + compute Time2Vec(start_time) → 61-dim input.
2. Insert into sliding-window graph (evict edges outside 120s window).
3. E-GATv2 encoder forward pass → 768-dim flow representation.
4. Stage F binary head → if benign, stop and return.
5. Stage G multiclass head → if confidence ≥ τ₂ and max similarity above τ, return predicted class.
6. Otherwise → Stage H prototypical match → return known class or "novel/zero-day" alert.

Target inference latency is <30ms per flow on T4-class GPU hardware. Only the encoder and heads (F, G, H) are exported; Stage D's decoder, Stage E's CVAE, and Stage A's split logic are training-only components.

---

## 5. Experimental Setup

### 5.1 Hyperparameters

Table 4 summarizes all hyperparameters by stage. Key configuration:

| Stage | Component | Key Parameters | Optimizer/LR | Epochs | Batch Size |
|---|---|---|---|---|---|
| A | Graph construction | window=120s, split=70/15/15 | — | — | — |
| B | Time2Vec | k=16, dim=17 | Joint w/ C | Joint | Joint |
| C | E-GATv2 | 3 layers, hid=256, heads=8, fanout=[15,10,5] | Joint w/ D–G | Joint | Joint |
| D | MAE pretrain | mask=40%, FGSM ε=0.01–0.03 | AdamW 1e-3 | 30 | 4096 |
| E | CVAE augment | latent=64, β=0.5 | Adam 5e-4 | 50 | 512 |
| F | Binary head | focal γ=2, PGD ε=0.03/steps=7 | 1e-3→1e-5/1e-4 | 5+15 | 4096 |
| G | Multiclass head | focal γ=2, per-class threshold | AdamW 1e-5 | 20 | 2048 |
| H | Prototypical net | 5-way/5-shot, cosine | Adam 1e-4 | 30 | Episodic |

### 5.2 Compute Environment

All experiments are conducted on a Kaggle notebook environment with dual NVIDIA T4 GPUs (T4x2), 16 GB VRAM per GPU. Mixed precision (fp16) training is used throughout via PyTorch's automatic mixed precision (`torch.cuda.amp`). The complete pipeline from raw data to final results requires approximately 15–30 GPU-hours, with the MAE pretraining (Notebook 2, ~4–8 hours) and ablation retraining (Notebook 7, ~6–12 hours) dominating compute time.

### 5.3 Reproducibility

We set global seed = 42 for Python `random`, NumPy, and PyTorch (including `torch.backends.cudnn.deterministic = True`). All stages use the same chronological split indices, persisted once in Notebook 1 and loaded identically by every downstream notebook. Every stage checkpoints its model weights with a complete hyperparameter dictionary (`config.json`) and a structured results log (`results_log.json`). Figures and tables are saved immediately upon generation in both editable (.svg, .md) and publication-ready (.png 300dpi, .csv) formats. The complete codebase, trained checkpoints, and output artifacts are available at `[REPOSITORY URL TO BE ADDED]`.

### 5.4 Leakage Prevention

Chronological leakage — where information from later-in-time flows influences predictions on earlier flows — is a documented threat to validity in temporal GNN research. We implement a six-point leakage checklist verified at the start of every notebook:

1. Split indices loaded from the persisted file, never recomputed inline.
2. Scaler and temporal normalizer fit only on E_train, loaded frozen elsewhere.
3. `label_map.yaml` and `feature_manifest.yaml` loaded identically across notebooks.
4. No global aggregate statistics computed across the full unsplit dataset.
5. Graph neighbor sampling for G_val/G_test cannot reach E_train-only edges.
6. Time2Vec's min-max time normalization fit on E_train's time range only.

Additionally, the physical separation of G_train, G_val, and G_test ensures that neighbor sampling during validation and testing cannot traverse edges from future time windows — a structural guarantee beyond the post-hoc masking approach used in some prior work.

### 5.5 Evaluation Metrics

- **In-domain:** Macro-F1, per-class precision/recall/F1, false alarm rate (FAR), AUC-ROC on the chronological test split.
- **Cross-dataset:** Same metrics on blind test datasets with zero fine-tuning. In-schema (ToN-IoT, BoT-IoT) and out-of-schema (DDoS2019, Darknet2020) results are reported separately and clearly labeled.
- **Adversarial robustness:** Macro-F1 at PGD ε = {0, 0.01, 0.03, 0.05}, producing a robustness curve rather than a single-point measurement.
- **Zero-day detection:** Per-class precision/recall/F1 from leave-one-class-out evaluation, plus ROC and precision-recall curves aggregated across all 11 leave-one-out runs.
- **Inference latency:** Milliseconds per flow, single-flow path and batched throughput, benchmarked on T4 GPU.

### 5.6 Ablation Design

To quantify the contribution of each architectural component, we retrain four variants with identical seed (42) and splits, each removing one component:

1. **no-Time2Vec:** Time2Vec removed; 44-dim edge input (no temporal encoding).
2. **no-CVAE:** Stage E skipped entirely; no synthetic minority-class embeddings in Stage G.
3. **no-adversarial-training:** PGD adversarial training removed from Stages F and G.
4. **no-prototypical-stage:** Stage H removed; Stage G multiclass output used directly for all flows.

The full model and all variants are evaluated on the same test splits. Macro-F1 delta from the full model is reported for each variant.

### 5.7 Explainability Analysis

We employ two complementary XAI methods, cross-validated against each other:

1. **Native E-GATv2 attention weights:** Per-edge attention scores extracted from the final encoder layer (layer 3) at inference time, aggregated per unified attack class, revealing which neighboring flows and hosts the model relies on for each class.

2. **SHAP feature attribution:** GradientSHAP (with KernelSHAP fallback) computed on the 61-dim input space (44 raw features + 17 Time2Vec), not the opaque 768-dim embedding — keeping attributions interpretable as named flow-level features. Background: 100 benign flows from E_train. Sample: ~2000 flows per class stratified from the validation set.

Where SHAP and attention findings agree, that constitutes our strongest interpretability claim. Where they disagree, the disagreement is reported explicitly as discussion material. We further tie XAI findings back to the per-class threshold calibration (Stage G): if classes requiring separate thresholds share top SHAP features, this explains why a global threshold was insufficient — connecting interpretability to an architectural design decision.

---

## `[RESULTS PLACEHOLDER]`

### 6. Results

`[To be filled after Notebook 7 evaluation completes. Sections: 6.1 In-Domain Results (tab05, fig15), 6.2 Cross-Dataset Generalization (tab06, fig11), 6.3 Adversarial Robustness (tab08, fig10), 6.4 Zero-Day Detection (tab09, fig12), 6.5 Inference Latency (tab10)]`

### 7. Ablation Study

`[To be filled after Notebook 7 ablation completes. Discuss tab07 component by component, connecting each result to the architectural decision it justifies.]`

### 8. Explainability Analysis

`[To be filled after Notebook 7 XAI completes. Sections: 8.1 SHAP Feature Importance (fig13, tab11), 8.2 Attention Visualization (fig14), 8.3 Cross-Validation of SHAP and Attention, 8.4 Tie-Back to Per-Class Threshold Calibration]`

---

## 9. Limitations

We identify the following limitations of this work:

**Flow-level temporal granularity:** As discussed in Section 3.4, NF3 timestamps are flow-level, not packet-level. Time2Vec encoding captures flow ordering and inter-flow temporal patterns, but sub-flow packet timing dynamics are lost. A packet-level (pcap) implementation of the same architecture would likely extract richer temporal signal and is an important direction for future work.

**Compute constraints:** All experiments were conducted on dual T4 GPUs (16 GB VRAM each). While this is sufficient for the architecture described, scaling to larger graphs (e.g., backbone-network traffic at 100M+ flows) or deeper encoders would require additional GPU memory or distributed training, which we do not evaluate.

**Out-of-schema generalization:** Our out-of-schema evaluation depends on feature mapping between the NF3 schema and raw CIC-format datasets. Features that cannot be mapped are dropped, which may disadvantage performance on those datasets relative to a model trained natively on CIC-format data. `[Additional limitation: note if CIC-DDoS2019/Darknet2020 were unavailable.]`

**Taxonomic ambiguity:** The merging of Fuzzers into Exploits and Analysis into Reconnaissance, while behaviorally motivated, necessarily loses some granularity. A deployment in an environment where fuzzing attacks are a primary concern might prefer to keep these classes separate if sufficient training data exists.

**Hyperparameter sensitivity:** While we report a complete hyperparameter table and use consistent seeds, we do not conduct a formal hyperparameter sensitivity analysis across all stages — the compute budget on T4x2 constrains this. The reported results represent a single well-tuned configuration rather than a distribution over hyperparameter samples.

`[Additional limitations to be added based on incident log in 07_WORK_COMPLETION.md and any metric shortfalls.]`

---

## 10. Conclusion

`[To be written after results are finalized. Restate contributions, summarize headline findings, no new claims.]`

---

## References

`[CITATION NEEDED: Complete verified reference list — all citations below require DOI/author/venue verification]`

1. [CITATION NEEDED: TE-G-SAGE — temporal edge graph SAGE NIDS paper]
2. [CITATION NEEDED: Time2Vec — Kazemi et al., "Time2Vec: Learning a Vector Representation of Time"]
3. [CITATION NEEDED: GATv2 — Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022]
4. [CITATION NEEDED: ARGANIDS — adversarial GAN-based NIDS paper]
5. [CITATION NEEDED: Problem-space structural adversarial attacks against GNN-NIDS]
6. [CITATION NEEDED: IoT GNN + FGSM/DeepFool evaluation paper]
7. [CITATION NEEDED: Prototypical networks — Snell et al., NeurIPS 2017]
8. [CITATION NEEDED: Prototypical capsule network for NIDS]
9. [CITATION NEEDED: CSCVAE-NID — conditional VAE for NIDS class imbalance]
10. [CITATION NEEDED: MalMoE — multi-expert NIDS paper]
11. [CITATION NEEDED: Cross-domain heterogeneous ensemble NIDS paper]
12. [CITATION NEEDED: PPT-GNN — edge-feature-aware GNN for NIDS]
13. [CITATION NEEDED: StrGNN — structural GNN for NIDS]
14. [CITATION NEEDED: TGN — temporal graph networks, Rossi et al.]
15. [CITATION NEEDED: REAL-IoT — cross-dataset IoT NIDS evaluation]
16. [CITATION NEEDED: Sarhan et al. — standard NetFlow feature set for NIDS]
17. [CITATION NEEDED: NF3 dataset paper — UQ eSpace NetFlow v3]
18. [CITATION NEEDED: CSE-CIC-IDS2018 dataset paper]
19. [CITATION NEEDED: UNSW-NB15 dataset paper — Moustafa et al.]
20. [CITATION NEEDED: ToN-IoT dataset paper — Moustafa et al.]
21. [CITATION NEEDED: BoT-IoT dataset paper — Koroniotis et al.]
22. [CITATION NEEDED: CIC-DDoS2019 dataset paper]
23. [CITATION NEEDED: CIC-Darknet2020 dataset paper]
24. [CITATION NEEDED: Survey/meta-review documenting lack of cross-dataset evaluation in NIDS]

---

*This draft was prepared as part of a staged NIDS research project. All placeholder results will be filled from experimental outputs. No cited reference has been fabricated — [CITATION NEEDED] markers indicate references whose exact DOI, author list, and venue require verification before submission.*
