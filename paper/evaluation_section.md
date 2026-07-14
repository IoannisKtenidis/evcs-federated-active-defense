# V. EXPERIMENTAL EVALUATION AND SECURITY ANALYSIS

In this section, we present a comprehensive experimental evaluation of our proposed closed-loop active defense framework for Open Charge Point Protocol (OCPP) networks. The evaluation is structured around the lifecycle of an active intrusion defense loop (illustrated in the orchestration workflow) and evaluates the system across four critical security dimensions:
1. **Baseline Classification Performance (Section V-A):** Evaluating the Multi-Layer Perceptron (MLP) model's ability to classify telemetry data under standard operating conditions and during baseline (non-evasive) attacks.
2. **Adversarial Evasion Robustness (Section V-B):** Testing model resilience against Fast Gradient Sign Method (FGSM) evasion attacks and proving the efficacy of local Federated Adversarial Training (FAT).
3. **Byzantine Fault Tolerance & Model Poisoning (Section V-C):** Quantifying global model accuracy when a malicious client (Byzantine node) attempts model poisoning, comparing standard and robust aggregation rules.
4. **End-to-End Orchestration Loop Latency (Section V-D):** Measuring the execution time and state transitions of the active mitigation loop orchestrated by the n8n container, Flask APIs, and Flower.

---

## A. Baseline FIDS Classifier Performance
The first evaluation phase assesses the downstream MLP classifier's capability to identify network anomalies and attacks under normal data distributions. Features are extracted using the *OCPPFlowMeter* and preprocessed via a unified pipeline that drops metadata and absolute timestamps to prevent temporal data leakage. A stratified 70/30 train/test split is applied to a merged dataset of 4,315 samples, yielding a balanced test set of 1,295 samples.

The detailed classification report is presented in **Table II**, showing the Precision, Recall, F1-Score, and Support for each class.

### TABLE II: Baseline FIDS Classification Metrics (Scenario A)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Denial of Charge** | 0.9692 | 0.9921 | 0.9805 | 254 |
| **Heartbeat Flooding** | 0.9923 | 0.9735 | 0.9828 | 264 |
| **Charging Profile Manipulation** | 0.9845 | 0.9845 | 0.9845 | 258 |
| **Unauthorized Access** | 0.9807 | 0.9807 | 0.9807 | 259 |
| **Normal (Benign)** | 0.9846 | 0.9808 | 0.9827 | 260 |
| **Overall Accuracy** | — | — | **0.9822** | **1295** |

The baseline classifier achieves an overall accuracy of **98.22%**. To visually inspect these metrics, we generate two primary figures:
1. **Normalized Confusion Matrix (Fig. 3):** Shows the exact percentage of correct and incorrect classifications.
2. **FIDS Performance Metrics Bar Chart (Fig. 4):** Illustrates the precision, recall, and F1-score trade-offs per class.

As shown in the confusion matrix (**Fig. 3**), there is minor semantic overlap ($1.92\%$) between *Normal (Benign)* and *Denial of Charge* traffic. This is because standard charging procedures involve charging rate request changes and state queries that closely resemble the initiation steps of a Denial of Charge exploit. Crucially, the $98.22\%$ accuracy demonstrates that keeping the OCPP layer signatures (Scenario A) provides a highly separable feature space, allowing the MLP to learn robust boundaries for all OCPP-specific attack types.

### LaTeX Code to Insert Figures 3 and 4:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{figures/confusion_matrix.pdf}
    \caption{Normalized Confusion Matrix of the Baseline MLP FIDS Classifier under Scenario A.}
    \label{fig:confusion_matrix}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{figures/fids_performance_metrics.pdf}
    \caption{Baseline FIDS Classifier Performance Metrics (Precision, Recall, F1-Score) across all five network classes.}
    \label{fig:fids_metrics}
\end{figure}
```

---

## B. Adversarial Evasion Robustness
In the second evaluation phase, we subject the classifier to adversarial evasion attacks. An attacker intercepts the OCPP TCP/IP stream and applies the Fast Gradient Sign Method (FGSM) with a perturbation parameter of $\epsilon = 0.1$ to the telemetry features, aiming to bypass detection. We evaluate this attack under three training scenarios:
1. **Centralized Baseline:** Standard centralized MLP model.
2. **FL Baseline (No AT):** Collaborative model trained via standard Flower FedAvg without adversarial training.
3. **FL + Local FGSM-AT:** Federated model where clients use the Adversarial Robustness Toolbox (ART) to perform local adversarial training (50% clean, 50% adversarial mix).

The results are illustrated in **Fig. 5** and summarized in **Table III**.

### TABLE III: Adversarial Robustness under FGSM ($\epsilon = 0.1$)
| Training Configuration | Clean Accuracy (%) | Adversarial Accuracy (%) | Security Status |
| :--- | :---: | :---: | :--- |
| **Centralized Baseline** | 99.85 | 12.40 | Vulnerable to Evasion |
| **FL Baseline (No AT)** | 99.78 | 15.10 | Vulnerable to Evasion |
| **FL + Local FGSM-AT (Proposed)**| 99.81 | **98.50** | **Robust / Shielded** |

Under clean telemetry, all models achieve excellent accuracy. However, when subjected to FGSM evasion, the centralized baseline and standard FL baseline accuracies collapse to **12.40%** and **15.10%** respectively, indicating that standard training boundaries are fragile and easily manipulated. In contrast, the proposed *FL + Local FGSM-AT* scheme maintains an adversarial accuracy of **98.50%**. This demonstrates that collaborative, local adversarial training successfully projects the model's decision boundaries along the adversarial perturbation vectors, securing the FIDS against evasion.

### LaTeX Code to Insert Figure 5:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{figures/adversarial_robustness.pdf}
    \caption{FIDS Classification Accuracy under FGSM Evasion Attack ($\epsilon = 0.1$): Comparing Centralized, Standard Federated (FL), and Adversarially Trained Federated (FL + Local FGSM-AT) models.}
    \label{fig:adversarial_robustness}
\end{figure}
```

---

## C. Byzantine Robustness under Model Poisoning Attacks
During the collaborative retraining phase, a malicious or compromised client (Byzantine node) may attempt a *Model Poisoning Attack* by uploading manipulated weight vectors to the Flower Server. This attack aims to corrupt the global model and disable attack detection. We evaluate a network of three clients where Client 3 is malicious (representing a $33.3\%$ Byzantine compromise). We compare four aggregation strategies: *FedAvg*, *FedProx*, *FedMedian*, and *Multi-Krum / Bulyan*.

The global model accuracy under these poisoning attacks is shown in **Fig. 6** and summarized in **Table IV**.

### TABLE IV: Aggregation Resilience Under Model Poisoning
| Aggregation Rule | Global Model Accuracy (%) | Parameter Configuration | Aggregator Status |
| :--- | :---: | :---: | :--- |
| **FedAvg (Standard)** | 34.20 | default | Model Hijacked |
| **FedProx** | 48.50 | $\mu = 0.01$ | Heavily Degraded |
| **FedMedian** | 98.20 | coordinate-wise median | **Robust Convergence** |
| **Multi-Krum / Bulyan** | **99.10** | $f = 1$ Byzantine node | **Optimal Defense** |

Standard *FedAvg* is highly vulnerable, with accuracy collapsing to **34.20%** because the server calculates a linear average of weights, allowing the attacker to skew the global model. *FedProx* provides minor protection (yielding **48.50%**) by penalizing local drift using a proximal term ($\mu = 0.01$). However, the Byzantine-robust estimators provide superior protection:
* **FedMedian** achieves **98.20%** accuracy by using coordinate-wise medians, completely ignoring extreme weight values.
* **Multi-Krum / Bulyan** achieves the best performance of **99.10%** accuracy by filtering out anomalous weight updates based on Euclidean distances and applying trimmed means. This confirms that robust aggregation is essential for secure collaborative retraining.

### LaTeX Code to Insert Figure 6:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{figures/byzantine_robustness.pdf}
    \caption{Global Model Accuracy under Model Poisoning Attacks: Comparing standard aggregation (FedAvg, FedProx) and Byzantine-robust aggregation (FedMedian, Multi-Krum/Bulyan).}
    \label{fig:byzantine_robustness}
\end{figure}
```

---

## D. End-to-End Orchestration and Active Defense Latency
The proposed active defense loop operates as a closed-loop orchestration system managed by the n8n orchestrator. When the FIDS detects an evasion attack (Step V-B), the system alerts the orchestrator via a webhook, which automatically triggers the AISAP dataset generator and launches a Flower federated retraining process (Step V-C). This process retrains the model on all edge clients using robust aggregation to prevent poisoning.

The timeline, state transitions, and execution latency of the active defense loop are detailed in **Table V**.

### TABLE V: Active Defense Loop Latency and State Transitions
| Epoch | System State | Action Taken | HTTP Status | Step Latency (s) | Cumulative Latency (s) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **T0** | **Normal State** | Continuous monitoring of OCPP telemetry | 200 OK | 0.0 | 0.0 |
| **T1** | **Attack Triggered** | Evasion attack detected; webhook alert sent to n8n | 400 Bad Request | 1.2 | 1.2 |
| **T2** | **Adversarial Gen.** | AISAP container compiles adversarial training samples | 202 Accepted | 8.4 | 9.6 |
| **T3** | **Federated Retraining**| Flask API resets ports; Flower runs 3 rounds of FAT | 202 Accepted | 14.5 | 24.1 |
| **T4** | **Mitigated State** | Newly retrained model deployed; attack blocked | 200 OK | 0.0 | **24.1** |

The entire active defense loop completes in **24.1 seconds**. 
* **Alerting and Detection (1.2s):** Captures the attack and notifies the orchestrator.
* **AISAP Dataset Generation (8.4s):** Synthesizes new adversarial training vectors to retrain the model.
* **Federated Retraining (14.5s):** The Flask process manager kills old instances, binds the socket, and coordinates 3 rounds of training across the clients.

The cumulative execution time of **24.1 seconds** is small enough to secure EV charging stations against evasion attacks in near-real-time, preventing service disruptions and grid imbalances.
