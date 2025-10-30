# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
---
**Paper**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning  
**Authors:** Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song (DeepSeek Research Team)    
**ArXiv:** [https://arxiv.org/abs/2305.13245  ](https://arxiv.org/abs/2501.12948)  
**Presented by:** Zhiqi(Camille) Zhang  
**Date:** Thursday, October 30, 2025  

## Overview
---
### Context
Imagine you are solving a tough math problem. At first, you try a direct approach, fail, then pause, reflect, and rethink your strategy. That little internal dialogue like “Wait, that doesn’t seem right… let’s try another way” is what we call **reasoning**.  

Modern language models are beginning to mimic this process. OpenAI recently introduced the idea of **inference-time scaling**, where a model can “think longer” by extending its **Chain-of-Thought (CoT)** during test time. The longer the reasoning chain, the more likely the model is to reach a correct, verifiable conclusion.  

This simple trick letting models reason for more steps has brought dramatic improvements across domains such as mathematics, programming, and scientific reasoning.  
However, it also raises a deeper question:  

> *Can a model learn when and how to think—without humans manually scripting its reasoning patterns?*
---
### Problem Statement: The Challenge of Effective Test-Time Reasoning
Despite the progress of inference-time scaling, the **research community still lacks an effective method** to control and extend reasoning at test time. The difficulty lies in how to train models that can *self-decide* when to explore deeper reasoning paths.

Previous efforts have tried several directions:
- **Process-based Reward Models (PRM):** assign feedback for each reasoning step,  but they require fine-grained annotation and often fall into *reward hacking*.
- **Search Algorithms (e.g., MCTS or Beam Search):** imitate AlphaGo’s exploration strategy, yet token-level search spaces make them computationally infeasible at scale.
- **Hybrid Reinforcement Learning Approaches:** combine supervised labels and RL signals, but they depend heavily on expensive labeled data.

As these methods struggle, researchers began to ask:  
> “What if reasoning could *emerge naturally* through reinforcement learning alone?”

---
### How the problem was Addressed: From R1-Zero to R1 to R1-Distill
> Why Post-Training?  
> Improved reasoning accuracy, human-aligned behavior, efficiency

This paper answers that question by designing a **comprehensive post-training pipeline** centered on **pure Reinforcement Learning (RL)**, a process that allows a pre-trained language model to refine its reasoning ability without additional supervised data. At the heart of this work implemented through a custom algorithm called **Group Relative Policy Optimization (GRPO)**. A remarkable finding is that, even **without any supervised fine-tuning (SFT)**, the model can self-evolve into a reasoning agent purely through RL feedback. The base model (DeepSeek-V3-Base) starts with general linguistic capability but no explicit reasoning structure.  

| Stage | Model | Core Mechanism | Strengths | Limitations |
|:------|:-------|:---------------|:-----------|:-------------|
| **1** | **DeepSeek-R1-Zero** | Applies pure Reinforcement Learning on DeepSeek-V3-Base using GRPO | Emergent self-reflection, spontaneous long CoT reasoning, major benchmark gains | Low readability, language mixing |
| **2** | **DeepSeek-R1** | Adds cold-start SFT data + multi-stage RL pipeline | Improves coherence and stability, matches OpenAI o1-1217 performance | Higher training cost |
| **3** | **DeepSeek-R1-Distill** | Distills R1’s reasoning into smaller dense models (Qwen / Llama) | Efficient, low-cost reasoning comparable to larger models | Slight drop on hardest benchmarks |

![image](image1.png)

#### Model Performance
- DeepSeek-R1-Zero, achieves **71% → 86.7%** (with majority voting) on AIME 2024, rivaling OpenAI-o1-0912.  
- DeepSeek-R1, using a hybrid SFT+RL pipeline, reaches 79.8% on AIME and **95.6%** on MATH-500, surpassing o1-1217.  
- Distilled versions, such as R1-Distill-Qwen-14B, maintain similar reasoning ability while cutting inference cost dramatically.  
---

## Model Architecture

### DeepSeek-R1-Zero: Group Relative Policy Optimization (GRPO)
**Goal:**  
Teach a pre-trained model (DeepSeek-V3-Base) to reason logically without any supervised labels.

**Key Idea:**  
Instead of learning from human-annotated reasoning chains, the model learns through reward signals that tell it which answers are better.

**Components Used:**
- **GRPO (Group Relative Policy Optimization)** — a lightweight reinforcement learning algorithm.  
- **Reward Model** — evaluates whether each answer is accurate and well-formatted.  
---

#### GRPO

```text
Input: Pre-trained model πθ (DeepSeek-V3-Base)
Repeat for each batch of prompts Q:
    For each question q in Q:
        1. Generate G possible answers {o₁, o₂, ..., o_G}
        2. Compute reward for each answer:
            - Accuracy reward: Is the answer correct?
            - Format reward: Does it use <think>...</think> structure?
        3. Compute relative advantage Aᵢ = (rᵢ - mean(r)) / std(r)
        4. Update model parameters θ using:
               J(θ) = E[min(r(θ)*A, clip(r(θ), 1−ε, 1+ε)*A)] − β * KL(πθ || πref)
Output: Updated model πθ'
````
The model generates multiple answers, compares them to each other, learns which one is “better,” and updates itself accordingly.

#### Reward Model
In DeepSeek-R1-Zero, the reward model replaces human supervision by automatically scoring each generated answer. It uses two simple rule-based signals: an **accuracy reward**, which checks if the final answer is correct, and a **format reward**, which ensures the reasoning process follows the `<think>...</think>` structure. Unlike neural reward models that require human feedback and risk reward hacking, this rule-based setup is lightweight, objective, and scalable. Together with GRPO, it lets the model learn from its own outputs—gradually reinforcing clearer, more logical reasoning without any labeled data.

#### Emergent “Aha Moment”

During training, DeepSeek-R1-Zero began to display fascinating self-reflective reasoning behaviors.
At times, it paused, questioned its own logic, and revised previous steps just like a human realizing a mistake.

Example:

> “Wait, that seems wrong. Let’s check the equation again…”

This spontaneous self-correction, called the **“Aha Moment”**, was not programmed. It emerged naturally from reinforcement learning.

![image](image2.png)
---

#### Outcome

- **Emergent Reasoning:** Model develops structured, multi-step logical reasoning.
- **Performance:** On AIME 2024, R1-Zero improved from 15.6% → **71%** pass@1 accuracy.
- **Limitation:** Outputs were sometimes verbose or mixed languages; reasoning lacked readability.

---

### DeepSeek-R1: Hybrid SFT + RL

**Goal:**
Make R1-Zero’s powerful reasoning more readable, aligned, and human-like.

---

#### Training Enhancements

1. **Cold Start (Supervised Fine-Tuning):**
   Use a few thousand manually curated long reasoning examples to initialize the model.

   * Helps stabilize early RL learning.
   * Introduces clear reasoning format and readable English.

2. **Reinforcement Learning with Cold Start:**
   Apply the same GRPO process, but now starting from a stable checkpoint.

   * Encourages deeper, more coherent reasoning chains.
   * Reduces nonsensical outputs.

3. **Rejection Sampling:**
   After RL, only the best response* (by reward score) are kept for further fine-tuning.

   * Prevents overfitting to bad reasoning samples.
   * Sharpens logical accuracy.

#### Why It Works Better

By alternating between **SFT (human clarity)** and **RL (self-improvement)**, R1 finds a balance between human-like reasoning structure and machine-level consistency.It not only reasons well, but explains why it reached an answer in an interpretable `<think>` section.

---

#### Outcome

* **Performance:** AIME 2024: **79.8%**, MATH-500: **95.6%**, rivaling OpenAI o1-1217.
* **Behavior:** Structured, consistent, and interpretable reasoning steps.
* **Limitation:** Requires multiple fine-tuning stages; still resource-intensive.

---

### DeepSeek-R1-Distill: Teaching Smaller Models to Reason

**Goal:**
Transfer the reasoning ability of large R1 models to smaller dense models like **Qwen** and **Llama**, making reasoning affordable and accessible.

---

#### Distillation Process

1. **Teacher:** DeepSeek-R1 (full model).
2. **Students:** Smaller dense models (Qwen-1.5B, Qwen-14B, Llama-8B, Llama-70B).
3. **Method:** Generate ~800k reasoning samples with `<think>` and `<summary>` sections.

   * These serve as *teaching material* for the smaller models.
4. **Training:** Student models learn to imitate both *the reasoning structure* and *final answers*.

#### 📈 Effectiveness

| Model                            | AIME 2024 | MATH-500 | GPQA Diamond | Codeforces |
| -------------------------------- | --------- | -------- | ------------ | ---------- |
| **GPT-4o**                       | 9.3       | 74.6     | 49.9         | 32.9       |
| **DeepSeek-R1-Distill-Qwen-14B** | **69.7**  | **94.3** | **59.1**     | **53.1**   |

This shows that distilled models, though smaller, **outperform many larger non-reasoning LLMs** like GPT-4o.


