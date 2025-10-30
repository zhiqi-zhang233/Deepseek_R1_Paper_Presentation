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

![Figure: DeepSeek-R1 Evolution Overview](images/r1_evolution_diagram.png)

#### Model Performance
- DeepSeek-R1-Zero, achieves **71% → 86.7%** (with majority voting) on AIME 2024, rivaling OpenAI-o1-0912.  
- DeepSeek-R1, using a hybrid SFT+RL pipeline, reaches 79.8% on AIME and **95.6%** on MATH-500, surpassing o1-1217.  
- Distilled versions, such as R1-Distill-Qwen-14B, maintain similar reasoning ability while cutting inference cost dramatically.  
---

## Model Architecture

### DeepSeek-R1-Zero & Group Relative Policy Optimization (GRPO)

DeepSeek introduces **GRPO**, a modification of PPO that removes the need for a critic network.

**Key Idea:**  
Each batch of responses to the same question is evaluated **relatively**—the model learns from the *best responses in its own group*.

\[
A_i = \frac{r_i - \bar{r}}{s_r}
\]

\[
L = \mathbb{E}_i[\min(r_i(\theta)A_i, \text{clip}(r_i(\theta), 1 - \epsilon, 1 + \epsilon)A_i)] - \beta \cdot KL(\pi_\theta \| \pi_{\text{old}})
\]

---

### 2.2 Reward Design

- **Accuracy Reward:** checks correctness of final answer (e.g., math results, code outputs).  
- **Format Reward:** enforces clean reasoning structure within `<think>` … `</think>` tags.  
- **Language Reward:** ensures single-language reasoning (avoids English–Chinese mixing).  

![Figure: RL Training Loop](images/grpo_training_loop.png)  
> **Figure Suggestion:** show RL loop with *prompt → multiple responses → reward → policy update*.

---

### 2.3 R1-Zero’s Self-Evolution & “Aha Moment”

Without any supervision, **R1-Zero** starts producing longer and more structured reasoning chains.  
It *spontaneously learns* to reflect, revise steps, and even express realization moments:

> *“Wait… that seems wrong. Let’s reconsider.”*

This behavior emerges **solely from reward signals**, not from any human example —  
a genuine *emergent reasoning phenomenon*.

![Figure: Aha Moment Example](images/aha_example.png)  
> **Figure Suggestion:** side-by-side text showing model’s initial wrong path and its self-correction using reflection.

---

### 2.4 R1: Cold Start and Multi-Stage RL

R1 builds upon R1-Zero by introducing a **multi-stage hybrid training pipeline**:

1. **Cold Start (SFT):** a few thousand curated long-CoT samples initialize stability and readability.  
2. **Reasoning-Oriented RL:** large-scale RL enhances reasoning accuracy and depth.  
3. **Rejection Sampling Fine-Tuning:** filters the best RL outputs for stable supervised learning.  
4. **Final RL for All Scenarios:** includes helpfulness and harmlessness alignment.

![Figure: Multi-Stage Training Pipeline](images/multistage_pipeline.png)
> **Figure Suggestion:** a 4-step pipeline diagram from SFT → RL → RSFT → RL, each labeled with objectives.

---

### 2.5 Distillation: Teaching Smaller Models to Think

R1’s final contribution is **distillation** — transferring reasoning knowledge to smaller, dense models like **Qwen** and **Llama** (1.5B–70B).  

These distilled models maintain high reasoning accuracy while cutting compute cost dramatically.  
For instance, **R1-Distill-Qwen-14B** matches or surpasses **o1-mini (OpenAI)** in math and logic benchmarks.

---

## 📊 3. Experimental Results

| Benchmark | o1-mini | R1-Zero | R1 | R1-Distill-Qwen-14B |
|------------|----------|----------|----------|----------------------|
| **AIME 2024 (pass@1)** | 63.6 | 71.0 | **79.8** | 69.7 |
| **MATH-500 (pass@1)** | 90.0 | 95.9 | **97.3** | 94.3 |
| **GPQA Diamond** | 60.0 | 73.3 | **71.5** | 59.1 |
| **Codeforces Rating** | 1820 | 1444 | **2029** | 1481 |

> **Insight:** R1’s combination of cold start + RL outperforms pure-RL R1-Zero, showing the importance of hybridization.

---

## 🔍 4. Critical Analysis

| Strengths | Limitations |
|------------|-------------|
| ✅ Demonstrates *pure RL reasoning emergence* for the first time. | ❌ High compute cost; GRPO is efficient but still expensive. |
| ✅ Multi-stage hybrid pipeline stabilizes long reasoning chains. | ❌ Some outputs remain verbose or language-mixed. |
| ✅ Distillation enables lightweight reasoning models. | ❌ Reward design still partly handcrafted; lacks interpretability. |
| ✅ Emergent self-reflection reveals meta-cognition potential. | ❌ General-domain reasoning (non-STEM) underexplored. |

---

## 🌍 5. Impacts and Significance

**Scientific Impact:**  
DeepSeek-R1 proves that reasoning can *emerge naturally* from reinforcement learning, opening new directions in AI cognition research.

**Technological Impact:**  
Distillation makes powerful reasoning affordable — enabling 7B–14B models to achieve performance comparable to 70B+.

**Ethical Impact:**  
Hybrid RL training introduces better control over reasoning transparency, helping reduce hallucination via structured `<think>` segments.

**Research Landscape Shift:**  
DeepSeek-R1 positions open-source research to directly compete with proprietary reasoning models like OpenAI’s o1.

![Figure: Reasoning Landscape Shift](images/ai_reasoning_landscape.png)
> **Figure Suggestion:** a timeline showing GPT-4 → o1 → DeepSeek-R1 → R1-Distill, marking the open-source leap.

---

## ❓ 6. Audience Questions

**Question 1:**  
> Why did the researchers move from R1-Zero (pure RL) to R1 (SFT + RL hybrid)?  
> What trade-offs did this solve?

**Question 2:**  
> How does GRPO eliminate the critic network, and why does this matter for scaling RL in LLMs?

*(Give classmates 1 minute to discuss before explaining your perspective.)*

---

## 🧠 7. Key Takeaways

1. **DeepSeek-R1-Zero** — demonstrated emergent reasoning through *pure reinforcement learning*.  
2. **DeepSeek-R1** — hybridized SFT and RL, greatly improving readability, stability, and reasoning depth.  
3. **DeepSeek-R1-Distill** — transferred reasoning capability into *smaller open models*, achieving scalability and accessibility.  
4. **Core Lesson:** RL can *incentivize reasoning* when guided by structured rewards — no explicit reasoning supervision required.

---

## 📚 8. Resource Links

| Type | Link |
|------|------|
| 📄 Paper | [arXiv:2501.04508](https://arxiv.org/abs/2501.04508) |
| 💻 DeepSeek GitHub | [https://github.com/deepseek-ai](https://github.com/deepseek-ai) |
| 🧮 Benchmarks | [AIME 2024 Dataset](https://maa.org/math-competitions/aime) |
| 🧩 Related Work | [OpenAI o1 Report](https://openai.com/research/o1) |
| 🧠 Technical Blog | [DeepSeek-R1 Overview](https://medium.com/@deepseek-ai) |

---

## 🧾 9. Citation
> DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in Large Language Models via Reinforcement Learning.* arXiv preprint arXiv:2501.04508 (2025).

---

## 🏁 10. Summary Slide (Visual Suggestion)

![Figure: Summary Slide Visual](images/r1_summary_slide.png)
> **Figure Suggestion:**  
> Three blocks horizontally aligned —  
> **R1-Zero:** Emergent reasoning through RL  
> **R1:** Hybrid SFT + RL with cold-start data  
> **R1-Distill:** Efficient small models with transferred reasoning  

