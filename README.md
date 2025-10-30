# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**Paper:** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
\
**Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai (Google Research)  
**Conference:** EMNLP 2023  
**ArXiv:** https://arxiv.org/abs/2305.13245

**Presented by:** Zhiqi(Camille) Zhang
**Date:** Thursday, October 30, 2025
---

## 1. Introduction and Motivation

Large Language Models (LLMs) have shown remarkable performance in text generation, yet **true reasoning** remains their weakest point.  
Most reasoning models rely heavily on **Supervised Fine-Tuning (SFT)** with labeled *Chain-of-Thought (CoT)* data, which is expensive and limits the model’s ability to generalize.

The **DeepSeek-R1 project** rethinks this paradigm.  
It explores a bold question:

> 🧠 *Can reasoning ability emerge purely through Reinforcement Learning (RL), without supervision?*

---

### From R1-Zero to R1 — The Evolution of Reasoning

| Stage | Model | Core Idea | Key Features |
|--------|--------|------------|---------------|
| **1** | **R1-Zero** | *Reasoning through pure RL (no supervision)* | GRPO algorithm; rule-based rewards; “aha moment” emergent reasoning |
| **2** | **R1** | *Hybrid multi-stage pipeline (SFT + RL)* | Cold-start data, multi-round RL fine-tuning, readability and human alignment |
| **3** | **R1-Distill** | *Transferring reasoning to smaller models* | Knowledge distillation to Qwen & Llama (1.5B–70B) |

The development from **R1-Zero → R1 → R1-Distill** mirrors the journey from *raw emergent intelligence* to *refined, teachable reasoning*.

---

![Figure: DeepSeek-R1 Evolution Overview](images/r1_evolution_diagram.png)
> **Figure Suggestion:**  
> A flow diagram showing the 3 stages:  
> **R1-Zero (pure RL)** → **R1 (multi-stage hybrid)** → **R1-Distill (small models)**,  
> annotated with key techniques (GRPO, rejection sampling, distillation).

---

## ⚙️ 2. Technical Approach

### 2.1 Group Relative Policy Optimization (GRPO)

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

