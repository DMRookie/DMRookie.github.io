---
title: 视觉Token"失声"：揭秘多模态大模型为何"睁眼说瞎话"
date: 2026-03-10 19:19:00
categories: 视觉大模型
tags: 大模型 多模态
---

## 1. 摘要

多模态大语言模型（Multimodal Large Language Models, MLLM）近年来取得了突破性进展，然而其在精细视觉理解任务上的表现仍存在显著瓶颈。本文系统性地梳理了2024-2025年，arXiv上关于MLLM视觉Token监督不足问题的最新研究。研究表明，当前主流MLLM采用的"仅文本监督"训练范式导致视觉路径处于欠监督状态，进而引发视觉注意力退化、表征学习不充分及视觉遗忘等一系列问题[1][2]。本文将现有解决方案归纳为三大类别：训练时方法（辅助损失函数、强化学习）、推理时方法（动态干预、注意力调制）及架构改进方法（特征重采样、模块化设计），并对20余篇代表性论文进行了详细分析。

## 2. 研究背景

### 2.1 多模态大模型的发展现状

随着大语言模型（LLM）技术的成熟，研究者们致力于将其强大的语言理解与生成能力扩展至多模态领域。以LLaVA、GPT-4V、Gemini为代表的多模态大语言模型通过引入视觉编码器（如CLIP、SigLIP）将图像信息转化为Token序列，再与文本Token拼接后输入语言模型进行联合处理。这种架构设计在视觉问答、图像描述等任务上取得了令人瞩目的成果（架构演化可参考《[从QwenVL与InternVL的演进看多模态大模型的范式收敛](https://dmrookie.github.io/2025/10/16/%E4%BB%8EQwenVL%E4%B8%8EInternVL%E7%9A%84%E6%BC%94%E8%BF%9B%E7%9C%8B%E5%A4%9A%E6%A8%A1%E6%80%81%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%8C%83%E5%BC%8F%E6%94%B6%E6%95%9B/) 》）。

然而，深入的实证研究揭示了一个关键问题：尽管MLLM在通用视觉对话场景中表现优异，但在需要精细视觉感知的任务（如物体计数、空间关系推理、OCR文字识别、图表理解等）上仍存在明显不足[1][5]。这一现象的根源在于当前主流训练范式对视觉路径的监督不足。

### 2.2 "仅文本监督"范式的局限性

当前绝大多数MLLM采用"仅文本监督"（Text-only Supervision）的训练策略。在这种范式下，模型的训练目标仅为预测下一个文本Token，视觉信息的学习完全依赖于文本生成任务的间接反馈。具体而言，给定图像-文本对，模型的损失函数通常定义为：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, I)$$

其中$I$为输入图像，$y_t$为第$t$个文本Token。这种设计导致视觉编码器和投影层仅能获得来自语言模型的稀疏梯度信号，难以学习到丰富的视觉表征[3][5]。

## 3. 问题分析

针对MLLM视觉感知能力不足的问题，研究者们从多个维度进行了深入剖析。本节将这些问题归纳为四个核心类别，并阐述其产生机制与具体表现。

### 3.1 视觉Token监督不足

视觉Token监督不足是导致MLLM视觉能力受限的根本原因。在标准的仅文本监督训练范式中，视觉编码器生成的Token序列需要经过投影层转换后才能进入语言模型，而训练损失仅作用于输出端的文本预测。这意味着视觉路径的优化完全依赖于从语言模型反向传播的梯度，其监督强度和信息密度远低于直接监督[1][5]。

研究表明，这种间接监督机制导致模型倾向于丢弃与当前文本生成任务不直接相关的视觉细节。例如，当回答"图中有几只猫？"时，模型可能只关注猫的存在性而忽略其精确数量；当描述一张复杂场景图像时，模型往往只捕获最显著的视觉元素而遗漏背景细节[1]。

### 3.2 视觉注意力不足与退化

视觉忽视（Visual Neglect）现象是指MLLM在推理过程中过度依赖语言先验（Language Priors）而忽略输入图像信息的倾向。研究发现，模型在面对视觉信息与语言先验产生冲突的场景时，往往选择"相信"其语言知识而非"看到"的内容[2][8]。

更值得关注的是视觉注意力退化（Attention Degradation）问题。实验证据表明，随着生成序列长度的增加，模型对视觉Token的注意力权重呈现显著下降趋势。这种现象在长文本生成任务中尤为突出，是导致物体幻觉（Object Hallucination）的核心原因之一[8][19]。

### 3.3 视觉表征学习不充分

视觉编码器（如CLIP）与语言解码器之间存在显著的模态鸿沟（Modality Gap）。尽管投影层（如线性映射或MLP）的引入旨在桥接这一差距，但简单的投影操作难以充分捕捉高维视觉空间中的复杂语义结构[4][15]。

现有研究指出，经过投影后的视觉Token往往只能表达全局语义信息，而缺乏对局部物体、细粒度属性及空间关系的精确表征能力。这导致模型在处理需要精细视觉理解的任务时表现欠佳[15]。

### 3.4 视觉遗忘

视觉遗忘（Visual Forgetting）是指在指令微调（Instruction Tuning）阶段，预训练视觉编码器中积累的丰富特征表达能力被逐渐削弱甚至破坏的现象[3][14]。

这一问题的产生源于微调阶段的梯度更新对视觉表征空间的过度改造。为了更好地对齐下游任务分布，模型倾向于"遗忘"预训练阶段学习到的通用视觉知识，导致其在分布外（Out-of-Distribution, OOD）场景下的泛化能力显著下降[3]。

## 4. 解决方案分类

针对上述问题，研究者们提出了多种技术方案。本节首先以表格形式呈现解决方案的分类概览，随后分别详细介绍各类方法的核心思想与代表性工作。

### 4.1 方案分类概览

|类别|核心思想|技术手段|代表性研究|
|:---|:---|:---|:---|
|训练时方法|在训练阶段引入额外监督信号强化视觉路径|辅助损失函数、对比学习、互信息最大化、强化学习|VIRAL[1]、JARVIS[4]、VisualLoss[5]、VISTA[13]、SAYO[7]|
|推理时方法|在推理阶段动态调整模型行为以增强视觉感知|注意力干预、动态解码|V-ITI[2]、IKOD[8]|
|架构改进方法|重新设计模型架构以优化视觉信息流|特征重采样、双工注意力、感知Token、梯度解耦|Vision Remember[6]、MODA[10]、Visual Perception Token[11]、MDGD[3]|

### 4.2 训练时方法

训练时方法的核心理念是在标准文本损失之外引入辅助监督信号，直接或间接地强化视觉路径的学习。这类方法的优势在于能够从根本上改善模型的视觉表征能力，但通常需要额外的计算开销和训练数据。

辅助损失函数设计是该类方法的主流技术路线。研究者们提出了多种损失函数形式，包括视觉表征对齐损失（将MLLM内部视觉表征与外部视觉基础模型对齐）[1]、重建损失（要求模型能够从视觉Token重建原始图像信息）[4]、以及互信息最大化损失（增强文本Token与视觉隐藏状态之间的信息关联）[13]。

强化学习框架的引入为视觉监督提供了新的技术路径。通过设计基于视觉定位准确性的奖励函数，可以显式地将视觉注意力纳入优化目标，引导模型在复杂推理过程中保持对视觉信息的关注[7]。

### 4.3 推理时方法

推理时方法的特点是无需修改模型权重，而是通过在推理阶段动态调整模型的内部状态或解码策略来增强视觉感知。这类方法的优势在于即插即用、部署灵活，但其效果受限于预训练模型的固有能力。

视觉忽视检测与干预是该类方法的典型代表。通过设计专门的检测器识别模型何时出现视觉忽视行为，并在检测到异常时通过预存的视觉信息调制激活状态，可以有效缓解视觉忽视问题[2]。

协同解码策略则从解码层面入手。通过将原始输出逻辑值与来自高视觉注意力短序列的逻辑值进行融合，可以在长序列生成过程中维持模型对视觉Token的关注度[8]。

### 4.4 架构改进方法

架构改进方法从模型设计层面重新思考视觉信息的处理流程，旨在构建更适合多模态感知的网络结构。这类方法的改进通常较为根本，但也面临与现有训练范式兼容性的挑战。

特征重采样技术允许视觉Token在模型推理过程中多次重新获取原始编码器的高分辨率特征，有效缓解了Token压缩带来的信息损失问题[6]。模块化双工注意力机制则通过解耦模态对齐与Token混合过程，防止细粒度视觉信息被语言权重淹没[10]。

视觉感知Token的引入使模型具备了自主感知能力，能够像生成文本一样自主触发对特定图像区域的进一步编码，实现了从被动感知到主动感知的转变[11]。

## 5. 代表性论文详解

### 5.1 训练增强与损失函数优化

#### 5.1.1 VIRAL：视觉表征对齐
![image.png](https://km.woa.com/asset/00010002260200cd3b7a97cdaa4f5b01?height=800&width=800)

VIRAL（Visual Representation Alignment）是针对视觉路径监督不足问题提出的解决方案[1]。

**问题诊断**：研究者发现，仅文本监督导致MLLM的视觉路径丢弃了大量精细视觉细节，尤其是与当前文本生成任务不直接相关的属性信息（如纹理、颜色细节等）。

**技术方案**：VIRAL提出将MLLM内部的视觉表征与DINOv2等视觉基础模型（Vision Foundation Model, VFM）的输出进行对齐。通过引入对齐损失，强制MLLM的视觉路径保留VFM所捕获的细粒度属性信息。

**创新点**：该方法充分利用了预训练VFM的优质视觉表征作为监督信号，无需额外标注数据即可显著提升MLLM的精细视觉感知能力。

#### 5.1.2 JARVIS：JEPA启发的视觉增强
![image.png](https://km.woa.com/asset/0001000226020029d92d557c124d4301?height=800&width=800)

JARVIS借鉴了I-JEPA（Image Joint-Embedding Predictive Architecture）的学习范式，为MLLM引入自监督视觉学习机制[4]。

**问题诊断**：模型过度依赖语言先验进行视觉推理，忽视了图像本身的结构和语义规律。

**技术方案**：引入掩码预测损失，要求模型基于部分可见的图像区域预测被遮挡区域的表征。这种预训练任务迫使模型学习图像的内在结构规律，而非简单依赖语言模式。

**创新点**：将JEPA范式成功迁移至MLLM训练框架，实现了视觉自监督学习与多模态对齐的有效结合。

#### 5.1.3 VisualLoss/PerceptLLM：独立视觉表征学习
![image.png](https://km.woa.com/asset/000100022602004e734d28f476456301?height=742&width=812)

VisualLoss（又称PerceptLLM）针对MLLM默认使用语言相关性而非视觉输入进行推理的问题提出了创新解决方案[5]。

**问题诊断**：标准训练下，MLLM的视觉表征与语言表征高度耦合，导致模型难以建立独立的视觉理解能力。

**技术方案**：引入辅助VisualLoss，确保语言模型主干能够构建，独立于文本的丰富图像表征。此外，提出BlankTokens技术，在训练时随机将部分文本Token替换为空白，强制模型更多依赖视觉输入。

**创新点**：首次明确提出视觉-语言表征解耦的训练目标，并设计了简洁有效的实现方案。

#### 5.1.4 VISTA：跨模态互信息最大化
![image.png](https://km.woa.com/asset/00010002260200cdc6b7e368e54c7001?height=400&width=802)

VISTA（Vision-Text Alignment）通过最大化跨模态互信息来强化视觉-文本对齐[13]。

**问题诊断**：标准交叉熵损失仅关注文本预测准确性，未能显式建模视觉信息与文本生成之间的关联。

**技术方案**：在不增加额外模块的前提下，通过最大化文本Token与视觉隐藏状态之间的互信息来强化对齐。具体而言，引入对比学习目标，拉近语义相关的视觉-文本表征对，推远不相关的对。

**创新点**：提出了一种轻量级的视觉监督增强方案，计算开销小且易于与现有训练流程集成。

#### 5.1.5 SAYO：强化学习驱动的视觉注意力优化
![image.png](https://km.woa.com/asset/0001000226030099bee2fe4d0d4fc201?height=576&width=1554)

SAYO将强化学习框架应用到了MLLM视觉注意力的优化过程中[7]。

**问题诊断**：在复杂推理任务中，模型的视觉焦点往往微弱且不稳定，难以持续关注与问题相关的图像区域。

**技术方案**：采用强化学习框架，引入基于区域级视觉注意力的奖励机制。当模型在推理过程中正确关注任务相关的图像区域时给予正向奖励，否则给予惩罚。通过策略梯度方法优化模型的注意力分配策略。

**创新点**：首次将视觉注意力纳入强化学习的奖励设计，实现了优化信号与视觉定位步骤的显式对齐。

### 5.2 推理时干预与动态调整

#### 5.2.1 V-ITI：视觉推理时干预
![image.png](https://km.woa.com/asset/0001000226030095fcae3531d3414801?height=338&width=802)

V-ITI（Visual Inference-Time Intervention）提出了一种轻量级的推理时干预策略[2]。

**问题诊断**：模型在生成过程中频繁出现视觉忽视现象，未能优先处理输入图像信息，导致输出与图像内容不符。

**技术方案**：开发视觉忽视检测器，通过监控模型内部状态识别何时出现视觉忽视模式。仅在检测到异常时，通过预存的视觉信息调制模型激活状态，引导其重新关注视觉输入。

**创新点**：提出"按需干预"的策略，避免了持续干预可能带来的副作用，在效果与效率之间取得良好平衡。

#### 5.2.2 IKOD：图像注意力引导解码
![image.png](https://km.woa.com/asset/00010002260300e9c6853a8bd84d9e01?height=506&width=808)

IKOD（Image Attention-guided Decoding）针对长序列生成中的视觉注意力退化问题提出了协同解码策略[8]。

**问题诊断**：随着生成序列变长，模型对视觉Token的注意力权重急剧下降，导致后续生成内容与图像的关联性减弱。

**技术方案**：在解码阶段，同时运行原始长序列生成和受限短序列生成。将两者的输出逻辑值进行融合，其中短序列由于长度限制通常保持较高的视觉注意力，其信息可以"传染"给长序列输出。

**创新点**：无需任何训练或模型修改，仅通过改变解码策略即可有效抑制视觉幻觉。

### 5.3 架构改进与特征保持

#### 5.3.1 MDGD：模态解耦梯度下降

MDGD（Modality-Decoupled Gradient Descent）针对指令微调过程中的视觉知识遗忘问题提出了梯度层面的解决方案[3]。

**问题诊断**：微调阶段的梯度更新对视觉表征空间造成过度改造，导致预训练阶段学习到的视觉知识被覆盖。

**技术方案**：将任务损失的梯度分解为两个正交分量：一个沿着"视觉漂移方向"（会损害视觉表示），另一个与该方向垂直（不影响视觉表示）。通过移除梯度中损害视觉表示的分量，只保留与视觉漂移正交的部分进行参数更新，确保了模型在学习新任务时不会"侵入"视觉知识的参数空间。

**创新点**：提出了有效秩概念，从优化动力学角度理解并解决视觉遗忘问题，提出了通用的梯度调节框架。

#### 5.3.2 Vision Remember：视觉特征重采样
![image.png](https://km.woa.com/asset/0001000226030047fea5af3ca5466f01?height=254&width=868)

Vision Remember提出在解码器层间插入特征重采样模块，允许视觉Token动态重新获取编码器特征[6]。

**问题诊断**：现有架构中，视觉信息仅在输入端一次性注入，经过多层处理后信息逐渐衰减，导致OCR和图表理解等任务中的细粒度信息丢失。

**技术方案**：在解码器特定层（如中间层）插入特征重采样模块。该模块允许视觉Token重新从视觉编码器获取高分辨率特征，实现视觉信息的"刷新"。

**创新点**：打破了视觉信息单向流动的限制，实现了编码器-解码器之间的双向信息交互。

#### 5.3.3 MODA：模块化双工注意力
![image.png](https://km.woa.com/asset/00010002260300484b3bb861e846ac01?height=504&width=1000)

MODA（Modular Duplex Attention）重新设计了多模态注意力机制[10]。

**问题诊断**：标准的跨模态注意力机制中，模态对齐与Token混合同时进行，导致细粒度视觉信息容易被语言权重淹没。

**技术方案**：提出模块化双工注意力，将模态对齐和Token混合解耦为两个独立的子过程。首先在各模态内部进行自注意力处理，然后通过专门的对齐模块进行跨模态信息交换。

**创新点**：通过解耦设计，确保视觉信息在跨模态融合过程中不被稀释，保留更多细粒度细节。

#### 5.3.4 Visual Perception Token：主动视觉感知
![image.png](https://km.woa.com/asset/000100022603008da3930bb8b440e301?height=450&width=1000)

Visual Perception Token引入了一种全新的主动感知机制[11]。

**问题诊断**：传统MLLM的视觉输入是静态的，模型只能被动接受预处理后的视觉Token，缺乏根据任务需求主动获取更多视觉信息的能力。

**技术方案**：引入特殊的感知Token（Perception Token），包括区域选择Token和重编码Token。当模型在推理过程中生成这些特殊Token时，系统会自动对指定图像区域进行进一步的高分辨率编码并注入序列。

**创新点**：使模型具备了类似人类"注视"的主动感知能力，实现了从被动感知到主动感知的范式转变。

### 5.4 其他重要研究

|论文名称|arXiv ID|核心贡献|
|:---|:---|:---|
|MR-MLLM[12]|2406.15768|通过共享查询融合机制实现感知与理解的相互强化|
|LynX[14]|2410.10491|使用双混合专家（MoE）架构，学习视觉定位的同时防止理解能力退化|
|Slot-MLLM[15]|2505.17726|采用对象中心化Token化，通过扩散解码器重建图像确保Token捕获局部细节|
|UniTok[16]|2502.20321|统一分词器，证明重建损失与理解任务可共存增强|
|Visual Jigsaw[17]|2509.25190|拼图式后训练任务，通过结构化排序增强空间推理|
|Latent Visual Reasoning[18]|2509.24251|在潜在空间进行自回归推理，重建关键视觉Token|
|Devils in Middle Layers[19]|CVPR 2025|揭示中间层是视觉信息处理关键，提出VAR分数检测注意力偏移|
|Hallucination Survey[20]|2404.18930|详尽分类视觉注意力失效导致的各类幻觉及缓解方案|

## 6. 研究趋势与展望

### 6.1 当前研究趋势

通过对上述研究的系统梳理，可以识别出以下几个重要的研究趋势：

**趋势一：从间接监督走向直接监督**。早期工作主要通过改进训练策略间接增强视觉路径，而近期研究（如VIRAL、VisualLoss）开始探索对视觉表征的直接监督方式，取得了更为显著的效果。

**趋势二：从静态架构走向动态架构**。Vision Remember、Visual Perception Token等工作表明，允许视觉信息在推理过程中动态流动的架构设计能够有效缓解信息衰减问题。

**趋势三：训练时与推理时方法的融合**。单一类型的方法各有局限，而融合多种技术的混合方案正展现出更强的竞争力。

### 6.2 未来展望

展望未来，以下几个方向值得重点关注：

**视觉-语言联合预训练**：当前大多数研究聚焦于微调阶段，而从预训练阶段即引入视觉直接监督的方案仍有待探索。

**效率与效果的平衡**：许多增强方法带来了额外的计算开销，如何在保持性能的同时实现高效部署是重要的工程挑战。

**评估体系的完善**：现有评估基准对视觉感知能力的细粒度测试仍不够全面，构建更具针对性的评估体系将有助于推动该领域的发展。

## 7. 参考文献

[1] arXiv, 2025-10-10. Visual Representation Alignment for Multimodal Large Language Models. https://arxiv.org/abs/2509.07979

[2] arXiv, 2025-12-03. V-ITI: Mitigating Hallucinations in Multimodal Large Language Models. https://arxiv.org/abs/2512.03542

[3] arXiv, 2025-01-23. Mitigating Visual Knowledge Forgetting in MLLM Instruction-tuning via Modality-decoupled Gradient Descent. https://arxiv.org/abs/2502.11740

[4] arXiv, 2025-12-17. Self-Supervised Visual Learning for Multimodal Large Language Models (JARVIS). https://arxiv.org/abs/2512.15885

[5] arXiv, 2025-07-02. Perceiving Beyond Language Priors: Enhancing Visual Comprehension and Attention in Multimodal Models (VisualLoss/PerceptLLM). https://arxiv.org/abs/2505.05626

[6] arXiv, 2025-06-04. Vision Remember: Recovering Visual Information in Efficient LVLM with Vision Feature Resampling. https://arxiv.org/abs/2506.03928

[7] arXiv, 2026-02-09. Do MLLMs Really See It: Reinforcing Visual Attention in Multimodal LLMs (SAYO). https://arxiv.org/abs/2602.08241

[8] arXiv, 2025-08-05. IKOD: Mitigating Visual Attention Degradation in Large Vision-Language Models. https://arxiv.org/abs/2508.03469

[9] arXiv, 2024-05-09. Boosting Multimodal Large Language Models with Visual Tokens Withdrawal for Rapid Inference (VTW). https://arxiv.org/abs/2405.05803

[10] arXiv, 2025-07-07. MODA: MOdular Duplex Attention for Multimodal Perception. https://arxiv.org/abs/2507.04635

[11] arXiv, 2025-02-24. Introducing Visual Perception Token into Multimodal Large Language Model. https://arxiv.org/abs/2502.17425

[12] arXiv, 2024-06-22. MR-MLLM: Mutual Reinforcement of Multimodal Comprehension and Vision Perception. https://arxiv.org/abs/2406.15768

[13] arXiv, 2025-05-19. VISTA: Enhancing Vision-Text Alignment in MLLMs via Cross-Modal Mutual Information Maximization. https://arxiv.org/abs/2505.10917

[14] arXiv, 2024-10-14. Learning to Ground VLMs without Forgetting (LynX). https://arxiv.org/abs/2410.10491

[15] arXiv, 2025-05-26. Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM. https://arxiv.org/abs/2505.17726

[16] arXiv, 2025-02-27. UniTok: A Unified Tokenizer for Visual Generation and Understanding. https://arxiv.org/abs/2502.20321

[17] arXiv, 2025-09-29. Visual Jigsaw Post-Training Improves MLLMs. https://arxiv.org/abs/2509.25190

[18] arXiv, 2025-09-24. Latent Visual Reasoning. https://arxiv.org/abs/2509.24251

[19] CVPR, 2025. Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens. https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Devils_in_Middle_Layers_of_Large_Vision-Language_Models_Interpreting_Detecting_CVPR_2025_paper.html

[20] arXiv, 2024-04-29. Hallucination of Multimodal Large Language Models: A Survey. https://arxiv.org/abs/2404.18930
