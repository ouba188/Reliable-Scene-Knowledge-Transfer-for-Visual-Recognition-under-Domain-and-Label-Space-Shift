改一下研究内容，用某个港口哨兵一号大景图（元数据，港口知识信息等信息可用，但是对应船只AIS信息不可用）做输入，已有源域（多个不同港口哨兵一号大景图）船只标签，AIS，元数据信息，港口知识信息等等都可用，并且源域与目标域的标签可能会存在有双向不同（不一定），只针对海面船只目标，近岸忽略不处理。训练源域大景图的检测识别分类模型，对目标域（也就是目标大景图）做船只检测并分类。或者使用VLM也行。
可以在我的这个想法上再完善和丰富。
我理解你的新设想。它与原来“在目标港口船舶 chip 上做知识条件伪样本适配”的任务相比，已经发生了三个本质变化：

1. 从**船舶切片分类**升级为**Sentinel-1 大景中的检测—识别—分类一体化**；
    
2. 目标港口不再有单船 AIS，目标知识只能作为**场景级、港口级上下文**，不能直接充当单船标签；
    
3. 源域和目标域的类别空间可能双向不一致，因此它不是普通闭集 UDA，而更接近**Universal Domain Adaptation for Object Detection，通用域适应目标检测**。
    

你原来的任务是利用目标港口可见但未标注的 SAR、AIS 衍生信息和港口知识生成伪样本，再做三类船型分类。 新方案可以提升为：

$$\boxed{ \textbf{无目标 AIS 条件下，面向 Sentinel-1 大景的跨港口海面船舶通用域适应检测与开放类别识别} }$$

英文工作题名可以暂定为：

> **Port-Aware Universal Domain Adaptation for AIS-Free Offshore Ship Detection and Open-Set Recognition in Sentinel-1 Imagery**

---

# 一、重新定义研究问题

## 1. 源域

设源港口集合为：

$$\mathcal P_S=\{p_1,\ldots,p_M\}.$$

每个源港口具有：

$$\mathcal D_p^S = \left( I_p,\, M_p,\, G_p,\, \mathcal A_p \right),$$

其中：

- $I_p$：Sentinel-1 VV/VH 大景图；
    
- $M_p$：入射角、轨道方向、极化、采集日期、产品类型等元数据；
    
- $G_p$：港口设施、锚地、航道、泊位、港口功能等结构化知识；
    
- $\mathcal A_p$：由 AIS 和人工质量控制得到的船舶框、位置和类别标签。
    

写成实例级标注：

$$\mathcal A_p = \left\{ (b_{pi},y_{pi}) \right\}_{i=1}^{N_p}.$$

这里建议：**源域 AIS 主要用于构建检测框和类别真值，不要把源船的 AIS 船型、MMSI、速度、目的地等作为分类器部署输入**。因为目标端没有这些信息，否则会形成明显的训练—部署信息不对称。

---

## 2. 目标域

目标港口 tt 只有：

$$\mathcal D_t^U = (I_t,M_t,G_t),$$

允许查看：

- 目标 Sentinel-1 大景；
    
- 目标港口静态设施与功能知识；
    
- 目标元数据；
    
- 从目标影像无监督得到的船舶密度、候选框、尺度和方向统计。
    

不允许查看：

$$Y_t,\quad \text{单船 AIS 类型},\quad \text{单船 MMSI 标签}.$$

目标输出为：

$$\widehat{\mathcal O}_t = \left\{ (\widehat b_j,\widehat c_j,\widehat u_j) \right\}_{j=1}^{\widehat N_t},$$

其中：

- $\widehat b_j$：船舶旋转框或普通框；
    
- $\widehat c_j$：已知类别预测；
    
- $\widehat u_j$：是否属于未知／目标私有类别。
    

---

# 二、源、目标类别不一致意味着什么？

设源、目标真实类别空间分别为：

$$\mathcal C_S,\qquad\mathcal C_T.$$

它们可分为：

$\mathcal C_{\mathrm{shared}} = \mathcal C_S\cap\mathcal C_T$, $\mathcal C_{\mathrm{source-private}} = \mathcal C_S\setminus\mathcal C_T$, $\mathcal C_{\mathrm{target-private}} = \mathcal C_T\setminus\mathcal C_S$.

你描述的“源域与目标域标签可能双向不同”，对应的是：

$$\mathcal C_{\mathrm{source-private}}\neq\varnothing, \qquad \mathcal C_{\mathrm{target-private}}\neq\varnothing.$$

这不是普通 closed-set UDA，而是 **Universal Domain Adaptation**。

现有 UniDAOD 已经明确把目标检测中的 closed-set、partial-set 和 open-set 统一到 Universal DAOD 问题中，并指出：如果错误地对齐私有类别，会产生明显负迁移。([DOI](https://doi.org/10.1609/aaai.v39i10.33156?utm_source=chatgpt.com "Universal Domain Adaptive Object Detection via Dual Probabilistic Alignment | Proceedings of the AAAI Conference on Artificial Intelligence")) SFUOD 则进一步研究目标中存在未定义类别时，如何同时检测已知对象和未知对象。([ML Anthology](https://mlanthology.org/iccv/2025/park2025iccv-sfuod/?utm_source=chatgpt.com "SFUOD: Source-Free Unknown Object Detection | ML Anthology"))

因此，你的系统不能强迫目标港口中的每艘船都归入源类别。至少需要输出：

$$\boxed{ \text{known shared class} \quad\text{或}\quad \text{unknown vessel type}. }$$

更进一步，VLM 可以尝试给 unknown cluster 命名，但“发现未知”和“给未知正确命名”应当分成两个任务。

---

# 三、海面船舶、忽略近岸，反而能把科学问题收敛得更好

建议定义一个严格的 offshore 区域：

$$\Omega_t^{\mathrm{off}} = \Omega_t^{\mathrm{sea}} \setminus \operatorname{Buffer} (\mathrm{coastline},r_{\mathrm{coast}}).$$

只评估：

$$b_j\subseteq\Omega_t^{\mathrm{off}}.$$

这样可以主动排除：

- 岸线强散射；
    
- 港口建筑；
    
- 码头、桥梁和防波堤；
    
- 近岸密集目标粘连；
    
- 陆地虚警。
    

已有工作已经研究用无监督海陆分割引导 SAR 船舶检测，并将岸线抑制作为重要场景先验；你们可以使用相同类别的海陆掩膜技术，但不需要把近岸问题同时纳入主任务。([arXiv](https://arxiv.org/abs/2506.12775?utm_source=chatgpt.com "Scene-aware SAR ship detection guided by unsupervised sea-land segmentation"))

不过，这也改变了港口知识的作用：

> 当船只已经位于外海，泊位、油罐、仓库等局部设施关系未必能直接分配给某一艘船。

因此，目标港口知识主要应承担：

1. 目标港口的船型候选集合和类别先验；
    
2. 航线、锚地、港口专业化和交通模式的场景上下文；
    
3. 目标域元数据和成像条件解释；
    
4. Unknown 类别发现与文本候选构造。
    

不宜将某个设施直接当作外海某艘船的硬标签。

---

# 四、推荐的总体框架：检测与分类解耦

我建议采用：

$$\boxed{ \text{Class-Agnostic Detection} \rightarrow \text{Universal/Open-Set Classification} }$$

而不是从一开始就训练一个完全类别耦合的 detector。

原因是：

> 目标私有类别虽然没有源类别标签，但它仍然是“船”，应该首先被检测出来。

---

## 模块 A：大景预处理与离岸区域构建

输入：

$$(I_p,M_p,G_p).$$

执行：

1. VV/VH 辐射定标与一致化；
    
2. 入射角和产品元数据编码；
    
3. 海陆分割；
    
4. 岸线缓冲区排除；
    
5. 大景重叠切块；
    
6. 记录 tile 到原始地理坐标的映射。
    

建议采用 coarse-to-fine：

$$\text{大景低分辨率候选生成} \rightarrow \text{高分辨率局部船舶检测／识别}.$$

这样比直接把整景缩小输入网络更合理。

---

## 模块 B：类别无关的船舶检测器

训练：

$$D_\theta(I) \rightarrow \{(b_j,o_j)\},$$

其中 $o_j$ 只表示：

$$\text{ship objectness}.$$

源域监督损失：

$$\mathcal L_{\mathrm{det}}^S = \mathcal L_{\mathrm{box}} + \mathcal L_{\mathrm{obj}} + \mathcal L_{\mathrm{angle}}.$$

如果船舶方向信息重要，建议使用 oriented bounding box。

目标端使用 teacher–student 一致性：

$$\mathcal L_{\mathrm{det}}^T = \sum_j w_j \left[ \mathcal L_{\mathrm{box-cons}} + \mathcal L_{\mathrm{obj-cons}} \right].$$

要点是：

- detection pseudo-label 先只判断“是不是船”；
    
- 不急于给目标船分源类别；
    
- 减少 target-private 类被错误压到某个 source class 的风险。
    

2025 年 DAOD 研究已经开始区分不同区域和实例的适应重要性，而不是整体全图对齐；这对大景 SAR 中前景极少、背景极多的场景尤其重要。([AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/33885?utm_source=chatgpt.com "Differential Alignment for Domain Adaptive Object Detection | Proceedings of the AAAI Conference on Artificial Intelligence"))

---

# 五、核心分类模块：港口条件化的 Universal Classifier

检测后得到局部特征：

$$z_j = E_\theta(I_t,\widehat b_j).$$

同时构造目标港口描述：

$$\boxed{ d_t = \Phi_M(M_t) \oplus \Phi_G(G_t) \oplus \Phi_U(I_t) }$$

其中：

### $\Phi_M(M_t)$

编码：

- incidence angle；
    
- orbit；
    
- polarization；
    
- acquisition date；
    
- resolution；
    
- product type。
    

### $\Phi_G(G_t)$

编码：

- 港口功能；
    
- 主要货类；
    
- 锚地与航道；
    
- 码头类型；
    
- 设施构成；
    
- 港口知识图结构。
    

### $\Phi_U(I_t)$

从目标无标签大景统计：

- 检测候选密度；
    
- 船舶尺寸与长宽比分布；
    
- 船舶航向／主方向分布；
    
- 空间密度热图；
    
- 视觉模型预测分布。
    

---

## 1. 港口条件化类别原型

源港口每个类别有基础原型：

$$\mu_c.$$

不是简单把港口向量拼在分类器后面，而是让港口上下文改变类别在目标域中的合理表示：

$$\boxed{ \mu_{c,t} = \operatorname{Norm} \left( \mu_c+ A_c d_t \right). }$$

或者更一般地：

$$\mu_{c,t} = T_\psi(\mu_c,d_t).$$

预测：

$$s_{jc} = \tau \cos(z_j,\mu_{c,t}).$$

这个模块可以在源港口中做 leave-one-port-out 元训练：

> 模拟某源港口为无标签目标，只使用其大景、元数据和港口知识预测该港口的类别原型变化。

由于独立港口只有几十个，建议使用低秩线性变换、小型残差网络或强正则化模型，而不是大规模超网络。

---

## 2. Unknown 检测

定义：

$$u_j = \min_{c\in\mathcal C_S} d(z_j,\mu_{c,t}).$$

若：

$$u_j>\tau_{\mathrm{unk}},$$

则标记：

$$\widehat c_j=\mathrm{unknown}.$$

阈值不能在真实目标标签上选择，应该用源港口轮流模拟目标的方式校准。

Unknown 样本可以进一步聚类：

$$\mathcal Z_t^{\mathrm{unk}} = \{z_j:u_j>\tau_{\mathrm{unk}}\}.$$

得到：

$$\mathcal K_t^1,\ldots,\mathcal K_t^R.$$

这些 cluster 可以暂时输出：

$$\text{unknown-1},\ldots,\text{unknown-R}.$$

如果 VLM 的文本匹配充分可靠，再尝试命名。

---

# 六、VLM 应该放在哪里？

SARCLIP 已经证明 SAR 专用视觉—语言模型可以支持图文检索、零样本分类、少样本分类和目标计数；其方法还通过检测标注转换为文本构建 SAR 图文数据。([DOI](https://doi.org/10.1109/tgrs.2025.3630131?utm_source=chatgpt.com "SARCLIP: The First Vision–Language Foundation Model for SAR Image"))

因此可以使用 VLM，但我不建议把 VLM 直接作为整景船舶检测器。更合理的是：

$$\boxed{ \text{传统／专用检测器负责定位，VLM负责语义分类与未知类别命名}. }$$

---

## VLM 方案

对每个检测 crop：

$$z_j^{\mathrm{vlm}} = E_{\mathrm{SAR-VLM}}(I_t[\widehat b_j]).$$

文本候选：

$$t_c = E_T( \text{ship-type prompt}_c ).$$

Prompt 可以包含：

- 类别名称；
    
- 常见尺寸；
    
- 船体用途；
    
- Sentinel-1 SAR 中可能出现的形态描述；
    
- 与港口类型的软关联。
    

例如：

> “an offshore bulk carrier observed in Sentinel-1 VV/VH SAR, elongated hull, large cargo vessel”

但**港口知识只能作为软上下文，不能把“散货港”直接等同于“bulk carrier”**。

最终可采用：

$$s_{jc}^{\mathrm{final}} = s_{jc}^{\mathrm{prototype}} + \lambda_{\mathrm{vlm}} s_{jc}^{\mathrm{vlm}}.$$

更保守的方式是：VLM 只用来给 unknown cluster 提供候选名称，不影响 shared-class 主分类。

---

## VLM 在这个任务中的限制

遥感 VLM 和开放词汇方法已经开始扩展到 SAR；SegEarth-OV/AlignEarth 通过把光学 VLM 语义知识蒸馏到 SAR 编码器，实现了 SAR 开放词汇解释。([arXiv](https://arxiv.org/abs/2508.18067?utm_source=chatgpt.com "Annotation-Free Open-Vocabulary Segmentation for Remote-Sensing Images"))

但对于 Sentinel-1 中低分辨率、细粒度船型：

- cargo、bulk、tanker 的 SAR 外观可能高度相似；
    
- VLM 可能只识别“ship”而不能稳定区分子类；
    
- 全景图中的船过小，直接做整图 VLM 很容易丢失目标。
    

所以建议：

$$\boxed{ \text{VLM是辅助语义分支，不是唯一识别器。} }$$

---

# 七、扩散伪样本仍然可以保留，但角色需要改变

你原来的目标条件生成思路仍然有价值，但应变成该框架的**可选数据适应模块**。

---

## 1. 已知源类别的目标背景合成

从源港口提取有标签船舶：

$$(x_s,y_s).$$

从目标大景提取无目标海面背景：

$$b_t.$$

根据目标元数据和海况生成：

$$\widetilde x_{s\to t} = G_\omega(x_s,b_t,M_t).$$

生成结果用于训练：

- 检测器；
    
- shared/source-known 类分类器。
    

---

## 2. 不要用扩散生成 target-private 类标签

如果某类在源域完全不存在：

$$c\in\mathcal C_{\mathrm{target-private}},$$

生成器没有真实源标签依据，不应凭文本提示直接制造监督标签。

Target-private 类应该：

1. 先作为 unknown 被检测；
    
2. 再由目标聚类和 VLM 语义进行候选命名；
    
3. 必要时通过少量人工验证进入增量学习。
    

---

## 3. 生成数据的主要价值

它主要解决：

$$P_S(X\mid Y=c) \neq P_T(X\mid Y=c)$$

对 shared/source-known 类的观测差异。

它不能自动解决：

$$\mathcal C_T\setminus\mathcal C_S.$$

这一区分应该写进框架。

---

# 八、一个完整的训练目标

可以写成：

$$\boxed{ \begin{aligned} \mathcal L ={}& \mathcal L_{\mathrm{det}}^S + \lambda_{\mathrm{cls}}\mathcal L_{\mathrm{cls}}^S\\ &+ \lambda_{\mathrm{detT}}\mathcal L_{\mathrm{det}}^T + \lambda_{\mathrm{cons}}\mathcal L_{\mathrm{cons}}^T\\ &+ \lambda_{\mathrm{proto}}\mathcal L_{\mathrm{proto}} + \lambda_{\mathrm{unk}}\mathcal L_{\mathrm{unknown}}\\ &+ \lambda_{\mathrm{port}}\mathcal L_{\mathrm{port-meta}} + \lambda_{\mathrm{syn}}\mathcal L_{\mathrm{synthetic}}. \end{aligned} }$$

解释如下。

### 源域检测与分类

$$\mathcal L_{\mathrm{det}}^S,\qquad \mathcal L_{\mathrm{cls}}^S.$$

### 目标检测一致性

$$\mathcal L_{\mathrm{det}}^T$$

仅使用高稳定 objectness 和 box pseudo-label。

### 目标分类一致性

只对高可信 shared-class 样本使用：

$$\mathcal L_{\mathrm{cons}}^T.$$

疑似 unknown 不参与 source-class 强制伪标签。

### 原型学习

$$\mathcal L_{\mathrm{proto}}$$

扩大 shared-class 类间距，并让源港口留一模拟目标时的类别原型可预测。

### Unknown 学习

$$\mathcal L_{\mathrm{unknown}}$$

使目标私有样本远离 source prototype，同时避免把普通低质量 shared 样本全部判 unknown。

### 港口元学习

$$\mathcal L_{\mathrm{port-meta}}$$

要求：

$$d_p \rightarrow \mu_{c,p}$$

在源港口留一任务中能够预测真实港口的类条件表示变化。

---

# 九、标签双向不一致时，最推荐的输出层级

不要只输出一个扁平类别。

建议采用层级输出：

$$\text{ship} \rightarrow \begin{cases} \text{known shared class},\\ \text{unknown ship type}. \end{cases}$$

如果 VLM 足够可信：

$$\text{unknown} \rightarrow \text{candidate text labels}.$$

更完整的层级可以是：

$$\text{ship} \rightarrow \{\text{cargo-like},\text{fishing-like},\text{service-like},\ldots\} \rightarrow \text{fine type}.$$

这样即使 fine-grained label space 不一致，也能评价粗粒度迁移。

---

# 十、实验协议必须怎样设计？

## 1. 外层港口留一

每次一个港口作为目标：

$$p_t.$$

其余港口为源。

目标 AIS／标签完全隐藏。

---

## 2. 目标适应场景与最终评价场景最好分开

若使用同一张目标大景做适应和评价，这属于：

$$\text{transductive UDA}.$$

这可以做，但应明确。

更有说服力的设置是：

$$I_t^{\mathrm{adapt}}$$

作为无标签适应大景，

$$I_t^{\mathrm{eval}}$$

为不同日期、轨道或产品的评价大景。

这样才能证明：

> 学到的是港口域规律，而不只是记住同一张影像。

---

## 3. 构建四种标签空间协议

### Closed-set

$$\mathcal C_S=\mathcal C_T.$$

### Partial-set

$$\mathcal C_T\subset\mathcal C_S.$$

### Open-set

$$\mathcal C_S\subset\mathcal C_T.$$

### Universal

$$\mathcal C_S\neq\mathcal C_T,$$

且双方都有私有类别。

Universal 应作为最终主设置，但先从 closed-set 调通检测器是更稳妥的工程路线。

---

# 十一、评价指标

## 检测

- AP50；
    
- AP50:95；
    
- rotated AP；
    
- ship recall；
    
- false alarms per km²；
    
- 小目标 recall。
    

## Shared-class 分类

- Accuracy；
    
- Balanced Accuracy；
    
- macro-F1；
    
- classwise recall；
    
- per-port worst-class recall。
    

## Universal/Open-set

- shared-class accuracy；
    
- unknown recall；
    
- unknown precision；
    
- AUROC；
    
- FPR95；
    
- OSCR；
    
- known/unknown harmonic score；
    
- target-private clustering NMI/ARI。
    

## 端到端

- class-aware detection mAP；
    
- known/unknown end-to-end AP；
    
- worst-port AP；
    
- worst-port Balanced Accuracy；
    
- source-private hallucination rate。
    

---

# 十二、建议的强基线

至少包括：

1. Source-only detector；
    
2. closed-set UDA detector；
    
3. teacher–student DAOD；
    
4. Differential Alignment；
    
5. Universal DAOD / DPA；
    
6. source-free unknown detector；
    
7. 2026 cross-domain SAR ship detector；
    
8. SARCLIP zero-shot crop classification；
    
9. port-knowledge-free prototype model；
    
10. metadata-only；
    
11. port knowledge only；
    
12. diffusion augmentation；
    
13. 完整方法。
    

通用 UniDAOD 已经有明确的 open/partial/closed 对照框架，应作为最接近的通用基线。([DOI](https://doi.org/10.1609/aaai.v39i10.33156?utm_source=chatgpt.com "Universal Domain Adaptive Object Detection via Dual Probabilistic Alignment | Proceedings of the AAAI Conference on Artificial Intelligence")) SAR 领域也已经出现光学→SAR 的语义提示跨域船舶检测，以及跨场景 SAR 船舶检测方法；你们的区别应落在**跨港口、全景 Sentinel-1、无目标 AIS、双向标签空间不一致和港口知识条件**。([DOI](https://doi.org/10.1016/j.jag.2026.105119?utm_source=chatgpt.com "Bridging optical and SAR images via semantic prompt-guided progressive alignment for rotated cross-domain ship detection - ScienceDirect"))

---

# 十三、这条新方向最可能形成的创新点

## 创新点 1：新问题定义与 benchmark

> 无目标 AIS、源目标标签空间双向不一致的 Sentinel-1 跨港口海面船舶检测与开放类别识别。

这比原来的 chip classification 更完整，也更符合真实监测流程。

---

## 创新点 2：检测与开放分类解耦

> 用类无关 objectness 保证 target-private 船舶仍被检测，再通过 universal classifier 完成 shared／unknown 分离。

这比普通类别耦合 detector 更适合标签空间变化。

---

## 创新点 3：港口条件化类别原型运输

> 用目标元数据、港口知识和无标签大景统计预测 shared 类在目标港口中的表示变化，而不是简单做全局域对齐。

注意这目前是候选创新，需要与普通条件分类器、FiLM、adapter、prototype alignment 做严格对照。

---

## 创新点 4：AIS-free 目标标签空间发现

> 不使用目标单船 AIS，通过目标 SAR cluster、SAR VLM 和港口知识发现 target-private 类或候选语义。

这一点与 SARCLIP 等近期 SAR VLM 可以形成良好衔接，但不应保证 VLM 一定能够细粒度命名。([DOI](https://doi.org/10.1109/tgrs.2025.3630131?utm_source=chatgpt.com "SARCLIP: The First Vision–Language Foundation Model for SAR Image"))

---

## 创新点 5：已知类生成适应＋未知类显式保留

> 扩散模型只为 source-known 类生成目标风格训练样本；target-private 类不被伪造标签，而是保留为 unknown 并单独发现。

这是一个很清楚的方法原则。

---

# 十四、需要提前承认的最大困难

## 1. 没有目标 AIS，最终怎样评价分类？

部署阶段没有目标 AIS 完全合理。

但科研评估仍然必须有隐藏目标真值，例如：

- 事后 AIS；
    
- 人工专家判读；
    
- 船舶注册数据库；
    
- 多源高可信标签；
    
- 后续时间匹配。
    

如果目标港口始终没有任何类别真值，只能评价：

- 船舶检测；
    
- known/unknown 稳定性；
    
- cluster consistency；
    
- VLM 候选语义。
    

无法科学证明细粒度分类准确率。

---

## 2. 港口知识对外海船舶可能只有弱统计作用

如果船已经远离港口，设施知识未必能给它提供可靠的实例级类型信息。

因此，应把知识定位成：

$$\boxed{ \text{场景级软先验和标签空间提示} }$$

而不是实例级硬规则。

---

## 3. 源 AIS 标签本身可能有噪声

AIS 类型编码、位置插值和 SAR 船舶匹配都可能出错。

建议：

- 多信号标签解析；
    
- 位置／时间质量阈值；
    
- 船长、轨迹和类型一致性检查；
    
- 低可信源标签单独建模。
    

否则域适应可能只是在传播源标签噪声。

---

# 十五、建议的实施顺序

不要一开始同时上 VLM、扩散、Universal DA 和知识图。

## Phase 1：完整大景闭集基线

- Offshore mask；
    
- 大景切块；
    
- class-agnostic detection；
    
- shared closed-set classification。
    

## Phase 2：Closed-set UDA

- teacher–student target detection；
    
- target pseudo-box；
    
- metadata-conditioned adaptation。
    

## Phase 3：Universal Label Space

- shared/source-private/target-private；
    
- unknown rejection；
    
- target-private clustering。
    

## Phase 4：港口知识条件化

- metadata only；
    
- knowledge only；
    
- unlabeled scene stats only；
    
- 三者联合；
    
- port-conditioned prototype transport。
    

## Phase 5：VLM

- SARCLIP crop embedding；
    
- source label text prototype；
    
- unknown cluster candidate naming。
    

## Phase 6：扩散伪样本

- shared/source-known 类的目标海面背景适配；
    
- 不生成 target-private 伪标签；
    
- 测试是否改善 shared-class AP/BA。
    

这个顺序能清楚定位每个模块的真实贡献。

---

# 十六、可直接用在论文或开题中的“研究内容”表述

> 本文研究无目标船舶 AIS 监督条件下的跨港口 Sentinel-1 SAR 海面船舶检测与开放类别识别问题。源域由多个港口的有标注 Sentinel-1 大景、船舶 AIS 关联标签、成像元数据和港口结构化知识构成；目标域仅提供无标注 Sentinel-1 大景、成像元数据和港口知识，不提供单船 AIS 与类别标签。考虑不同港口之间成像条件、海面杂波、船舶组成和港口功能的变化，以及源域与目标域类别空间可能存在双向私有类别，本文将任务建模为通用域适应目标检测问题。框架首先在离岸海域中进行类别无关船舶检测，使源域未出现的目标私有船型仍可被定位；随后利用源域类别原型、目标港口元数据、结构化港口知识及无标签目标场景统计，完成共享类别识别、源私有类别抑制与目标私有类别发现。进一步研究 SAR 视觉—语言模型对未知类别语义命名的辅助作用，以及扩散式目标背景生成对源已知类别观测迁移的补充作用，最终实现不依赖目标 AIS 的全景船舶检测、已知类别识别和未知类别发现。

---

# 我的总体判断

这个新设想比原来的 chip-level 知识条件生成更完整，也更容易形成一个清楚的实际任务：

$$\boxed{ \text{Full-scene detection} + \text{Universal domain adaptation} + \text{Port-aware context} + \text{Open-vocabulary recognition} }$$

但我建议把论文的**第一核心**放在：

$$\boxed{ \text{无目标 AIS、标签空间未知的跨港口海面船舶通用域适应} }$$

港口知识、VLM 和扩散生成分别作为：

- 场景上下文；
    
- 开放语义辅助；
    
- 已知类别视觉适应。
    

这样框架不会依赖某一个模块是否成功，也避免把所有创新都押在“扩散生成”上。