这个新实验不应被设计成“把原来的 chip 分类器换成大景检测器”，而应重新定义成一个完整的新问题：

$$\boxed{ \textbf{无目标 AIS 条件下的跨港口 Sentinel-1 海面船舶通用域适应检测与开放类别识别} }$$

它与原来“目标港口无标签 SAR／知识条件生成伪样本，再做三类船型分类”的实验已经有本质区别。原任务仍以船舶 chip 和固定三分类为中心；新任务以大景检测、未知类别发现和标签空间不一致为中心。

我建议把论文核心压缩为三个问题：

$$\boxed{ \begin{aligned} &\text{目标端没有单船 AIS，如何找到全部海面船舶？}\\ &\text{源、目标类别空间不一致，如何识别已知类并保留未知类？}\\ &\text{港口知识如何帮助跨港口迁移，而不成为错误的船型捷径？} \end{aligned} }$$

下面给出一套可以直接落地的实验设计和创新点构造方案。

---

# 一、先冻结论文的任务定义

## 1. 源域数据

源港口集合：

$$\mathcal P_S=\{p_1,\ldots,p_M\}.$$

每个源港口可以使用：

$$\mathcal D_p^S= \left\{ I_p,\, M_p,\, G_p,\, A_p \right\},$$

其中：

- $I_p$：Sentinel-1 VV/VH 大景；
    
- $M_p$：入射角、升降轨、极化、采集日期、产品类型、分辨率等元数据；
    
- $G_p$：港口设施、锚地、航道、泊位、功能类型、港口空间图；
    
- $A_p$：源域 AIS 与 SAR 关联得到的船舶框和类别标签。
    

写成实例级标注：

$$A_p= \left\{ (b_{pi},y_{pi},q_{pi}) \right\}_{i=1}^{N_p},$$

其中 $q_{pi}$ 是 AIS—SAR 匹配质量。

一个重要原则是：

> **源域 AIS 可以用于生成训练标签，但不应把部署时目标端不可用的 MMSI、AIS 船型字段、呼号等直接作为分类器输入。**

否则模型会利用训练—部署不对称信息。

---

## 2. 目标域数据

目标港口只有：

$$\mathcal D_t^U= \{I_t,M_t,G_t\}.$$

允许模型访问：

- 目标大景；
    
- 目标元数据；
    
- 港口静态知识；
    
- 目标无标签影像中的船舶密度、尺度、方向、背景等统计。
    

禁止访问：

$$Y_t,\qquad \text{目标单船 AIS 类型},\qquad \text{目标 MMSI 标签}.$$

模型输出：

$$\widehat{\mathcal O}_t = \left\{ (\widehat b_j,\widehat c_j,\widehat u_j) \right\}_{j=1}^{\widehat N_t},$$

其中：

- $\widehat b_j$：船舶旋转框；
    
- $\widehat c_j$：已知源类别；
    
- $\widehat u_j$：未知类别得分。
    

最终必须允许输出：

$$\boxed{\text{unknown ship type}}$$

而不是强迫所有目标船只进入源类别。

---

# 二、实验设置必须覆盖四种标签空间关系

设源、目标类别空间分别为：

$$\mathcal C_S,\qquad \mathcal C_T.$$

至少设计以下四种协议。

## 1. Closed-set

$$\mathcal C_S=\mathcal C_T.$$

作用：先确认基本检测与跨港口适应是否成立。

## 2. Partial-set

$$\mathcal C_T\subset\mathcal C_S.$$

目标港口缺少某些源类别。模型应避免把源私有类别大量幻觉到目标港口。

## 3. Open-set

$$\mathcal C_S\subset\mathcal C_T.$$

目标港口存在训练时未见类别，模型需要检测并标记 unknown。

## 4. Universal

$$\mathcal C_S\neq\mathcal C_T,$$

并且：

$\mathcal C_S\setminus\mathcal C_T\neq\varnothing$, $\mathcal C_T\setminus\mathcal C_S\neq\varnothing$.

这是你的最终主设置。

Universal DAOD 已经研究 closed、partial 和 open 等标签空间变化，并指出错误对齐 domain-private 类会造成负迁移；因此，“考虑标签空间不一致”本身不是足够的创新，必须进一步体现跨港口结构信息的作用。([DOI](https://doi.org/10.1609/aaai.v39i10.33156?utm_source=chatgpt.com "Universal Domain Adaptive Object Detection via Dual Probabilistic Alignment | Proceedings of the AAAI Conference on Artificial Intelligence"))

---

# 三、最重要的实验协议：目标适应大景与最终评价大景分开

这是论文可信度的关键。

## 推荐协议

对目标港口 tt，按采集日期、轨道或产品划分：

$$I_t^{\mathrm{adapt}}$$

和：

$$I_t^{\mathrm{eval}}.$$

### Adaptation scenes

可以看图像、元数据和知识，但不看标签：

$$\left( I_t^{\mathrm{adapt}}, M_t^{\mathrm{adapt}}, G_t \right).$$

### Evaluation scenes

在所有方法、阈值和 checkpoint 冻结后才评价：

$$I_t^{\mathrm{eval}}.$$

这样验证的是：

> 模型是否学到了目标港口域规律，而不只是传导式记住同一张大景。

另外可以附加一个 transductive 版本：

$$I_t^{\mathrm{adapt}} = I_t^{\mathrm{eval}},$$

但必须明确称为 transductive UDA，不能与跨采集泛化混在一起。

---

# 四、整个框架建议采用“检测—分类解耦”

$$\boxed{ \text{Class-Agnostic Offshore Detection} \rightarrow \text{Universal/Open-Set Ship Recognition} }$$

这一架构选择很重要，但不能单独作为核心创新。2025 年 LLaMA-Unidetector 已经将开放词汇遥感检测分成类别无关定位和 MLLM 识别两阶段；因此，你们真正的贡献应落在跨港口适应和港口结构条件化，而不是声称首次解耦检测与分类。([IEEE Xplore](https://ieeexplore.ieee.org/document/10976651/?utm_source=chatgpt.com "LLaMA-Unidetector: An LLaMA-Based Universal Framework for Open-Vocabulary Object Detection in Remote Sensing Imagery | IEEE Journals & Magazine | IEEE Xplore"))

---

# 五、模块一：离岸区域构建与大景切片

定义：

$$\Omega_t^{\mathrm{off}} = \Omega_t^{\mathrm{sea}} \setminus \operatorname{Buffer} (\mathrm{coastline},r).$$

只在：

$$\Omega_t^{\mathrm{off}}$$

内检测和评价。

## 建议流程

1. VV/VH 定标与统一缩放；
    
2. 构建海陆 mask；
    
3. 加岸线缓冲区；
    
4. COG 窗口读取；
    
5. 重叠 tile；
    
6. tile 检测结果回写大景；
    
7. 重叠框去重。
    

### 缓冲区不能在目标标签上调

可以在源验证港口上选择：

$$r\in\{500,1000,1500,\ldots\}\text{ m},$$

根据离岸检测召回与虚警综合确定。

## 需要单独报告

- 被 offshore mask 排除的真实船只比例；
    
- 离岸区域面积；
    
- 每平方公里虚警数；
    
- 大景边界和 tile 接缝区域的漏检率。
    

否则可能通过扩大岸线排除区“人为提高”检测性能。

---

# 六、模块二：类别无关船舶检测

检测器先只学习：

$$\text{ship}\quad\text{vs}\quad\text{background}.$$

源域：

$$\mathcal L_{\mathrm{det}}^S = \mathcal L_{\mathrm{obj}} + \mathcal L_{\mathrm{box}} + \mathcal L_{\mathrm{angle}}.$$

目标无标签大景使用 teacher–student：

$$\mathcal L_{\mathrm{det}}^T = \sum_jw_j \left[ \mathcal L_{\mathrm{obj-cons},j} + \mathcal L_{\mathrm{box-cons},j} \right].$$

这里暂时不为目标伪框分配细船型标签。

### 为什么要这样做？

如果 detector 从一开始就把“船舶存在性”和“源类别”绑定，那么 target-private 船型可能因为不属于任何源类别而被漏检。

类别无关检测的目标是：

$$\boxed{ \text{即使船型未知，也先把它作为船检测出来。} }$$

## 目标检测伪标签的筛选不应只看分类置信度

可综合：

- 多增强 box 一致性；
    
- teacher–student box IoU；
    
- 旋转角稳定性；
    
- offshore mask；
    
- 多尺度稳定性；
    
- objectness；
    
- 目标尺寸合理性。
    

2025 年 Differential Alignment 已经指出，不同前景实例与背景区域的域对齐重要性并不相同，并利用 teacher–student discrepancy 和前景不确定性分配对齐权重。你们应把它作为 DAOD 强基线，而不是只和普通全局域对齐比较。([AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/33885?utm_source=chatgpt.com "Differential Alignment for Domain Adaptive Object Detection | Proceedings of the AAAI Conference on Artificial Intelligence"))

---

# 七、模块三：目标港口描述符

从目标大景构造：

$$\boxed{ d_t= d_t^{\mathrm{meta}} \oplus d_t^{\mathrm{port}} \oplus d_t^{\mathrm{scene}}. }$$

## 1. 元数据描述

$$d_t^{\mathrm{meta}} = E_M(M_t)$$

包含：

- 入射角分布；
    
- 极化；
    
- 升降轨；
    
- 采集日期；
    
- 产品类型；
    
- 像素分辨率；
    
- 成像模式。
    

## 2. 港口知识描述

$$d_t^{\mathrm{port}} = E_G(G_t)$$

包含：

- 港口功能类型；
    
- 主要设施构成；
    
- 锚地、航道和外海交通区；
    
- 港口吞吐结构；
    
- 船厂、散货、液货、集装箱等设施存在性；
    
- 港口知识图统计。
    

## 3. 无标签大景统计

$$d_t^{\mathrm{scene}} = E_U(I_t^{\mathrm{adapt}})$$

可包含：

- 检测候选密度；
    
- 框尺度和长宽比；
    
- 船舶方向分布；
    
- VV/VH 背景统计；
    
- 目标 embedding 分布；
    
- 源模型预测分布；
    
- unknown score 分布。
    

### 重要约束

不能将目标港口 ID 直接作为输入。

否则模型可能只是记忆：

$$\text{Port ID}\rightarrow\text{class prior}.$$

---

# 八、建议的核心算法创新：港口条件化类别分布运输

这是我认为最值得作为模型创新的部分。

## 1. 不只学习一个全局类别原型

对每个源港口 pp、类别 cc，从真实检测 crop 得到：

$$\mu_{p,c}, \qquad \Sigma_{p,c}.$$

也就是说，每个类别在不同港口具有不同特征分布。

## 2. 使用目标描述预测目标类别分布

学习：

$$\boxed{ (\widehat\mu_{t,c},\widehat\Sigma_{t,c}) = T_\psi \left( \{\mu_{p,c},\Sigma_{p,c},d_p\}_{p\in\mathcal P_S}, d_t \right). } \tag{1}$$

不是只预测一个点原型，而是预测：

$$q_{t,c}(z) = \mathcal N \left( z; \widehat\mu_{t,c}, \widehat\Sigma_{t,c} \right).$$

分类分数：

$$s_{ic} = -\frac12 (z_i-\widehat\mu_{t,c})^\top \widehat\Sigma_{t,c}^{-1} (z_i-\widehat\mu_{t,c}) + \log \widehat\pi_{t,c}.$$

其中 $\widehat\pi_{t,c}$ 是目标类别先验的保守估计或区间，而不是由港口设施硬编码。

### 这比简单拼接港口向量强在哪里？

普通方法：

$$\widehat y=f(z_i,d_t).$$

这里则显式研究：

> 同一船型的 SAR 表征分布怎样随目标港口和成像条件变化。

它可以输出：

- 类别位置变化；
    
- 类内不确定性变化；
    
- 目标类别支持程度；
    
- unknown 判定所需的距离尺度。
    

---

# 九、如何训练这个“港口条件化运输”？

采用源港口 leave-one-port-out meta-training。

对每个源港口 qq：

1. 将 qq 当作伪目标；
    
2. 只把 $I_q,M_q,G_q$ 当成无标签输入；
    
3. 从其他源港口预测：
    
    $\widehat\mu_{q,c}, \widehat\Sigma_{q,c}$;
4. 最后才用 qq 的标签计算元训练损失。
    

例如：

$$\mathcal L_{\mathrm{transport}} = \sum_{q,c} \left[ \|\widehat\mu_{q,c}-\mu_{q,c}\|_2^2 + D_{\mathrm{cov}} (\widehat\Sigma_{q,c},\Sigma_{q,c}) \right].$$

还可直接优化：

$$\mathcal L_{\mathrm{cls-meta}} = \operatorname{CE} \left( s(z_{qi};d_q), y_{qi} \right).$$

真正目标港口 tt 的标签不参与任何训练和选择。

---

# 十、建议的第二个核心创新：双向标签空间模拟训练

仅在最终目标实验中突然出现类别空间变化，模型很难学会 unknown。

建议在每个源伪目标 episode 内，主动构造：

$$\mathcal C^{\mathrm{shared}}, \qquad \mathcal C^{\mathrm{source-private}}, \qquad \mathcal C^{\mathrm{pseudo-target-private}}.$$

## 训练方法

### Shared classes

同时出现在元训练源侧和伪目标侧。

### Source-private classes

在元训练源侧出现，但从伪目标港口中移除。

### Pseudo-target-private classes

在元训练时从有标签源类别集合中隐藏，只把它们的图像作为伪目标无标签样本。

模型需要：

- shared → 正确分类；
    
- source-private → 不应在目标大量预测；
    
- pseudo-target-private → 标记 unknown。
    

训练目标：

$$\mathcal L_{\mathrm{universal}} = \mathcal L_{\mathrm{shared}} + \lambda_{\mathrm{unk}} \mathcal L_{\mathrm{target-private}} + \lambda_{\mathrm{sp}} \mathcal L_{\mathrm{source-private}}.$$

这是比“在测试时设一个能量阈值”更完整的训练机制。

---

# 十一、Unknown 判定不能只使用最大 softmax

定义目标 crop embedding：

$$z_i.$$

对每个 source-known 类：

$$d_{ic} = (z_i-\widehat\mu_{t,c})^\top \widehat\Sigma_{t,c}^{-1} (z_i-\widehat\mu_{t,c}).$$

unknown score：

$$u_i = \min_{c\in\mathcal C_S}d_{ic}.$$

或者联合：

$$u_i = \alpha \min_cd_{ic} + \beta E_i + \gamma(1-\max_cp_{ic}),$$

其中 $E_i$ 是 energy。

阈值：

$$\tau_{\mathrm{unk}}$$

只能通过源港口 pseudo-target episodes 校准，不能查看目标类别标签。

SFUOD 已经研究在 source-free object detection 中同时识别 known 和 undefined objects，所以“增加 unknown 输出”不够新。你的创新应体现为**利用港口描述预测目标类分布，再做 unknown 判定**。([ML Anthology](https://mlanthology.org/iccv/2025/park2025iccv-sfuod/?utm_source=chatgpt.com "SFUOD: Source-Free Unknown Object Detection | ML Anthology"))

---

# 十二、VLM 应作为“未知类语义解释器”，而不是主检测器

SARCLIP 已经证明 SAR 视觉—语言模型可以支持检索、零样本分类、少样本分类和目标计数。([DOI](https://doi.org/10.1109/tgrs.2025.3630131?utm_source=chatgpt.com "SARCLIP: The First Vision–Language Foundation Model for SAR Image"))

因此，目标 unknown crop 可以经过：

$$z_i^{\mathrm{vlm}} = E_{\mathrm{SARCLIP}}(x_i).$$

Unknown cluster：

$$\mathcal K_t^1,\ldots,\mathcal K_t^R$$

与候选文本原型比较：

$$s(r,c) = \cos \left( \bar z_{\mathcal K_t^r}, E_T(\mathrm{prompt}_c) \right).$$

输出应区分：

- `unknown-cluster-1`；
    
- VLM 候选名称；
    
- 候选置信度。
    

不要把 VLM 候选名称当成无需验证的真标签。

遥感开放词汇检测已经发展出“先类别无关定位、再由 MLLM 识别”的结构，而且已有 LAE-DINO、LLaMA-Unidetector 等工作。因此 VLM 更适合作为辅助实验或未知类别解释模块，而不是论文唯一创新。([IEEE Xplore](https://ieeexplore.ieee.org/document/10976651/?utm_source=chatgpt.com "LLaMA-Unidetector: An LLaMA-Based Universal Framework for Open-Vocabulary Object Detection in Remote Sensing Imagery | IEEE Journals & Magazine | IEEE Xplore"))

---

# 十三、港口知识究竟怎样使用，才能形成有效创新？

对于离岸船舶，港口知识不宜直接做实例标签。

建议分成两类作用。

## 1. 类别支持范围

港口知识产生一个**软支持区间**：

$$\underline\pi_{t,c} \le \pi_{t,c} \le \overline\pi_{t,c}.$$

不是硬规则：

$$\text{散货港}\Rightarrow\text{bulk carrier}.$$

而是：

> 根据源港口规律和目标港口功能，某类别在目标港口的可能比例范围是多少？

在源伪目标港口上校准这个区间。

## 2. 类别表示变化

港口知识还可以参与式（1）中的：

$$\widehat\mu_{t,c}, \widehat\Sigma_{t,c}.$$

例如同样是 general cargo，在不同港口的背景、尺度、停留位置与观测条件不同。

## 3. Unknown 词表

港口知识可帮助构造目标可能出现的文本词表：

$$\mathcal V_t.$$

但词表只用于 VLM 候选命名，不能直接排除词表外类别。

---

# 十四、扩散伪样本放在第几阶段最合理？

我建议它不是第一核心，而是第四阶段增强。

只对：

$$c\in\mathcal C_{\mathrm{shared}}$$

或可信的 source-known 类生成目标风格样本：

$$\widetilde x_{s\to t} = G(x_s,b_t,M_t,G_t).$$

用途：

- 缩小 shared-class 视觉观测差异；
    
- 改善分类原型估计；
    
- 增强类别无关检测。
    

禁止：

- 给 target-private 类生成伪真标签；
    
- 仅凭 VLM 文本生成一类新船，然后当作监督真值；
    
- 根据目标港口设施硬生成某类别。
    

扩散模块的增量应单独报告：

$$\text{完整方法} - \text{无扩散版本}.$$

---

# 十五、最小实验矩阵

## 阶段 A：检测任务

|方法|目的|
|---|---|
|Source-only class-aware detector|普通基线|
|Source-only class-agnostic detector|验证检测/分类解耦|
|Teacher–student UDA detector|标准 UDA|
|Differential Alignment|前景/实例差异对齐强基线|
|Proposed detector|离岸、类别无关、目标一致性|

主要看：

- 所有船召回；
    
- target-private 船召回；
    
- 虚警/km²；
    
- rotated AP。
    

---

## 阶段 B：Closed-set 分类

|条件|描述|
|---|---|
|source prototype|不使用目标信息|
|metadata only|只用成像元数据|
|port knowledge only|只用港口结构|
|scene stats only|只用目标无标签大景统计|
|metadata + knowledge|去掉无标签视觉统计|
|full target descriptor|完整方法|

目标是证明增益不是简单的目标类别先验。

---

## 阶段 C：Universal label-space

比较：

- DPA UniDAOD；
    
- SFUOD；
    
- 能量阈值；
    
- 普通 open-set classifier；
    
- port-conditioned distribution transport；
    
- transport + episodic label-space simulation。
    

指标包括：

- shared-class mAP/BA；
    
- unknown recall；
    
- unknown precision；
    
- OSCR；
    
- H-score；
    
- target-private detection recall；
    
- source-private hallucination rate。
    

---

## 阶段 D：VLM

比较：

- 通用 CLIP；
    
- SARCLIP；
    
- port-knowledge prompt；
    
- 无 port prompt；
    
- VLM 仅命名 unknown；
    
- VLM 参与所有分类。
    

重点验证：

> VLM 是否真的能区分细粒度船型，还是只会识别“ship”。

---

## 阶段 E：扩散生成

比较：

- 无生成；
    
- 普通目标背景增强；
    
- 元数据条件生成；
    
- 港口知识条件生成；
    
- 完整 relation-valid 生成。
    

只对 shared/source-known 类评价。

---

# 十六、必须做的几个关键反事实实验

## 1. Wrong-port knowledge

保持目标大景不变，将：

$$G_t$$

替换为相似规模的其他港口知识。

如果预测几乎不变，说明知识模块未被使用。

如果性能显著下降，说明知识进入模型；但还需排除港口 ID shortcut。

## 2. Metadata shuffle

打乱：

- incidence angle；
    
- orbit；
    
- acquisition date。
    

判断性能是否依赖真实成像条件。

## 3. Scene-statistics shuffle

保持港口知识不变，打乱目标船舶尺度和密度统计。

检验 target scene statistics 是否有独立作用。

## 4. Label-support corruption

人为把某个 source-private 类声明为目标高概率类，观察是否产生幻觉。

## 5. Unknown quality stratification

将 unknown 样本按：

- 目标尺寸；
    
- 信噪比；
    
- 与已知原型距离；
    

分层，检验模型是否只是把低质量 shared 类当 unknown。

---

# 十七、统计设计

独立层级至少包括：

$$\text{tile} \subset \text{acquisition} \subset \text{port}.$$

不要把 tile 当成独立跨域样本。

建议报告：

- acquisition-level cluster bootstrap；
    
- port-level paired effect；
    
- per-port performance；
    
- mean-port；
    
- worst-port；
    
- 按类别和标签空间类型分层。
    

所有超参数应在源港口 nested leave-one-port-out 中选择。

---

# 十八、创新点如何正式构造？

我建议最终只保留三个主创新，不要堆六七个。

## 创新点一：问题与 benchmark 创新

> 提出无目标单船 AIS、源目标类别空间可双向不一致的 Sentinel-1 跨港口海面船舶通用域适应检测与开放类别识别任务，并建立港口级、采集级严格划分协议。

这不是单纯“新数据集”，而是新的实际任务设定。

---

## 创新点二：港口条件化类别分布运输

> 不做全局特征对齐，而是利用目标元数据、港口结构和无标签大景统计，预测 shared 类在目标港口中的类条件特征分布及不确定性。

形式上：

$$\boxed{ d_t \rightarrow \left\{ \widehat\mu_{t,c}, \widehat\Sigma_{t,c}, [\underline\pi_{t,c},\overline\pi_{t,c}] \right\}_{c\in\mathcal C_S}. }$$

这比“港口知识拼接到分类头”更具结构。

---

## 创新点三：双向标签空间模拟的统一训练

> 在源港口 meta-episodes 中主动制造 shared、source-private 和 pseudo-target-private 类，使模型在没有真实目标标签的情况下学习共享类迁移、源私有类抑制和目标私有类发现。

这个训练机制直接对应你真正的双向标签空间不一致，而不是测试时才临时加 unknown 阈值。

---

# 十九、哪些内容不宜单独写成创新点？

下面这些应作为实现或支撑，不宜单独承担论文主创新：

- 使用旋转框；
    
- 海陆 mask；
    
- COG/PNG 数据格式；
    
- teacher–student；
    
- energy unknown score；
    
- 使用 SARCLIP；
    
- 加一个 VLM prompt；
    
- diffusion augmentation；
    
- metadata FiLM；
    
- prototype classifier 本身；
    
- class-agnostic detector 本身。
    

这些方向都已有直接先例或非常接近的工作。RotCD-Ship 已经使用 SAR 知识提示、跨域层级对齐和旋转框；SARCLIP 已具备零样本分类等能力；Universal DAOD 与 source-free unknown detection 也已有明确方法。([DOI](https://doi.org/10.1016/j.jag.2026.105119?utm_source=chatgpt.com "Bridging optical and SAR images via semantic prompt-guided progressive alignment for rotated cross-domain ship detection - ScienceDirect"))

---

# 二十、论文的最简主线

整篇论文最好讲成：

### 现实问题

源港口有 AIS 标签，目标港口没有单船 AIS；目标港口可能有新船型，也可能缺少部分源船型。

### 传统方法的问题

- 类别耦合检测器会漏掉 target-private 船型；
    
- 全局域对齐会错误对齐 private classes；
    
- 港口知识若直接作为类别先验，可能造成标签捷径；
    
- 通用 VLM 难以稳定细分中低分辨率 SAR 船型。
    

### 核心洞见

> 将“船舶是否存在”与“属于什么类别”分开；通过目标港口描述预测已知类别在目标域的特征分布，同时显式保留未知类别空间。

### 方法

$$\boxed{ \text{Offshore Objectness} \rightarrow \text{Port-Conditioned Known Distribution} \rightarrow \text{Known/Unknown Recognition} }$$

### 增强

- SAR VLM 用于 unknown cluster 解释；
    
- diffusion 用于 shared-class 目标观测适配。
    

---

# 二十一、明确的失败判据

以下任何一项成立，都应收紧或撤回对应创新主张：

1. 类别无关检测器不能提高 target-private 船舶召回；
    
2. 港口条件化分布与普通 metadata 拼接没有显著差异；
    
3. 改用错误港口知识后结果不变，说明知识未被利用；
    
4. 港口知识提升平均性能，却降低 worst-port 或 unknown recall；
    
5. unknown 主要由低信噪比 shared 类组成；
    
6. 目标类别支持预测导致真实稀有类被错误排除；
    
7. SARCLIP/VLM 只能区分 ship/non-ship，不能提供可靠细类候选；
    
8. 结果只在同一目标大景传导设置中成立，跨日期评价失效；
    
9. 提升全部来自扩散增强，而非核心 universal adaptation；
    
10. 使用目标标签选择了 unknown 阈值、类别词表或模型 checkpoint。
    

---

## 最终建议

先实现一篇最小但逻辑完整的版本：

$$\boxed{ \text{离岸类别无关检测} + \text{港口条件化类别分布运输} + \text{双向标签空间模拟} }$$

暂时把 VLM 和扩散作为第二阶段。

如果这三个部分在以下设置中成立：

- closed；
    
- partial；
    
- open；
    
- universal；
    
- 同港口跨日期；
    

那么创新主线已经比较完整。

之后再加入：

- SARCLIP unknown naming；
    
- target-conditioned diffusion；
    
- 更复杂港口知识图。
    

这样可以避免第一轮实验过度复杂，也最容易判断真正的创新究竟来自哪里。