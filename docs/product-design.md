# J-GraphRAG 产品设计共识（v0.1 → 发布）

> 2026-07-23 设计讨论结论记录。每条决策附依据；与实验相关的待验证项已登记在 experiment 分支 AGENTS.md §6（第 8/9/10 项）。

## 1. 产品定位：Comet 模式

J-GraphRAG 不是又一个 GraphRAG 框架，而是**现有 GraphRAG 框架中 LLM `generate()` 提取段的替换层**（类比 DataFusion Comet 之于 Spark：不动宿主的检索/存储/服务，只换执行核心）。形态为 Python 库为主 + CLI（基于宿主方案改造）+ 可选计算 server。

依据：Phase 50/51/53 终审已完成同类替换验证（LightRAG-J 1.011/1.104 retained；HippoRAG-J 双域 ≥0.9，novel 线 Phase 54 rubric 重判后翻正）。

## 2. 决策清单

| # | 决策点 | 结论 | 依据/备注 |
|---|---|---|---|
| 1 | 产品形态 | 库为主 + CLI + 可选 server | Comet 模式嵌入用户现有栈 |
| 2 | 首选宿主 | HKU LightRAG；长期探索 J-Lens 原生 GraphRAG 方案 | 终审成绩最好、社区最活跃、Storage 抽象与 stores 协议同构 |
| 3 | 替换范围 | 建图提取 + 查询侧关键词提取（phase50 ah 臂原样）；答案生成 LLM 用户自带 | 已验证的最大替换面 |
| 4 | LLM 查询关切 | 预留 QueryProcessor 协议，默认恒等（bge 直查）；实现待实验验证后收编 | experiment §6-8；Phase 42 证伪概念路由≠证伪种子增强；recognition memory 值 +0.07-0.08 说明查询侧 LLM 有潜在增益 |
| 4b | 嵌入模型去留 | **EmbedProvider 是架构必需件，bge-m3 确认留任**（Phase 56/56b 铁案 2026-07-26）；单模型闭合（LLM 自向量替代嵌入）方向划除，不再预留 ws 后端替换路线 | Phase 56：ws 向量替代 bge 种子全形态证伪（概念聚合死于 25% 覆盖空洞；逐 chunk 编码 dense 0.000/sparse≤0.031——文体陷阱+词表错层）；56b：失败按域划界但任何域不实用。J-Lens 向量有效域=图内接地/消歧/多跳判别；总观层面匹配必须嵌入模型 |
| 5 | 集成姿势 | 存储注入（建图产物写入宿主存储格式）+ 查询钩子（子类覆写关键词提取） | 适配面最小；深钩子/假 LLM 均因宿主提取协议为文本生成设计而被否 |
| 6 | 使用姿势 | 子类无感：`rag = JLightRAG(...)`；`rag.insert()` / `rag.query()` 即宿主 API | 用户不配 LLM 即可用 |
| 7 | 增量语义 | v0.2：insert = 追加语料 + 全量重建，chunk 级提取缓存摊销成本；**v0.3 起真增量**（Phase 59 已 SUPPORTED 2026-07-26） | phase59 产品语义：① 剪枝只剪节点、保留 DF 计数器（无需墓碑，复活自然发生）② IDF 定期后台刷新（陈旧漂移 0.994/0.967 可忽略）③ J-Lens 读出边选择集 ~30% 漂移，接受或定期重选。边界：小规模验证，规模化重测 |
| 8 | 语言范围 | v1 英文-only；EntityDetector 协议 + ConceptNormalizer 钩子（默认恒等）预留 | 中文投影实验 §6-10（概念→英文概念空间映射，渊源 Phase 9 xlmr_vec2vec） |
| 9 | 规模边界 | 声明中小语料（≤1万 chunk）验证可用，大规模不保证；Qdrant 薄适配（extras，低投入） | 与验证范围一致的诚实声明；向量库对核心定位是外围问题 |
| 10 | 依赖分层 | 核心轻（numpy/scipy/scikit-learn）+ extras：`[lens]`(torch/jlens/bnb)、`[bge]`(FlagEmbedding)、`[qdrant]`、`[lightrag]` | Comet 用户已有自己的环境，不强制 GPU 依赖 |
| 11 | Server 边界 | 仅计算面服务化：LensProvider 面 + EmbedProvider 面；编排/过滤/存储/检索全在 client | 服务无状态、易扩；RemoteLens/RemoteEmbedProvider 走同一协议 |
| 12 | Server API | 自定义 REST（FastAPI）：`POST /extract/concepts`、`/extract/relation`、`/embed` | OpenAI 兼容外观是伪兼容（J-Lens 不吃文本指令），明确否决 |
| 13 | 质量回归 | 126 单测（日常）+ GPU 黄金小语料冒烟脚本（scripts/ 手动）+ 发布前与 experiment 终审产物对齐检查 | 评测链（DeepSeek judge）不收编，留 experiment |
| 14 | 发布 | v0.1-0.3 GitHub-only 验证功能完整性，功能到位后再发 PyPI；lightrag-hku 锁版本区间 + 兼容矩阵声明 | 防宿主版本漂移 |

## 3. 版本路线

- **v0.1（已完成，ec6ece0）**：走通模式——核心包（providers/stores 协议 + 默认实现 + 建图/检索管线）、126 单测、GPU 冒烟通过。
- **v0.2（下一步）**：LightRAG 系单一技术栈完成验证——`JLightRAG` 子类（存储注入 + 查询钩子）、EntityDetector/ConceptNormalizer/QueryProcessor 协议预留、chunk 级提取缓存、extras 依赖分层、Qdrant 薄适配、GPU 冒烟脚本入库、CLI（insert/query）。
- **v0.3**：横向扩展——计算 server（REST + Remote providers）、**真增量 insert**（Phase 59 三语义：保 DF 计数器剪枝、IDF 后台刷新、关系对定期重选）、其他 GraphRAG 方案适配（HippoRAG2 等）、更多存储后端。
- **之后**：PyPI 发布评估（功能完整性达标 + 对齐检查通过）。

## 4. 实验依赖（experiment 分支 §6）

- 第 8 项：查询侧 LLM 关切生成臂（决策 4 的收编前提）——待验证
- ~~第 9 项：增量建图可行性~~ **已闭环（Phase 59 SUPPORTED）**，三语义并入决策 7，落 v0.3
- 第 10 项：跨语言概念投影（决策 8 的扩展前提）——条件触发

## 5. 不变红线（继承 AGENTS.md §3）

已验证参数不可改（MAX 聚合、衰减 0.5、cos 0.95、DF≥2、MIN_COOCC=2、边上限 20000、prompt 原文）；"糙但已验证"的行为原样保留；任何算法变更先回 experiment 分支走可证伪 Phase 流程。
