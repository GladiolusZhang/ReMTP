# Adaptive RAVEN：112 请求三候选搜索结果与分析

> 采用范围更新（2026-10-14）：用户明确D4/N12相对原Adaptive D3/N9不公平，deep4_conservative不予采用。本报告D4对D3的点结果仅作历史配置实验，无法独立归因于方法机制；相同预算条件未包含在原点门槛中。balanced3虽为D3/N9，历史结果也未通过质量/速度联合门槛。[后续固定预算计划](../plan/adaptive-raven-equal-budget-2026-10-14.md)。

核验日期：2026-10-10（UTC+8）。开始：2026-10-09 11:21:13；完成：**2026-10-10 00:32:59**，持续13小时11分46秒。

## 1. 结论

**后台完整正常结束，112/112 次生成全部完成，exit0、0重试。两个 D4 候选在当前修复后的同批开发集点估计上同时改善原 Adaptive RAVEN 的正确率、吞吐和平均整题耗时。尚没有候选同时超过 Native、Cactus、SpecCascade。**

这次与之前的旧数值版本历史峰值不同：相同当前冻结环境、相同8道训练题×2seed、完整2048输出预算、公共数值/模型/原prompt/评分一致；B与三个基线的64条输出IDs/文本/finish均复现前次记录。

- **deep4_balanced：**12/16→16/16，吞吐+2.36%，平均整题−4.99%，MAL+10.71%；同批对Native也通过质量与两种速度点门槛。
- **deep4_conservative：**12/16→14/16，吞吐+0.73%，平均整题−16.56%；同批对Native也通过点门槛。
- balanced3正确数12→11、吞吐−1.49%，虽整题缩短但未改善联合指标。
- 对三个主要基线的联合门槛为false，开发集点赢家=None；未自动采用或启动留出/正式实验。

**以上是已观察开发题的点结果，不代表总体正确率100%或稳定速度增益已被证明。** 截断/正常结束/逐题逐seed差异按用户要求仅分析，没有作为额外否决条件。

## 2. 运行和CPU复核

用户级服务 `remtp-raven-dominance-search-20261009.service` 已inactive/dead。exit记录与journal均支持正常结束，主请求112/112、启动7/7、预热7/7、数值调用0/0；journal记内存峰值11.7G、swap0。七份server.log未发现Traceback、OutOfMemoryError、CUDA error、EngineDeadError；8000端口无监听。

独立CPU复核实际通过：112逐题记录与112原始payload/response、固定题/seed、官方gold/冻结原评分、真实API prompt/output IDs、时间/token/ledger、7份有效配置、4组64条历史输出复现、7份summary与analysis全部一致；12组比较和每组2000次按题聚类区间重算一致。474份输入hash与冻结source hash核对通过：

`d25bb51a8c22d2f6ba86fea7b70b5dd7e6fdd1478e34458344808de09e614950`

本轮没有额外数值重放/fresh Target检查；沿用既有数值实现。本次CPU核验新增GPU/HTTP调用均为0。原末次heartbeat的server_alive与current_work是退出前历史快照，不表示服务器当前存活。

## 3. 同批七方法结果

8道已观察训练题1532/2950/6009/3213/1231/5180/308/641，每题base seed42/43，共16次生成/方法。FastMTP FP4 Target/BF16物理MTP head0递归、temperature0.6、上下文4096、输出2048、KV1GiB、TP1/单请求/eager/no-thinking、prefix cache0、观测探针关闭。

| 方法 | 原评分正确率 ↑ | E2E token/s ↑ | 平均整题秒 ↓ | MAL ↑ | decode 秒/轮 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 13/16（81.25%） | 2.5274 | 458.31 | 2.9146 | 1.1493 |
| Cactus | 14/16（87.50%） | 2.8932 | 403.21 | 3.3436 | 1.1507 |
| SpecCascade | 14/16（87.50%） | 2.6083 | 367.98 | 3.0153 | 1.1510 |
| 原 Adaptive RAVEN（B） | 12/16（75.00%） | 2.6249 | 460.26 | 3.5122 | 1.3336 |
| balanced3（D3） | 11/16（68.75%） | 2.5856 | 417.98 | 3.4566 | 1.3311 |
| deep4_balanced（D4） | 16/16（100.00%） | 2.6868 | 437.31 | 3.8883 | 1.4411 |
| deep4_conservative（D4） | 14/16（87.50%） | 2.6440 | 384.06 | 3.7986 | 1.4303 |

MAL=`1+总接受草稿token/总验证轮数`，按轮数加权；decode秒/轮取服务器decode计时/验证轮数。吞吐为总输出token/总客户端秒；平均整题为总客户端秒/16。请求时间包括实际生成，不包括服务加载和单列预热。

候选设置：B=D3/N9/δ1/mix0.5；balanced3=D3/N9/δ0.75/mix0.25；deep4_balanced=D4/N12/δ0.75/mix0.25；deep4_conservative=D4/N12/δ0.5/mix0.25。均head输入行批1/full proposal/adaptive/residual_hit_anchor，risk none、cap/lazy/repeat guard关闭；D4并非此前旧anchor-only候选。三个基线本单元均D3。

## 4. 原方法和三基线逐项比较

| 候选 | 对照 | 原正确率差/百分点 | 吞吐变化 | 整题耗时变化 | 点门槛未过项 |
| --- | --- | ---: | ---: | ---: | --- |
| balanced3 | 原 Adaptive RAVEN（B） | -6.25 | -1.49% | -9.19% | 正确率、吞吐 |
| balanced3 | Native | -12.50 | +2.31% | -8.80% | 正确率 |
| balanced3 | Cactus | -18.75 | -10.63% | +3.66% | 正确率、吞吐、整题耗时 |
| balanced3 | SpecCascade | -18.75 | -0.87% | +13.59% | 正确率、吞吐、整题耗时 |
| deep4_balanced | 原 Adaptive RAVEN（B） | +25.00 | +2.36% | -4.99% | 通过 |
| deep4_balanced | Native | +18.75 | +6.31% | -4.58% | 通过 |
| deep4_balanced | Cactus | +12.50 | -7.13% | +8.46% | 吞吐、整题耗时 |
| deep4_balanced | SpecCascade | +12.50 | +3.01% | +18.84% | 整题耗时 |
| deep4_conservative | 原 Adaptive RAVEN（B） | +12.50 | +0.73% | -16.56% | 通过 |
| deep4_conservative | Native | +6.25 | +4.61% | -16.20% | 通过 |
| deep4_conservative | Cactus | +0.00 | -8.61% | -4.75% | 吞吐 |
| deep4_conservative | SpecCascade | +0.00 | +1.37% | +4.37% | 整题耗时 |

目前质量最好的deep4_balanced相对Cactus仍吞吐−7.13%、整题+8.46%；相对SpecCascade吞吐+3.01%，但整题+18.84%。conservative与Cactus/SpecCascade同为14/16正确，整题比Cactus短4.75%，但吞吐比Cactus低8.61%，整题比SpecCascade长4.37%。联合门槛失败来自这些实际指标；截断未参与否决。

### 按seed诊断

| 方法 | base seed42 正确 | base seed43 正确 |
| --- | ---: | ---: |
| Native | 7/8 | 6/8 |
| Cactus | 8/8 | 6/8 |
| SpecCascade | 8/8 | 6/8 |
| 原 Adaptive RAVEN（B） | 6/8 | 6/8 |
| balanced3（D3） | 4/8 | 7/8 |
| deep4_balanced（D4） | 8/8 | 8/8 |
| deep4_conservative（D4） | 8/8 | 6/8 |

deep4_balanced对B恢复4个原判错配对，没有新增原评分损失；每个seed均8/8。每题两个seed相关，统计单位是8个题簇，不是16独立题。它16个原判对中有4个finish=length，仍按用户指定的原评分计入，不添加正常结束门槛。

## 5. 为什么仍未超过三个基线

### 接受效率改善了，树的轮成本仍较高

deep4_balanced相对Cactus：MAL3.8883对3.3436（+16.29%），decode轮成本1.4411对1.1507秒（+25.24%）。更高接受收益没有覆盖更大的轮成本，导致E2E吞吐仍低7.13%。这是服务器聚合计数和轮耗时所支持的成本判断，本单元未做组件profiling，不能据此指认某个kernel或KV函数就是全部根因。

对B，MAL增长10.71%，轮成本增长8.06%，因此吞吐点估计改善2.36%。单纯收紧D3的δ/mix并没有改善质量；增加到D4/N12后的组合带来了本批更好的接受/质量表现。该实验比较参数组合，尚不能拆分深度、节点数、预算池与自适应的各自贡献。

### 回答长度决定整题耗时

deep4_balanced输出18799 token，SpecCascade15357 token；前者输出多22.41%，吞吐只高3.01%，所以整题更慢18.84%。conservative输出16247 token，比balanced少13.57%，整题更短，但正确数16→14。不能把整题耗时只解释为解码吞吐差。

### 小幅速度收益尚不稳定

按8题聚类2000次bootstrap，deep4_balanced相对B的吞吐比95%区间为[0.9993, 1.0438]，仍包含1；相对Native为[1.0390, 1.0851]，相对Cactus为[0.9081, 0.9549]。这支持本批对Native的速度收益与对Cactus的速度落后，不能把对B的+2.36%点估计称为已证明稳定提速。区间是开发数据描述，未消除选参/多候选选择偏差。

## 6. 下一步最值得推进的方向

保留两个D4候选：balanced提供质量参照，conservative距离完整速度目标更近。下一轮重点是RAVEN额外proposal/树/同步/KV成本的实测与计算复用；验证行为等价的优化时保留完整输出路径与评分，公共内核若改变应让所有基线同步应用。

当前基线刻度：正确≥14/16、吞吐>2.8932、平均整题<367.98秒。假设输出token数和判分保持本轮结果，且只是缩短时间：

- **deep4_balanced：**平均输出1174.94token/题。只追平Cactus吞吐还不足以超过SpecCascade整题耗时；需吞吐>3.1929，即较当前再提升约18.84%。
- **deep4_conservative：**平均输出1015.44token/题。需吞吐>2.8932，即再提升约9.43%；保持输出下也能满足当前整题刻度。因此它适合优先做成本优化，balanced作为质量备选。

上述是固定本轮输出与基线数值的工程距离计算，不保证对未见题可实现，也不保证公共优化后的基线保持不变。单元内先补组件成本/选择下一具体实现，再登记有限预算；本次只检查已完成结果，没有启动下一实验。

尚需代表性验证、三个基线同等D3/D4选参预算、独立留出/正式多seed。8道已观察训练题上的100%原评分不是论文结论。

## 7. 复查文件

- [原控制器报告](RESULT-RAVEN-DOMINANCE-LIVE.md)、[原analysis](../results/raven_dominance_search_20261009/analysis.json)、[原ledger](../results/raven_dominance_search_20261009/ledger.json)。
- [独立CPU核验脚本](../results/raven_dominance_search_20261009/offline_audit_20261010.py)、[核验JSON与输入hash](../results/raven_dominance_search_20261009/offline_analysis_20261010.json)。
- [可导出CSV汇总](../results/raven_dominance_search_20261009/summary_checked_20261010.csv)、[固定输出下的优化距离](../results/raven_dominance_search_20261009/optimization_distance_20261010.json)。
- [本轮计划](../plan/adaptive-raven-dominance-2026-10-09.md)、[旧原版本100题专表](RESULT-RAVEN-ORIGINAL-VS-BASELINES-2026-10-09.md)、[历史最佳汇总](RESULT-RAVEN-HISTORICAL-BEST-2026-10-09.md)。旧100题与本轮训练题不能跨表直接比较。

原analysis/gate/冻结计划/原始payload未覆盖；当前默认未自动切换候选，原负结果与历史证据保留。
