# 残差对齐树式 MTP

## 1. 方法概述

本文提出一套面向原生多 Token 预测模型的树式松弛解码方法：

> **Residual-Aligned Tree MTP，残差对齐树式 MTP。**

它解决的核心问题是：单链 MTP 一旦在中间位置被拒绝，本轮已经完成的后续草拟和目标
验证通常全部失效；直接从多个树候选中挑选一个继续生成，又容易让候选树替代目标模型
做决定，从而损害输出质量。

本方法把一次 speculative round 分成四个紧密衔接的阶段：

1. MTP 生成一条随机主路径，并在每个主路径位置保留少量高概率纠正分支；
2. 目标模型通过一次 Tree Attention forward 并行验证整棵稀疏树；
3. 主路径采用候选条件化的受控松弛分布进行概率验证；
4. 主路径拒绝时，先从验证残差中确定纠正 token，再检查树是否已经覆盖该 token。

树不能选择纠正 token。它只能在纠正 token 已经由概率验证过程确定以后，复用一个
token ID 完全相同的分支。若命中，则利用已经算好的目标状态额外生成一个目标锚点；
若没有命中，则退化为普通的单链残差纠正。

因此，本方法的核心不是“从更多候选里挑一个更容易接受的”，而是：

> **让候选树提前覆盖可能发生的目标纠正，并在纠正确实发生时回收已经完成的验证计算。**

---

## 2. 基本记号

设当前真实前缀为 \(s\)，MTP 最大递归深度为 \(D\)。在第 \(i\) 个位置：

- \(Q_i(x)\)：MTP 在对应父前缀下的完整提议分布；
- \(P_i(x)\)：目标模型在同一父前缀下的原始分布；
- \(y_i\sim Q_i\)：主路径实际采样的草稿 token；
- \(H_i\)：针对 \(y_i\) 构造的临时松弛验证分布；
- \(R_i\)：拒绝 \(y_i\) 后使用的残差纠正分布。

当前实现使用深度 \(D=3\)，候选树最多包含 9 个 MTP 节点，每个父节点最多保留 3 个
候选。节点预算是上限而不是必须填满的配额。

---

## 3. 阶段一：构造纠正覆盖树

### 3.1 随机主路径

MTP 首先从完整分布中逐层采样：

\[
y_1\sim Q_1,\qquad
y_2\sim Q_2(\cdot\mid y_1),\qquad
\ldots,\qquad
y_D\sim Q_D(\cdot\mid y_{<D}).
\]

这些 token 构成主路径：

```text
y1 -> y2 -> ... -> yD
```

主路径保留完整的 \(Q_i\)，后续接受概率和残差计算必须使用与实际采样一致的分布。

### 3.2 高概率纠正分支

在每个主路径父节点下，除实际采样的 \(y_i\) 外，再从 \(Q_i\) 中保留少量高概率
backup token：

```text
parent
  ├─ yi          # 实际采样的主路径 token
  ├─ backup-1    # 高 Q 候选
  └─ backup-2    # 高 Q 候选
```

backup 必须同时满足绝对概率下限和相对 top-1 概率下限。树宽由 MTP 自身的不确定度决定，
构树阶段不能访问目标分布 \(P\)，避免使用验证信息反向选择草稿。

全局节点不足时，分配器优先保证：

1. 主路径能够达到最大深度；
2. 每个主路径拒绝位置都有机会保留纠正候选；
3. 已保留纠正分支可以拥有必要的状态节点；
4. 最后才增加更低概率的横向候选。

该结构不是追求覆盖所有 MTP 高概率路径，而是专门提高“残差纠正 token 已经存在于树中”
的概率。

---

## 4. 阶段二：一次目标模型树验证

整棵树构造完成后，目标模型通过一次 Tree Attention forward 验证全部节点。

对于任意树节点 \(v\)：

- 它只能关注公共真实前缀和自己的祖先节点；
- 不能关注同层兄弟或其他分支；
- position ID 等于真实前缀位置加树深度；
- 目标 logits 对应其父节点代表的真实条件前缀。

因此，目标模型得到的是：

\[
P_v(x)=P_{\text{target}}(x\mid s+\operatorname{path}(parent(v))).
\]

一次 forward 同时产生：

- 主路径每个位置的目标分布；
- 每个 backup token 所在正确父前缀下的目标状态；
- 每个已验证分支末端的下一 token 分布。

目标模型调用次数仍为每轮一次，增加的是该次 forward 内的树节点数。

---

## 5. 阶段三：主路径的受控松弛验证

### 5.1 构造候选条件化验证分布

对主路径候选 \(y_i\)，记其原始目标概率为：

\[
p_i=P_i(y_i).
\]

使用松弛强度 \(\delta\) 将候选概率提高为：

\[
\gamma_i=
\min\left(
p_i+\sqrt{2\delta p_i(1-p_i)},
1
\right).
\]

临时验证分布 \(H_i\) 定义为：

\[
H_i(y_i)=\gamma_i,
\]

\[
H_i(x)=P_i(x)\frac{1-\gamma_i}{1-p_i},
\qquad x\neq y_i.
\]

也就是说，只提高当前草稿 token 的概率，其余 token 按相同比例缩小，保持它们之间的
相对顺序不变。\(\delta=0\) 时恢复原始目标分布；\(\delta\) 越大，主路径越容易通过。

### 5.2 概率接受

由于 \(y_i\) 是从 \(Q_i\) 中采样得到的，其接受概率为：

\[
A_i=min\left(1,\frac{H_i(y_i)}{Q_i(y_i)}\right).
\]

采样 \(u_i\sim U(0,1)\)：

- 若 \(u_i\le A_i\)，接受 \(y_i\)，继续验证主路径下一位置；
- 否则拒绝 \(y_i\)，停止验证后续主路径并进入残差纠正。

如果全部 \(D\) 个主路径 token 都被接受，则从最后一个节点后的原始目标分布采样一个
bonus token，并结束本轮。

---

## 6. 阶段四：残差对齐与树状态复用

### 6.1 先确定纠正 token

当 \(y_i\) 被拒绝时，构造残差分布：

\[
R_i(x)=
\frac{\max(H_i(x)-Q_i(x),0)}
{\sum_z\max(H_i(z)-Q_i(z),0)}.
\]

随后先完成纠正采样：

\[
c_i\sim R_i.
\]

在这一刻，当前真实前缀接下来应该提交什么 token 已经确定。树尚未参与任何选择。

### 6.2 精确残差命中

设当前主路径节点的兄弟候选集合为 \(B_i\)。只检查：

\[
c_i\in B_i\ ?
\]

这里要求 token ID 完全相同，不使用语义相似度、目标排名、近似 margin 或路径评分。

#### 未命中

若 \(c_i\notin B_i\)，提交：

```text
已经接受的主路径前缀 + correction ci
```

本轮立即结束。树不会改变输出。

#### 精确命中

若树中存在某个 sibling \(b_i=c_i\)，则：

1. 提交同一个 correction \(c_i\)；
2. 复用该 sibling 在 Tree Attention 中已经计算的目标 hidden 和 KV 状态；
3. 从该分支后的原始目标分布采样一个 anchor token：

\[
z_i\sim P_{b_i+1}(\cdot);
\]

4. 本轮最终提交：

```text
已经接受的主路径前缀 + correction ci + target anchor zi
```

相较未命中情况，精确命中额外获得一个 token。这个 token 来自原始目标分布，不再进行
松弛。

---

## 7. 为什么树不会替验证器做决定

一个看似自然但风险很高的做法是：主路径拒绝后，在多个 sibling 中选择目标概率最高或
路径最长的一个继续。该做法让“树是否包含某个候选”直接改变输出 token，容易形成明显
的分布偏移。

本方法严格固定因果顺序：

```text
先完成概率拒绝
    -> 再从残差分布采样 correction
        -> 最后查询树中是否已经存在同一个 token
```

因此树查询只能影响“是否能提前多走一步”，不能影响“纠正方向是什么”。

从概率上看，主路径的接受—拒绝—残差采样恢复的是临时验证分布 \(H_i\)。精确命中时，
输出的 correction 仍然是已经从 \(R_i\) 中采样的 \(c_i\)，之后追加的 anchor 又来自
实际前缀下的原始 \(P\)。所以树状态复用不会在受控松弛之外增加新的 token 选择偏移。

---

## 8. 平均接受长度收益

设普通单链在一轮内接受的主路径 token 数为 \(L\)，无论主路径完整通过还是发生拒绝，
本轮都会产生一个 bonus 或 correction token，因此基础 MAL 为：

\[
\operatorname{MAL}_{base}=1+\mathbb{E}[L].
\]

设残差纠正精确命中树分支的事件为 \(E_{hit}\)。当前版本每次命中额外提交一个 target
anchor，因此：

\[
\operatorname{MAL}_{tree}
=1+\mathbb{E}[L]+\Pr(E_{hit}).
\]

这给出了非常直接的优化目标：

> 在不改变主路径概率语义的条件下，提高残差纠正 token 的树覆盖率。

当前固定 20 题 pilot 中，精确命中约发生在全部 speculative rounds 的 21%，因此树
复用带来约 \(+0.21\) MAL：

| 数据集 | 主路径贡献 | 精确命中贡献 | 最终 MAL |
|---|---:|---:|---:|
| GSM8K | 2.359 | +0.210 | 3.569 |
| HumanEval | 2.330 | +0.209 | 3.540 |

表中的最终 MAL 还包含每轮固定的 correction、bonus 或 anchor 项 1。

---

## 9. 正确的跨轮状态提交

树式验证不仅要选对 token，还必须提交与所选路径一致的模型状态。

树节点按 BFS 顺序存储，target forward 的 row 0 对应公共 anchor，树节点 \(j\) 的输出
状态对应 row \(j+1\)。若最终路径为：

```text
[0, 2, 5]
```

下一轮 MTP 的 root hidden 必须读取：

```text
selected_leaf_index + 1 = 5 + 1 = row 6
```

不能根据“接受了三个 token”错误地读取 row 3。后者属于单链假设，会把另一个兄弟路径
的 hidden 送入下一轮 MTP，使 \(Q\) 从第二轮开始持续漂移。

完整状态提交需要同步处理：

- 选中路径对应的目标 KV cache；
- Tree Attention 节点到线性真实前缀的压缩；
- 选中叶节点的 target hidden；
- 分支局部 MTP 递归状态；
- position ID 与父节点映射；
- 未选中 sibling 状态的丢弃。

状态正确性不是单纯的工程优化，而是算法能够维持后续接受长度的必要条件。

---

## 10. 完整伪代码

```python
def residual_aligned_tree_round(prefix, mtp, target, delta):
    # 1. MTP-only proposal. Target information is unavailable here.
    tree = build_sparse_correction_tree(
        prefix=prefix,
        depth=3,
        max_nodes=9,
        max_children=3,
    )

    # 2. One target forward verifies every node under its own ancestry.
    target_rows, target_states = target.tree_forward(prefix, tree)

    committed = []

    # 3. Verify only the sampled primary path.
    for node in tree.sampled_primary_path:
        q = node.full_mtp_distribution
        p = target_rows[node.parent_row]
        y = node.token_id

        h = candidate_conditioned_relaxation(p, y, delta)
        accept_prob = min(1.0, h[y] / max(q[y], 1e-30))

        if uniform_random() <= accept_prob:
            committed.append(y)
            continue

        # 4. The correction is sampled before the tree is consulted.
        residual = normalize(maximum(h - q, 0.0))
        correction = sample(residual)
        committed.append(correction)

        hit = tree.find_sibling_with_same_token(
            parent=node.parent,
            token_id=correction,
        )
        if hit is None:
            return commit(committed, terminal_state="correction")

        # Reuse the already verified target state and append a target anchor.
        anchor = sample(target_rows[hit.next_target_row])
        committed.append(anchor)
        return commit(
            committed,
            selected_leaf=hit,
            terminal_state="residual-hit-anchor",
        )

    # Full primary-path acceptance produces the ordinary target bonus.
    bonus = sample(target_rows[tree.primary_leaf.next_target_row])
    committed.append(bonus)
    return commit(
        committed,
        selected_leaf=tree.primary_leaf,
        terminal_state="full-accept-bonus",
    )
```

---

## 11. 当前推荐配置

| 参数 | 当前值 | 作用 |
|---|---:|---|
| MTP 逻辑深度 | 3 | 主路径最大草拟长度 |
| 最大树节点数 | 9 | 限制一次 target forward 的节点预算 |
| 每父节点最大候选 | 3 | 一条主候选加少量 correction backup |
| backup/top-1 最低比例 | 0.01 | 删除极弱 sibling |
| MTP 最低绝对概率 | 0.001 | 删除几乎不可能命中的 token |
| 松弛强度 \(\delta\) | 1.0 | 控制主路径候选概率提升 |
| 路径延长 | 1 个 target anchor | 精确命中后的低风险扩展 |
| target forward | 每轮 1 次 | 与单轮推测验证保持一致 |

这些值是当前 pilot 的工程起点，不应被解释为跨模型最优参数。更换模型、温度、MTP
深度或任务后，应重新测量残差命中率、节点利用率和质量—MAL 曲线。

---

## 12. 方法边界与后续优化

当前版本已经证明：候选树的价值可以来自纠正状态复用，而不必来自更激进的多路径选路。
但仍存在以下限制：

1. 树平均验证约 6 个节点，而主路径只有 3 个，当前 Python/vLLM 实现的吞吐收益尚未
   覆盖额外节点成本；
2. backup 只能使用构树时可见的 \(Q\) 作为残差命中代理，无法提前知道真实 \(R\)；
3. 当前每次命中只追加一个 target anchor，没有继续复用更深的纠正分支；
4. 小样本结果不能证明跨数据集、跨 seed 的普遍质量优势。

后续优化应优先围绕：

- 用更少节点保持相同的残差命中率；
- 将构树、Tree Attention metadata 和状态压缩融合到 GPU kernel；
- 统计各深度的残差命中贡献，按位置动态分配 backup；
- 在不改变 correction 的前提下研究严格验证的多步状态复用；
- 使用独立样本和多个 seed 验证质量与 MAL。

整套方法可以概括为一句话：

> **主路径用受控松弛提高通过率，拒绝残差决定真实纠正方向，稀疏树提前覆盖纠正状态，
> 精确命中后用原始目标分布安全延长。**
