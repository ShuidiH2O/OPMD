# OPMD 与 NTP 差异：真实训练案例

本文件用于论文中直观展示 On-Policy Distillation（OPMD）与固定 Next-Token Prediction（NTP）两种 mid-training 目标的区别。所有内容均来自真实训练数据与真实模型前向，未经人工改写。

## 实验来源与符号约定

- 数据：MegaMath-Web-Pro 的 OPMD `rand-0.05` 训练集（`megamath_web_pro_opmd_rand_0.05_train.parquet`），对应 14B teacher → 1.7B student 实验。
- Student：Qwen3-1.7B-Base 的 OPMD checkpoint（`global_step_1000`，已合并为 HuggingFace 格式）。
- Teacher：Qwen3-14B-Base（冻结，仅用于打分）。
- Anchor：由代理模型（Distill-1.5B）逐 token 计算预测熵、`rand` 臂随机采样的目标 token。`anchor_entropy` 为该 token 的代理模型预测熵；熵越高说明该位置越难预测、越适合作为"思考点"。
- 生成设置：与训练 rollout 一致，`temperature=1.0`、`top_p=0.95`；示例截取 `max_new_tokens=128`。
- Token 级信号定义（下文表中的 `reward`）：`log π_teacher(token) − log π_student(token)`。训练中该信号再经 top-16(`only_stu`) 与 teacher-p 加权后作为优势（`token_reward_direct`），此处直接展示最核心的逐 token 差异。
- `i` 为从 anchor 处开始的生成位置编号。

---

## 案例一：关键位置走偏，教师信号集中于分叉处

这个案例对应"学生在某个关键位置开始走偏、教师信号集中出现在该分叉处"。

### 1. 原始语料片段与被选中的 anchor

原始语料是 Go 图像抖动（dithering）实现的代码片段（MegaMath 语料中混入的编程内容）。被选中的 anchor 是 token `(gray`，位于 `rgbaToGray` 之后：

- `anchor_token` = `(gray`
- `anchor_entropy` = `8.0625`（高熵，代理模型认为此处极难预测）
- `anchor_position` = `582`，`doc_index` = `297`

### 2. 实际输入学生的 prefix（anchor 之前，截取末尾关键上下文）

完整 prefix 为 anchor 前 1024 个 token，此处展示末尾关键部分：

```text
…The following code snippet demonstrates the implementation of grid distribution:
```go
func GridDither(gray *image.Gray, k int, gamma float64) *image.Gray {
    var (
        bounds = gray.Bounds()
        dithered = newWhite(bounds)
    )
    for i := 0; i < bounds.Max.X; i += k {
        for j := 0; j < bounds.Max.Y; j += k {
            var (
                cell = rgbaToGray
```

### 3. 原始 corpus continuation（NTP 的固定目标）

若用标准 NTP，学生应被强制预测如下固定文本（从 anchor `(gray` 开始）：

```text
(gray.SubImage(image.Rect(i, j, i+k, j+k)))
                mu = avgIntensity(cell)
                n = (1-mu)*gamma
            )
            for k := 0; k < int(n); k++ {
                var (
                    x = randInt(i, min(i+k, bounds.Max.X), rng)
                    y = randInt(j, min(j+k, bounds.Max.Y), rng)
                )
                dithered.SetGray(x, y, color.Gray{0})
            }
        }
    }
    return dithered
}

func avgIntensity(gray *image.Gray) float64 {
```

### 4. 学生在该 prefix 上生成的 rollout

学生（OPMD 训练的 1.7B）实际生成了与 Go 上下文完全无关的 JavaScript 内容：

```text
.createTextNode("No results found.")
  return
}

if !isEmpty(s) {
  for _, s := range s {
    h, err := s.split()
    if err != nil {
      continue
    }

    if len(h) >= 2 {
      res = append(res, h[1])
    }
  }
}

return res
```

## Stippling

Stippling is a halftoning technique that uses small dots to represent the image's intensity...
```

### 5. 分叉处的逐 token 信号

关键分叉出现在 `i = 0`：正确答案应首先生成 `(gray`（完整的 Go 调用 `rgbaToGray(gray.SubImage(...))`），学生却生成了 `.createTextNode`（JavaScript DOM API）：

| `i` | 生成 token | student logp | teacher logp | reward = teacher − student |
|---|---|---|---|---|
| 0 | `.createTextNode` | −4.78 | −19.57 | **−14.78** |
| 1 | `("` | −1.89 | −1.68 | +0.21 |
| 2 | `No` | −5.72 | −7.12 | −1.40 |
| 3 | ` results` | −5.92 | −4.80 | +1.12 |
| 4 | ` found` | −0.34 | −0.23 | +0.10 |
| 5 | `.")\n` | −1.90 | −2.35 | −0.45 |
| 6 | ` ` | −5.29 | −4.27 | +1.02 |
| 7 | ` return` | −2.67 | −4.76 | −2.09 |
| 8 | `\n` | −1.59 | −2.33 | −0.74 |

### 6. 关键标注

- **分叉点**：`i = 0`。教师给出全场最大的负 reward（−14.78），直接把这一"走偏"压下去——教师强烈认为 `.createTextNode` 在该上下文中几乎不可能。
- **后续位置**：一旦走偏，后续 token 的 `reward` 迅速回落到 ±1 量级且正负交错，说明教师的主要纠错信号集中在分叉处，而不是均匀分布在整个 rollout。
- **两种目标的差异**：NTP 会硬性迫使模型背下 `(gray.SubImage(image.Rect(...))` 这段固定 Go 代码序列；OPMD 在 `i=0` 处只给 token 级软信号，让学生自己的探索分布碰壁，但教师并不重新生成"标准答案"。

---

## 案例二：原始语料后缀带噪声，学生生成不同但合理的路径

这个案例对应"原始语料后缀存在噪声，或学生生成了不同但合理的路径，能体现固定 NTP target 与 teacher-guided on-policy learning 的差别"。

### 1. 原始语料片段与被选中的 anchor

原始语料是一个 GMAT 论坛帖子，包含一道"排列组合"题（5 个孩子中 2 个是兄弟、不能相邻坐），随后是论坛元数据。被选中的 anchor 是 token `some`：

- `anchor_token` = `some`
- `anchor_entropy` = `5.875`
- `anchor_position` = `421`，`doc_index` = `451203`

### 2. 实际输入学生的 prefix（截取末尾关键上下文）

```text
…Mon Feb 19, 2007 8:15 am
Among 5 children there are 2 siblings. In how many ways can the children be seated in a row so that the siblings do not sit together?
(A) 38
(B) 46
(C) 72
(D) 86
(E) 102

oa coming when
```

注意：prefix 末尾 `oa coming when` 已经是一个残缺的、语义断裂的片段（原始语料噪声）。

### 3. 原始 corpus continuation（NTP 的固定目标）

若用标准 NTP，模型必须预测下面这段噪声明显的固定文本：

```text
 some people respond with explanations. from diff math doc

Neo2000 Legendary Member
Joined
27 Jan 2007
Posted:
519 messages
31
Test Date:
30/09
Target GMAT Score:
710
Mon Feb 19, 2007 8:51 am
The 5 kids can be arranged among themselves in 5! ways

Consider the 2 siblings as 1. Then 4 kids can be arranged among themselves in 4! x 2! ways ( since the 2 can be arranged among themselves...
```

其中 `from diff math doc` 是语料中的注释性元数据/噪声，并非数学内容本身。

### 4. 学生在该 prefix 上生成的 rollout

学生没有逐字复现上面那段论坛噪声，而是生成了一条"论坛化"的解题回复，包含枚举与最终答案：

```text
1 2 3

800guy Master | Next Rank: 500 Posts
Joined
27 Jun 2006
Posted:
354 messages
Followed by:
5 members
11
The solution to this question is
120-16=104

1 2 3
a b a c d
d a b c a

b a c d a
b c a d a

c b a d a
c a b d a

d b a c a
d a c b a

132 different ways
```

### 5. 分叉处的逐 token 信号

| `i` | 生成 token | student logp | teacher logp | reward = teacher − student |
|---|---|---|---|---|
| 0 | `1` | −5.28 | −6.79 | −1.51 |
| 1 | ` ` | −2.38 | −4.65 | −2.27 |
| 2 | `2` | −0.25 | −0.61 | −0.36 |
| 3 | ` ` | −0.28 | −1.21 | −0.93 |
| 4 | `3` | −0.03 | −0.05 | −0.02 |
| 5 | `\n\n` | −2.94 | −2.86 | +0.07 |
| 6 | `8` | −2.29 | −3.97 | −1.68 |
| 7 | `0` | −0.01 | −0.00 | +0.01 |
| 8 | `0` | −0.00 | −0.00 | 0.00 |
| 9 | `g` | −0.00 | −0.00 | +0.00 |
| 10 | `uy` | −0.00 | −0.00 | −0.00 |
| 11 | ` Master` | −0.01 | −0.39 | −0.37 |

从 `i = 6` 开始，学生进入"复刻论坛模板"模式（`800guy Master | Next Rank...`），teacher 与 student 的 logp 都接近 0（模型在对已学过的模板高置信地抄写）。

### 6. 关键标注

- **原文噪声**：correct continuation 开头出现 `oa coming when` → `some people respond with explanations. from diff math doc`，属于断裂残句 + 元数据噪声。
- **NTP 的问题**：固定 target 会迫使模型把这段噪声原样背下来，包括 `from diff math doc` 这类与数学无关的内容。
- **OPMD 的差别**：教师只对 student 的探索 token 给软信号（`i=0` 的 `1` 得到 −1.51 的轻微负反馈），但不会把学生的生成强行拉回到那段噪声固定 track；学生最终生成的是"论坛格式 + 排列枚举 + 答案 132"——与原文不同的、但语义上更偏"解题回复"的路径。这正体现了 teacher-guided on-policy learning 相对固定 hard-label 的差别。

---

## 完整数据

上述两个案例的完整 128 步逐 token `student_logp / teacher_logp / reward` 数据保存在：

- `/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/eval_results/example_rollouts_raw.json`

如需在论文中复现或扩展更多位置，可用该文件直接查询任意 token 的信号。