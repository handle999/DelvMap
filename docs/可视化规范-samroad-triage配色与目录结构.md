# 可视化规范 — samroad triage 配色与目录结构

> 主题：评估可视化的配色、画法、输出目录结构，对齐 samroad 官方规范。
> 影响文件：`adaptive_map_completion/viz_region.py`、`adaptive_map_completion/run_didi_eval.py`

## 配色规范（samroad triage）

来源：samroad `tools/viz_compare_samroad_p2cnet.py`，commit `e1ed765 refactor(viz): 对比图配色回归 triage 风格`。

| 元素 | RGB | BGR (cv2) | 尺寸 |
|------|-----|-----------|------|
| graph 边 | (253,160,15) 橙黄 | (15,160,253) | thickness=4 |
| graph 节点 | (255,255,0) 黄 | (0,255,255) | radius=4 |
| 抗锯齿 | — | — | cv2.LINE_AA |

> 注：samroad inferencer `postprocess/triage.visualize_image_and_graph` 用的是这套配色的等价 BGR 值（边 (15,160,253)、节点 (0,255,255)）。

## 画法

```python
# 边: (row,col) → (x=col, y=row), LINE_AA
cv2.line(img, (int(c1), int(r1)), (int(c2), int(r2)), (15,160,253), 4, cv2.LINE_AA)
# 节点
cv2.circle(img, (c, r), 4, (0,255,255), -1, cv2.LINE_AA)
```

didi 图字典节点是 (row,col)∈[0,400]，直接当像素坐标（无缩放/重投影），因 didi region 本就是 400px 栅格。

## 输出目录结构（对齐 samroad infer）

参考 `sam_road/save/tmp/smoke_infer/`，`run_didi_eval.py` 在 amc 模式为每个 region 生成：

```
{work_dir}/
├── graph/{idx}.p          # fused 路网 pickle
├── mask/{idx}_road.png    # fused 渲染的二值 road mask (road=255, 背景=0)
├── viz/{idx}.png          # sat + fused 叠加 (triage 配色)
└── results/               # 评估指标 json (apls.json, topo.json)
```

- 只保存 **fused**（本方法输出）。partial 是 input 不保存，GT 不可视化。
- `mask/{idx}_road.png` 用 cv2 画白线到 400×400，road=255。

## viz_region.py（单 region 调试工具）

独立脚本，单图保存（不拼接），对齐 `viz/{idx}.png` 命名，用于单 region 调试：

```bash
python adaptive_map_completion/viz_region.py --region 92 --out_dir experiments/viz
```

输出：
- `{idx}_partial.png` — sat + partial（triage 配色）
- `{idx}_fused.png` — sat + fused（triage 配色）
- `{idx}_diff.png` — sat + partial(黄) vs fused(绿) 叠加，看补全增量

> 批量评估时的 mask/viz 由 `run_didi_eval.py` 统一生成，本脚本仅调试用。

## diff 图配色
partial(黄) vs fused(绿) 叠加看补全增量时，fused 用粗线（thickness=3）盖住重合的 partial：
- `DIFF_PARTIAL = (0,255,255)` 黄
- `DIFF_FUSED = (0,255,0)` 绿
