# 评估指标与诊断 — APLS / TOPO 与碎片根因

> 主题：APLS / TOPO 指标含义、AMC baseline 的指标表现、4-邻接碎片根因与 8-邻接修复。
> 关键结论：4-邻接碎片是 APLS 偏低 + viz 节点过密的共同根因，改用带对角保护的 8-邻接后，AMC 在 APLS/TOPO 上全面超越 partial。

## 指标定义

### APLS（Average Path Length Similarity）
- 在 GT/pred 图上各选控制点（图节点），按最近邻匹配，用 Dijkstra 算两图最短路长度
- `APLS = mean(1 - min(d_pred/d_gt, d_gt/d_pred))`
- **对拓扑连通性极敏感**：不可达节点对当 0 分
- 由 Go binary 实现（`metrics/apls/main.go`，现场编译）

### TOPO（拓扑匹配）
- pred 边中能匹配到 GT 边 buffer 内的比例（Precision）
- GT 边被 pred 覆盖的比例（Recall）
- **局部匹配，不要求连通**
- 由 `metrics/topo/` Python 实现

## AMC baseline 最终指标（全量 test split，34/39）

| 模式 | APLS | TOPO F1 | TOPO P | TOPO R |
|------|------|---------|--------|--------|
| AMC (fused, 8-邻接) | **0.812** | **0.841** | 0.977 | **0.738** |
| partial baseline | 0.683 | 0.586 | 0.996 | 0.415 |

**结论**：修碎片后 AMC 在 APLS 和 TOPO 上全面碾压 partial——补全既提升覆盖（TOPO R +0.32），又保持连通（APLS +0.13）。

## 根因诊断：4-邻接碎片（已修）

### 现象（修复前）
4-邻接时代 AMC APLS 仅 0.617 < partial 0.683，且 viz 里 fused 节点过密（region_303 边长从 partial 的 22m 崩到 3m）。

### 曾误判：AMC fuse 炸碎路网（已纠正）
最初发现 fused 有 19 个连通分量（partial 仅 4 个），误以为 AMC 把路炸碎。**这是错的**：
```
inferred_rn (densify前):  87节点 53边  34分量   ← fuse 之前就 34 分量!
fused (AMC后未压缩):      120节点 101边 19分量   ← AMC 反而 34→19, 在修碎片!
```
AMC fuse 是在合并碎片，不是制造碎片。

### 真根因：4-邻接切断对角过渡
`raster_to_shp.skeleton_to_graph` 原用 **4-邻接**建像素图，骨架里 40 处对角像素过渡被切断：
```
骨架连通分量: 4-邻接=42  8-邻接=2   ← 40 个碎片是 4-邻接副作用
```

#### 4-邻接 vs 8-邻接
- **4-邻接**：只认上下左右 4 个正交方向，对角线不算相邻
- **8-邻接**：4 正交 + 4 对角，8 个全算相邻

一条 45° 斜路的骨架像素是阶梯状对角排列（如 (0,0) 和 (1,1)），4-邻接下不相邻 → 路断开；8-邻接下相邻 → 路连通。

### 碎片的两个表现（同一根因）
1. **APLS 偏低**：inferred 碎成多段，孤立段节点对最短路不可达 → APLS 扣分
2. **viz 节点过密**：碎片产生大量 degree=1 孤儿端点（region_303 有 183 个），AMC 投影连不上 partial，堆成密集节点（边长崩到 3m）

### 为何 TOPO 不受碎片影响
TOPO 是局部边匹配，不要求连通——inferred 的边即便碎成段，每段仍能和 GT 局部匹配，所以补全反而提升 TOPO。

## 修复：带对角保护的 8-邻接

纯 8-邻接会在十字/T字交叉处把正交线的相邻像素通过对角连成虚假高 degree 节点（region_0 实测出现 deg>=5）。解法：**8-邻接建图，但对角边加保护**——仅当对角相邻两像素没有共同的正交邻居时才连对角。

`raster_to_shp._build_pixel_graph`：
- 正交方向（4-邻接）无条件连
- 对角方向：仅当两端无共同正交邻居时连（有共同正交邻居 = 交叉处擦肩，不连）

共同正交邻居判定：对角像素 p=(y,x) 与 q=(y+dy,x+dx)，检查 (y+dy,x) 和 (y,x+dx) 这两个正交位是否都是骨架——若是，说明 p、q 在十字交叉处擦肩，不连对角。

### 验证
- 节点密度回归：fused 边长 20-22m ≈ partial 18-21m（region_303 从 3m 回到 21m）
- 零虚假交叉：所有测试 region deg>=5 = 0
- 全量 APLS 0.617 → 0.812，TOPO 0.699 → 0.841

## compress 节点密度问题（已修）

`DelvMapConnector(out_compressed=True)` 会无差别合并所有 degree=2 中间节点，把 partial 原本合理的节点（间距 21m）也合并掉，使 fused 节点稀疏（边长翻倍到 39m），APLS 控制点对不齐。评估用 `out_compressed=False`（`run_didi_eval.py` + `viz_region.py`）。

## 相关文件
- `adaptive_map_completion/raster_to_shp.py`（skeleton_to_graph / _build_pixel_graph，8-邻接+对角保护）
- `adaptive_map_completion/adaptive_map_completion.py`（adaptive_fuse / compress_rn）
