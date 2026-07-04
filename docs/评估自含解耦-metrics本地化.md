# 评估自含解耦 — metrics 本地化

> 主题：把 samroad 的 metrics 评估代码与 didi_xian 数据 copy 进 DelvMap，消除跨项目耦合与软链接。
> 影响文件：`metrics/`、`datasets/didi/xian/`、`adaptive_map_completion/run_didi_eval.py`、`adaptive_map_completion/didi_bridge.py`、`adaptive_map_completion/viz_region.py`

## 背景

最初评估链路依赖 samroad 项目：
- `run_didi_eval.py` 调 `sam_road/metrics/eval.py`
- eval.py 的 `--dir` 强制要求相对 samroad project root，且内部用 `../{dir}/graph/{idx}.p`（cwd=metrics/）
- DelvMap 的 work_dir 不在 samroad 下，所以 `run_didi_eval.py` 用 `os.symlink` 在 samroad 根建软链接指向 DelvMap 实验目录

问题：软链接残留（评估完不清理），samroad 根堆了 11 个软链接，跨项目耦合。

## 解耦方案

把 samroad 评估所需的一切 copy 进 DelvMap，同路径放置，**eval.py 一个字都不用改**：

### 1. metrics/ （评估代码）
copy 自 `sam_road/metrics/`，只保留 didi_xian 相关：
```
metrics/
├── eval.py                 # 主评估入口 (APLS + TOPO)
├── configs/didi_xian.yaml  # didi_xian 配置
├── topo/                   # TOPO 评估 (Python)
│   ├── eval_parallel.py
│   ├── main.py
│   ├── topo.py / graph.py / HopcroftKarp.py / showTOPO.py
└── apls/                   # APLS 评估 (Go, 现场编译)
    ├── main.go / go.mod / go.sum
    └── convert.py
```
- APLS Go binary 每次 eval 现场编译（`go build -o eval_bin main.go`），不依赖预编译
- 删掉了 cityscale/spacenet 配置（不需要）

### 2. datasets/didi/xian/ （数据）
整个 copy 自 `sam_road/datasets/didi/xian/`（394M），**同路径** `datasets/didi/xian/`：
- `2019_400/` — region sat/gt/partial/active/traj（378 region）
- `data_split.json` — train/val/test 划分
- `osm/`、`processed/`

同路径放置使 eval.py / topo 里所有 `../datasets/didi/xian/...` 硬编码路径直接可用，无需改动。

## 代码改动

### run_didi_eval.py
- `SAMROAD_ROOT` → `PROJECT_ROOT`（DelvMap 根）
- `SAMROAD_PY` → `sys.executable`
- `DIDI_ROOT` / `SPLIT_FILE` 指向本地 `datasets/didi/xian/`
- `run_samroad_eval` → `run_eval`：调本地 `metrics/eval.py`，**删除全部 os.symlink 逻辑**
- `--dir` 传 work_dir 相对 PROJECT_ROOT 的路径（eval.py 内部用 `../{dir}/graph/`，cwd=metrics/）

### didi_bridge.py / viz_region.py
- `DIDI_ROOT` 默认值从 samroad 绝对路径改为本地相对路径（基于 `__file__` 推导）

## 验证

```bash
# 本地评估 (不碰 samroad)
python adaptive_map_completion/run_didi_eval.py --split test --work_dir experiments/didi_amc_test --pred_mode amc
```
- 调用 `/Users/highee/research/DelvMap/metrics/eval.py`（本地）
- samroad 根软链接数：0（评估前后均不新建）
- 指标与解耦前完全一致（AMC APLS 0.617、TOPO 0.699）

## 依赖
- `go` 命令（APLS 现场编译，`which go` 可用即可）
- Python: yaml, networkx, rtree, opencv, skimage, tqdm（samroad env 已含）

## gitignore
`metrics/apls/eval_bin`（编译产物）、`datasets/didi/xian/`（数据）、`experiments/`（实验产物）均不入 git，见 [.gitignore](../.gitignore)。
