"""
Unified metric evaluation script for SAM-Road.
Replaces the old apls_cityscale.py, apls_spacenet.py, topo_cityscale.py, topo_spacenet.py.

Usage:
    python eval.py --dataset cityscale --dir save/xxx
    python eval.py --dataset spacenet --dir save/xxx --workers 8
    python eval.py --dataset spacenet --dir save/xxx --metric apls
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import platform
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add this script's directory to path so we can import apls/topo modules.
# eval.py 内部大量历史路径按 cwd=metrics/ 书写 (../datasets, ../save, ../runs)。
# run.py 从项目根调用 `python metrics/eval.py`, 手动评估则通常 `cd metrics` 后调用。
# 为保证两种入口语义一致, 这里强制切到 metrics/ 目录。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)


def canonical_dataset(dataset):
    """Normalize legacy aliases to explicit dataset names."""
    # 历史上 metrics 用 didi 表示 Xian；现在项目 config 统一用 didi_xian，
    # 预留 didi_chengdu 等未来数据集，因此内部 canonical 一律使用 didi_xian。
    return 'didi_xian' if dataset == 'didi' else dataset


def load_dataset_config(dataset):
    """Load dataset-specific configuration."""
    dataset = canonical_dataset(dataset)
    config_path = os.path.join(SCRIPT_DIR, 'configs', f'{dataset}.yaml')
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_test_indices(dataset):
    """Get test image indices for the given dataset."""
    dataset = canonical_dataset(dataset)
    if dataset == 'cityscale':
        return [8, 9, 19, 28, 29, 39, 48, 49, 59, 68, 69, 79, 88, 89, 99,
                108, 109, 119, 128, 129, 139, 148, 149, 159, 168, 169, 179]
    elif dataset == 'spacenet':
        with open('../datasets/spacenet/data_split.json', 'r') as f:
            return json.load(f).get('test', [])
    elif dataset == 'didi_xian':
        with open('../datasets/didi/xian/data_split.json', 'r') as f:
            return json.load(f).get('test', [])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def eval_apls(dataset, target_dir, workers=None):
    """Run APLS evaluation."""
    dataset = canonical_dataset(dataset)
    print(f"\n{'='*60}")
    print(f"  APLS Evaluation | Dataset: {dataset} | Dir: {target_dir}")
    print(f"{'='*60}")

    test_names = get_test_indices(dataset)
    out_dir = f"../{target_dir}/results/apls"
    os.makedirs(out_dir, exist_ok=True)

    # Go environment
    go_exe = shutil.which("go") or "go"
    go_bin_name = "eval_bin.exe" if platform.system() == "Windows" else "eval_bin"
    go_bin_path = os.path.join(SCRIPT_DIR, "apls", go_bin_name)
    apls_dir = os.path.join(SCRIPT_DIR, "apls")

    print("Pre-compiling Go evaluation script...")
    build_result = subprocess.run(
        [go_exe, "build", "-o", go_bin_name, "main.go"],
        cwd=apls_dir, capture_output=True, text=True
    )
    if build_result.returncode != 0:
        print(f"Warning: Go compilation failed!\n{build_result.stderr}")
        return False

    abs_go_bin = os.path.abspath(go_bin_path)

    # Path patterns based on dataset
    if dataset == 'cityscale':
        gt_pattern = '../datasets/cityscale/20cities/region_{}_graph_gt.pickle'
    elif dataset == 'spacenet':
        gt_pattern = '../datasets/spacenet/RGB_1.0_meter/{}__gt_graph.p'
    elif dataset == 'didi_xian':
        gt_pattern = '../datasets/didi/xian/2019_400/region_{}_graph_gt.pickle'

    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 2)

    def process_single_image(name):
        pred_path = f"../{target_dir}/graph/{name}.p"
        gt_path = gt_pattern.format(name)
        # 关键: 用绝对路径, 因为 Go binary 的 cwd 是 apls_dir (即 metrics/apls/),
        # 而 pred/gt 相对路径基准是 metrics/. 用绝对路径避免拼接基准目录错位 →
        # 之前 Go binary 用 os.Args[3] 写文件, 写到了 metrics/save/.../ 而不是
        # save/.../, 导致 out_txt 永远不存在 → APLS 报全 SKIP → 沉默 0 分。
        out_txt = os.path.abspath(f"../{target_dir}/results/apls/{name}.txt")

        if not os.path.exists(pred_path):
            return f"SKIP: Missing {pred_path}"

        temp_gt = f"gt_{name}.json"
        temp_prop = f"prop_{name}.json"

        try:
            convert_py = os.path.join(SCRIPT_DIR, "apls", "convert.py")
            res_gt = subprocess.run([sys.executable, convert_py, gt_path, temp_gt],
                                    capture_output=True, text=True)
            if res_gt.returncode != 0:
                return f"Convert GT Error ({name}): {res_gt.stderr}"

            res_prop = subprocess.run([sys.executable, convert_py, pred_path, temp_prop],
                                      capture_output=True, text=True)
            if res_prop.returncode != 0:
                return f"Convert Prop Error ({name}): {res_prop.stderr}"

            # temp_gt/temp_prop 也要传绝对路径 (Go binary cwd=apls_dir)
            res_go = subprocess.run(
                [abs_go_bin, os.path.abspath(temp_gt), os.path.abspath(temp_prop),
                 out_txt, dataset],
                cwd=apls_dir, capture_output=True, text=True
            )
            if res_go.returncode != 0:
                return f"Go Execution Error ({name}): {res_go.stderr}"
            # Go 在 returncode=0 时也可能没写文件 (路径错或权限错), 显式校验
            if not os.path.exists(out_txt):
                return f"Go Wrote No Output ({name}): expected {out_txt}, stderr={res_go.stderr[:200]}"

            return "SUCCESS"
        except Exception as e:
            return f"Exception ({name}): {str(e)}"
        finally:
            for tmp in [temp_gt, temp_prop]:
                if os.path.exists(tmp):
                    os.remove(tmp)

    print(f"Starting APLS evaluation with {workers} workers...")
    errors = []

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_image, name): name for name in test_names}

        if has_tqdm:
            iterable_futures = tqdm(as_completed(futures), total=len(test_names), desc="APLS Progress")
        else:
            iterable_futures = as_completed(futures)

        for future in iterable_futures:
            res = future.result()
            if res != "SUCCESS":
                errors.append(res)

    # Clean up compiled Go binary
    if os.path.exists(go_bin_path):
        os.remove(go_bin_path)

    # 报告 SKIP / Error 数, 避免"全 SKIP 静默出 0 分"的坑
    skipped = [e for e in errors if e.startswith("SKIP:")]
    failed = [e for e in errors if not e.startswith("SKIP:")]
    if skipped:
        print(f"⚠️  APLS: {len(skipped)} / {len(test_names)} samples SKIPPED (missing pred .p)")
        if len(skipped) <= 3:
            for s in skipped:
                print(f"     {s}")
        else:
            for s in skipped[:3]:
                print(f"     {s}")
            print(f"     ... and {len(skipped) - 3} more")
    if failed:
        print(f"⚠️  APLS: {len(failed)} samples FAILED:")
        for f in failed[:5]:
            print(f"     {f}")

    # Compute final metric
    apls_vals = []
    output_apls = []

    if os.path.exists(out_dir):
        for file_name in sorted(os.listdir(out_dir)):
            with open(os.path.join(out_dir, file_name)) as f:
                lines = f.readlines()
            if not lines or 'NaN' in lines[0]:
                continue
            val = float(lines[0].split(' ')[-1])
            apls_vals.append(val)
            output_apls.append([file_name, val])

    mean_apls = np.mean(apls_vals) if apls_vals else 0
    if not apls_vals:
        print(f"⚠️  APLS: ALL samples missing or failed → mean = 0.0 is meaningless. "
              f"Check that --dir points at a valid inference output (relative to project root).")
    print(f"APLS: {mean_apls:.8f}  (computed on {len(apls_vals)} / {len(test_names)} samples)")

    with open(f"../{target_dir}/results/apls.json", "w") as jf:
        json.dump({'apls': output_apls, 'final_APLS': mean_apls,
                   'n_evaluated': len(apls_vals), 'n_total': len(test_names),
                   'n_skipped': len(skipped), 'n_failed': len(failed)}, jf)

    return True


def eval_topo(dataset, target_dir, workers=None):
    """Run TOPO evaluation."""
    dataset = canonical_dataset(dataset)
    print(f"\n{'='*60}")
    print(f"  TOPO Evaluation | Dataset: {dataset} | Dir: {target_dir}")
    print(f"{'='*60}")

    out_dir = f"../{target_dir}/results/topo"
    os.makedirs(out_dir, exist_ok=True)

    # Clean old results
    for f in os.listdir(out_dir):
        if f.endswith('.txt'):
            os.remove(os.path.join(out_dir, f))

    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 2)

    script_dir = SCRIPT_DIR

    if workers > 1:
        # Use parallel evaluation
        main_py_path = os.path.join(script_dir, "topo", "eval_parallel.py")
        res = subprocess.run(
            [sys.executable, main_py_path, "-savedir", target_dir,
             "-dataset", dataset, "-workers", str(workers)]
        )
        if res.returncode != 0:
            print("Error: TOPO parallel evaluation failed.")
            return False
    else:
        # Use single-process evaluation
        main_py_path = os.path.join(script_dir, "topo", "main.py")
        res = subprocess.run(
            [sys.executable, main_py_path, "-savedir", target_dir, "-dataset", dataset],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    # Compute final metric
    precision = []
    recall = []

    if os.path.exists(out_dir):
        for file_name in os.listdir(out_dir):
            if not file_name.endswith('.txt'):
                continue
            with open(os.path.join(out_dir, file_name)) as f:
                lines = f.readlines()
            if not lines:
                continue

            p = float(lines[-1].split(' ')[0].split('=')[-1])
            r = float(lines[-1].split(' ')[-1].split('=')[-1])
            if p + r > 0:
                precision.append(p)
                recall.append(r)

    mean_p = np.mean(precision) if precision else 0
    mean_r = np.mean(recall) if recall else 0
    topo_f1 = 2 * mean_p * mean_r / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0

    print(f"TOPO: {topo_f1:.8f} | Precision: {mean_p:.8f} | Recall: {mean_r:.8f}")

    out_path = f"../{target_dir}/results/topo.json"
    with open(out_path, "w") as jf:
        json.dump({'mean topo': [topo_f1, mean_p, mean_r], 'prec': precision,
                   'recall': recall, 'f1': topo_f1}, jf)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified metric evaluation for SAM-Road")
    parser.add_argument('--dataset', choices=['cityscale', 'spacenet', 'didi_xian', 'didi'], required=True,
                        help="Dataset type ('didi' is a legacy alias of 'didi_xian')")
    parser.add_argument('--dir', required=True,
                        help='Output directory (relative to project root, e.g. save/xxx)')
    parser.add_argument('--metric', choices=['apls', 'topo', 'all'], default='all',
                        help='Which metric to evaluate (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')

    args = parser.parse_args()
    if args.dataset == 'didi':
        print("[WARN] dataset='didi' is deprecated; use 'didi_xian' instead.")
        args.dataset = 'didi_xian'

    # ---- Normalize --dir ----
    # eval 内部统一加 ../ 前缀, 所以 --dir 必须是"相对项目根"的路径 (例如 save/xxx),
    # 用户常见误传:
    #   --dir ../save/xxx       (从 metrics/ 习惯加了 ../)
    #   --dir save/xxx/         (尾部多余斜线)
    #   --dir /abs/path/save/xxx (绝对路径)
    # 自动剥掉这些, 失败则报错并退出, 避免静默 SKIP 全部样本而出 0 分。
    raw = args.dir.rstrip('/')
    if os.path.isabs(raw):
        # 绝对路径 → 转成相对项目根
        project_root = os.path.dirname(SCRIPT_DIR)
        try:
            raw = os.path.relpath(raw, project_root)
        except ValueError:
            print(f"Error: --dir must be inside project root, got: {args.dir}")
            sys.exit(1)
    while raw.startswith('../'):
        raw = raw[3:]
    # 验证 graph/ 子目录存在 (在 metrics/ cwd 下, eval 内部用 ../ 前缀)
    expected = os.path.join('..', raw, 'graph')
    if not os.path.isdir(expected):
        print(f"Error: cannot find pred graphs at <project_root>/{raw}/graph/")
        print(f"       --dir must be relative to project root (e.g. save/infer_xxx),")
        print(f"       not relative to metrics/. Got: {args.dir}")
        sys.exit(1)
    args.dir = raw
    print(f"Normalized --dir: {args.dir}")

    if args.metric in ('apls', 'all'):
        eval_apls(args.dataset, args.dir, args.workers)

    if args.metric in ('topo', 'all'):
        eval_topo(args.dataset, args.dir, args.workers)
