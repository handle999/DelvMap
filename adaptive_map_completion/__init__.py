"""Adaptive Map Completion (Stage 2) package.

把本目录加到 sys.path，使子模块 ``adaptive_map_completion.py`` 里的
``from tptk...`` / ``from walkway_completion...`` 绝对导入能解析到本包自带的
``adaptive_map_completion/tptk/`` 与 ``adaptive_map_completion/walkway_completion/``。
这样无论从仓库根还是任意 cwd ``import adaptive_map_completion`` 都一致，
不再依赖调用方手动 ``sys.path.insert``。
"""
import os as _os
import sys as _sys

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)

from .adaptive_map_completion import (  # noqa: E402,F401
    DelvMapConnector,
    SplitOP,
    compress_rn,
    densify_rn,
    obtain_segmented_trajs,
    index_trajs,
    perpendicular_intersection,
)

__all__ = [
    'DelvMapConnector',
    'SplitOP',
    'compress_rn',
    'densify_rn',
    'obtain_segmented_trajs',
    'index_trajs',
    'perpendicular_intersection',
]
