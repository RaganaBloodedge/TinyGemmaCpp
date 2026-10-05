#!/usr/bin/env python3
"""权重与分词器的路径发现。

本仓库不带权重：`.sbs` 与 `tokenizer.spm` 由 Google 随 Gemma 模型单独发布，
体积在 GB 量级。脚本按同一个顺序找这两个文件，换机器时不必改代码——

    1. 环境变量（`TG_WEIGHTS` / `TG_TOKENIZER`）
    2. `<仓库>/weights/<文件名>`
    3. `<仓库>` 同级的 `gemma.cpp-main/weights/<文件名>`

第 3 条是因为常见布局是把本仓库与 gemma.cpp 放在同一层目录下。
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_asset(env_var: str, filename: str) -> str:
    """返回按上述顺序找到的第一个路径。

    都不存在时返回仓库内的候选路径，让调用方在真正打开时报错——
    错误信息里带完整路径比在这里抛异常更好定位。
    """
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env
    candidates = (
        os.path.join(HERE, "weights", filename),
        os.path.join(os.path.dirname(HERE), "gemma.cpp-main", "weights", filename),
    )
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]
