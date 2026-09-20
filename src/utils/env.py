"""从 .env 加载环境变量（不覆盖已有环境变量）。"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """
    读取仓库根目录（或指定路径）的 .env，写入 os.environ。

    - 已存在的环境变量不会被覆盖（便于服务器先 export）
    - 文件不存在则静默跳过
    """
    if path is None:
        # runtime 在 src/utils/，仓库根为 parents[2]
        path = Path(__file__).resolve().parents[2] / ".env"
    else:
        path = Path(path)

    if not path.is_file():
        return None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
    return path
