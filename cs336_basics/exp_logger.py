from __future__ import annotations

import json
import os
import time
from typing import Any


class ExperimentLogger:
    def __init__(self, log_path: str):
        self.log_path = str(log_path)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self.start_time = time.time()
        self._fh = open(self.log_path, "a", encoding="utf-8")

    def log(self, step: int, **metrics: Any) -> None:
        entry = {"step": int(step), "time_sec": time.time() - self.start_time}
        for k, v in metrics.items():
            entry[k] = float(v) if isinstance(v, (int, float)) else v
        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
