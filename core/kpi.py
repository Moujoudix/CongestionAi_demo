from __future__ import annotations
import time, os, json
from pathlib import Path
from typing import Dict, Any
class KPILogger:
    def __init__(self, path: str | os.PathLike = 'data_sample/metrics.csv'):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text('timestamp,run_id,metric,value,meta\n')
    def log(self, run_id: str, metric: str, value: float, meta: Dict[str, Any] | None = None):
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        line = f"{int(time.time())},{run_id},{metric},{value},{meta_json}\n"
        with self.path.open('a', encoding='utf-8') as f:
            f.write(line)
