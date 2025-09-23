from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TbLogger:
    def __init__(self, log_dir: Path | str, flush_secs: int = 5):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._writer.add_scalar(name, value, step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
