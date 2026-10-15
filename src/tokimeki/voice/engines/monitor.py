"""Peak GPU memory while an external engine runs (it is another process, so nvidia-smi)."""

import subprocess
import threading
from types import TracebackType

from tokimeki.models.gpu import nvidia_smi


def memory_used() -> int:
    """MiB in use on the first GPU, by every process (the desktop too)."""
    result = subprocess.run(
        [str(nvidia_smi()), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.strip().splitlines()
    return int(lines[0]) if result.returncode == 0 and lines else 0


class MemoryPeak:
    """Samples GPU memory every `interval` seconds inside a `with` block."""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.before = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, memory_used())

    def __enter__(self) -> "MemoryPeak":
        self.before = self.peak = memory_used()
        self._thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def added(self) -> int:
        """MiB the block added at its peak over what was in use before it."""
        return max(0, self.peak - self.before)
