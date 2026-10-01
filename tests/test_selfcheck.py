import pytest

from tokimeki.stages.selfcheck import run_checks


@pytest.mark.gpu
def test_gpu_check_passes() -> None:
    failed = [check for check in run_checks() if not check.ok]
    assert failed == []
