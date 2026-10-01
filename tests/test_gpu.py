import threading

import pytest

from tokimeki.models.gpu import loaded, loaded_model, prefetched


class Model:
    closed = False

    def close(self) -> None:
        self.closed = True


def test_one_model_at_a_time() -> None:
    with loaded("first", Model) as first:
        assert loaded_model() == "first"
        with (
            pytest.raises(RuntimeError, match="first is still on the GPU"),
            loaded("second", Model),
        ):
            pass
    assert first.closed
    assert loaded_model() is None


def test_model_is_freed_when_the_batch_fails() -> None:
    with pytest.raises(ValueError), loaded("first", Model):
        raise ValueError
    assert loaded_model() is None


def test_prefetched_prepares_the_next_item_while_the_current_one_is_used() -> None:
    started = [threading.Event() for _ in range(3)]

    def prepare(item: int) -> int:
        started[item].set()
        return item * 10

    seen: list[tuple[int, int]] = []
    for item, value in prefetched(range(3), prepare):
        if item + 1 < 3:
            assert started[item + 1].wait(timeout=5)
        seen.append((item, value))
    assert seen == [(0, 0), (1, 10), (2, 20)]
