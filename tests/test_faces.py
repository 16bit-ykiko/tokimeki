import numpy as np
import pytest

from tokimeki.models.faces import decode_yolo, nms


def test_nms_keeps_the_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.7) == [1, 2]


def test_decode_yolo_normalises_and_thresholds() -> None:
    output = np.zeros((5, 3), dtype=np.float32)
    output[:, 0] = [320, 160, 64, 32, 0.9]
    output[:, 1] = [100, 100, 10, 10, 0.1]
    output[:, 2] = [322, 161, 64, 32, 0.8]
    (face,) = decode_yolo(output, 640, 640, threshold=0.3)
    assert (face.x0, face.y0, face.x1, face.y1) == pytest.approx((0.45, 0.225, 0.55, 0.275))
    assert face.score == pytest.approx(0.9)
