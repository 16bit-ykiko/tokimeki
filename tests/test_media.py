from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from clips import make_clip
from PIL import Image

from tokimeki.media.decode import decode_for_transnet, extract_frames, gpu_resize
from tokimeki.media.probe import NvdecUnsupportedError, VideoInfo, probe, require_nvdec


def test_probe(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "clip.mkv", ["color=c=red", "color=c=blue"])
    info = probe(clip)
    assert (info.width, info.height, info.fps, info.codec) == (320, 180, Fraction(24), "h264")
    assert info.estimated_frames == 48


def test_hi10p_h264_is_refused() -> None:
    info = VideoInfo(1920, 1080, Fraction(24), 10.0, "h264", "yuv420p10le")
    with pytest.raises(NvdecUnsupportedError, match="Hi10P"):
        require_nvdec(info, Path("ep.mkv"))
    require_nvdec(VideoInfo(1920, 1080, Fraction(24), 10.0, "hevc", "yuv420p10le"), Path("ep.mkv"))


def test_gpu_resize_halves_before_the_last_step() -> None:
    steps = gpu_resize(1920, 1080, 240, 136)
    assert [s.split(":")[:2] for s in steps] == [
        ["scale_cuda=w=960", "h=540"],
        ["scale_cuda=w=480", "h=270"],
        ["scale_cuda=w=240", "h=136"],
    ]
    assert "lanczos" in steps[-1]
    assert gpu_resize(1920, 1080, 1280, 720) == [
        "scale_cuda=w=1280:h=720:interp_algo=lanczos:format=nv12"
    ]


@pytest.mark.gpu
def test_nvdec_decodes_every_frame(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "clip.mkv", ["color=c=red", "color=c=blue"])
    frames = decode_for_transnet(clip, 320, 180)
    assert frames.shape == (48, 27, 48, 3)
    assert frames[10, 13, 24].tolist() == pytest.approx([255, 0, 0], abs=12)
    assert frames[40, 13, 24].tolist() == pytest.approx([0, 0, 255], abs=12)


@pytest.mark.gpu
def test_extract_frames_by_index(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "clip.mkv", ["color=c=red", "color=c=blue"])
    outs = [tmp_path / "out" / f"{i}.jpg" for i in (30, 5)]
    extract_frames(clip, 320, 180, [30, 5], outs)
    blue, red = (np.asarray(Image.open(p).convert("RGB"))[90, 160].tolist() for p in outs)
    assert red == pytest.approx([255, 0, 0], abs=12)
    assert blue == pytest.approx([0, 0, 255], abs=12)
