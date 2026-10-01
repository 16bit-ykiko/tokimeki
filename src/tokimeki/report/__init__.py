"""The static HTML report, written into the series' data directory."""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from tokimeki.paths import SeriesPaths
from tokimeki.report.data import gather
from tokimeki.report.images import write_face, write_thumbnail
from tokimeki.report.render import render


def build_report(conn: sqlite3.Connection, paths: SeriesPaths) -> Path:
    """Rebuild `report/` from scratch, so no image outlives a shot the filter has since dropped."""
    out = paths.report_dir
    shutil.rmtree(out, ignore_errors=True)
    img = out / "img"
    img.mkdir(parents=True)
    data = gather(conn)
    images: set[str] = set()
    for ep in data.episodes:
        for kept in ep.kept:
            name = f"shot-{kept.shot.id}.jpg"
            if write_thumbnail(paths, ep.episode.id, kept.thumbnail.frame_index, img / name):
                images.add(name)
    for cluster in data.clusters:
        for face in cluster.samples:
            name = f"face-{face.face.id}.jpg"
            if write_face(paths, face, img / name):
                images.add(name)
    index = out / "index.html"
    index.write_text(render(data, paths.root, images, datetime.now()), encoding="utf-8")
    return index
