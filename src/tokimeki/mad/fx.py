"""Effects on a slot's clip: a white flash just after the cut, a punch-in on an accent beat and
a slow drift in across a still shot.

A zoom crops a shrinking rectangle anchored on `focus` (her face), so the face keeps its place
on screen while the frame grows around it; a punch inside a drift zooms within the drift's
crop. The flash is a white layer fading out over the first frames after the cut.

On the GPU: decoding, scaling, the flash (`overlay_cuda`) and encoding. On the CPU: the zooms,
through `perspective`, the one filter in this ffmpeg that crops each frame by its own sub-pixel
rectangle (`scale_cuda` fixes its size when it starts, and libplacebo has no Vulkan driver for
the GPU under WSL), and the few white frames the flash lays over the clip.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from tokimeki.mad.candidates import Candidate
from tokimeki.mad.motion import MIN_SYNC_ONSET, ShotMotion
from tokimeki.mad.plan import (
    DRIFT_ZOOM,
    FLASH_FRAMES,
    FLASH_STRENGTH,
    PUNCH_ZOOM,
    Fx,
    Plan,
    SlotPlan,
)
from tokimeki.mad.song.analysis import SongAnalysis

STRONG_ACCENT = 0.7
"""Beats that hit at least this hard (0-1) are the accents a punch may land on."""
PUNCH_ATTACK = 2
"""Frames the punch takes to reach its zoom, the last one on the accent."""
PUNCH_RELEASE = 0.4
"""Seconds it takes to ease back."""
PUNCH_MARGIN = 0.1
"""Seconds an accent must be inside its slot for refine to put a punch on it."""
DEFAULT_FOCUS = (0.5, 0.45)

AUTO_PUNCH = 0.8
"""`plan auto` punches in on accents at least this strong..."""
AUTO_PUNCH_MARGIN = 0.5
"""...at least this many seconds from either cut."""
AUTO_DRIFT_SLOT = 1.5
"""`plan auto` drifts across still shots in slots at least this long."""
BAR_TOLERANCE = 1 / 48


def strong_accents(
    analysis: SongAnalysis, start: float, end: float, minimum: float = STRONG_ACCENT
) -> list[tuple[float, float]]:
    """`(beat, strength)` of the beats from `start` to `end` that hit at least `minimum`."""
    if analysis.accents is None:
        return []
    return [
        (b, s)
        for b, s in zip(analysis.beats, analysis.accents, strict=True)
        if start - 1e-6 <= b <= end + 1e-6 and s >= minimum
    ]


def _source_time(slot: SlotPlan, t: float) -> float | None:
    """The episode second shown at song second `t` of the slot."""
    speed = slot.derived_speed
    if slot.source_in is None or speed is None:
        return None
    return slot.source_in + (t - slot.start) * speed


def _focus(
    candidate: Candidate | None, start: float | None, end: float | None
) -> tuple[float, float]:
    if candidate is None or start is None or end is None:
        return DEFAULT_FOCUS
    x, y = candidate.focus(start, end)
    return (round(x, 3), round(y, 3))


def resolve(fx: Fx, slot: SlotPlan, candidate: Candidate | None, analysis: SongAnalysis) -> Fx:
    """The effect with what it leaves out filled in: a punch on the slot's `accent` or its
    strongest beat, zooms toward her face in the window, and the default sizes."""
    if fx.kind == "flash":
        return replace(
            fx,
            frames=FLASH_FRAMES if fx.frames is None else fx.frames,
            strength=FLASH_STRENGTH if fx.strength is None else fx.strength,
        )
    if fx.kind == "punch":
        at = fx.at
        if at is None:
            at = slot.accent
        if at is None:
            inside = strong_accents(
                analysis, slot.start + PUNCH_MARGIN, slot.end - PUNCH_MARGIN, minimum=0.0
            )
            at = max(inside, key=lambda a: a[1])[0] if inside else None
        focus = fx.focus
        if focus is None and at is not None:
            moment = _source_time(slot, at)
            focus = _focus(candidate, moment, moment)
        return replace(fx, at=at, zoom=PUNCH_ZOOM if fx.zoom is None else fx.zoom, focus=focus)
    if fx.kind == "drift":
        focus = fx.focus or _focus(candidate, slot.source_in, slot.source_out)
        return replace(fx, zoom=DRIFT_ZOOM if fx.zoom is None else fx.zoom, focus=focus)
    return fx


@dataclass(frozen=True, slots=True)
class ClipFx:
    """A slot's effects as its clip renders them, frames counted from the cut."""

    flash: tuple[int, float] | None = None
    """Frames, and whiteness at the cut."""
    drift: tuple[float, tuple[float, float]] | None = None
    """Zoom at the end of the slot, and focus."""
    punch: tuple[float, tuple[float, float], int] | None = None
    """Zoom, focus, and the frame of the accent."""

    def key(self) -> object:
        def point(p: tuple[float, float]) -> list[float]:
            return [round(p[0], 3), round(p[1], 3)]

        return {
            "flash": None if self.flash is None else [self.flash[0], round(self.flash[1], 3)],
            "drift": None
            if self.drift is None
            else [round(self.drift[0], 4), point(self.drift[1])],
            "punch": None
            if self.punch is None
            else [round(self.punch[0], 4), point(self.punch[1]), self.punch[2]],
        }


def clip_fx(plan: Plan, slot: SlotPlan) -> ClipFx | None:
    """The slot's effects for its clip, or None when it has none to render."""
    fps = plan.frame_rate
    first = round((slot.start - plan.start) * fps)
    flash: tuple[int, float] | None = None
    drift: tuple[float, tuple[float, float]] | None = None
    punch: tuple[float, tuple[float, float], int] | None = None
    for f in slot.fx:
        if f.kind == "flash":
            strength = FLASH_STRENGTH if f.strength is None else f.strength
            flash = (f.frames or FLASH_FRAMES, strength)
        elif f.kind == "drift":
            drift = (f.zoom or DRIFT_ZOOM, f.focus or DEFAULT_FOCUS)
        elif f.kind == "punch" and f.at is not None:
            frame = round((f.at - plan.start) * fps) - first
            punch = (f.zoom or PUNCH_ZOOM, f.focus or DEFAULT_FOCUS, frame)
    if flash is None and drift is None and punch is None:
        return None
    return ClipFx(flash, drift, punch)


def _crop(fx: ClipFx, frames: int, fps: Fraction) -> tuple[str, str, str, str]:
    """Expressions for the left, top, width and height of the part of each frame (`W` by `H`)
    that fills the picture; `perspective` counts frames (`in`) from 1."""
    n = "(in-1)"
    left, top, width, height = "0", "0", "W", "H"
    if fx.drift is not None:
        zoom, (x, y) = fx.drift
        z = f"pow({zoom:.6g},{n}/{max(1, frames - 1)})"
        left, top = f"{x:.4f}*W*(1-1/{z})", f"{y:.4f}*H*(1-1/{z})"
        width, height = f"W/{z}", f"H/{z}"
    if fx.punch is not None:
        zoom, (x, y), at = fx.punch
        rise = at - PUNCH_ATTACK
        release = max(1, round(PUNCH_RELEASE * fps))
        envelope = (
            f"if(lte({n},{rise}),0,if(lte({n},{at}),1-pow(1-({n}-({rise}))/{PUNCH_ATTACK},2),"
            f"pow(max(0,1-({n}-({at}))/{release}),2)))"
        )
        z = f"(1+{zoom - 1:.6g}*{envelope})"
        share_x = f"clip(({x:.4f}*W-({left}))/({width}),0,1)"
        share_y = f"clip(({y:.4f}*H-({top}))/({height}),0,1)"
        left = f"{left}+{share_x}*({width})*(1-1/{z})"
        top = f"{top}+{share_y}*({height})*(1-1/{z})"
        width, height = f"({width})/{z}", f"({height})/{z}"
    return left, top, width, height


def zoom_filter(fx: ClipFx, frames: int, fps: Fraction) -> str | None:
    """A `perspective` filter (CPU) giving the clip its drift and punch, or None."""
    if fx.drift is None and fx.punch is None:
        return None
    left, top, width, height = _crop(fx, frames, fps)
    right, bottom = f"{left}+{width}", f"{top}+{height}"
    corners = {
        "x0": left, "y0": top, "x1": right, "y1": top,
        "x2": left, "y2": bottom, "x3": right, "y3": bottom,
    }  # fmt: skip
    options = ":".join(f"{k}='{v}'" for k, v in corners.items())
    return f"perspective={options}:interpolation=cubic:eval=frame"


def flash_source(fx: ClipFx, width: int, height: int, fps: str) -> str | None:
    """The white frames the flash lays over the clip, uploaded for `overlay_cuda`, or None."""
    if fx.flash is None:
        return None
    frames, strength = fx.flash
    alpha = f"255*{strength:.4f}*(1-N/{frames})"
    return (
        f"color=c=white:s={width}x{height}:r={fps},format=yuva420p,"
        f"geq=lum=235:cb=128:cr=128:a='{alpha}',trim=end_frame={frames},hwupload_cuda"
    )


def _on_bar(analysis: SongAnalysis, t: float) -> bool:
    return any(abs(b - t) <= BAR_TOLERANCE for b in analysis.bars)


def _strength(analysis: SongAnalysis, t: float) -> float:
    if analysis.accents is None or not analysis.beats:
        return 0.0
    i = min(range(len(analysis.beats)), key=lambda j: abs(analysis.beats[j] - t))
    return analysis.accents[i] if abs(analysis.beats[i] - t) <= BAR_TOLERANCE else 0.0


def _calm(slot: SlotPlan, motion: ShotMotion | None) -> bool:
    """A still shot where nothing starts moving in the slot's window."""
    if motion is None or not motion.still or slot.source_in is None or slot.source_out is None:
        return False
    first, last = slot.source_in, slot.source_out
    return not any(first <= o.time <= last and o.strength >= MIN_SYNC_ONSET for o in motion.onsets)


def suggest(
    plan: Plan, analysis: SongAnalysis, motion: Mapping[int, ShotMotion]
) -> tuple[Plan, list[str]]:
    """The draft's effects, for a plan with its windows placed: a flash on a chorus's first
    cut and on its downbeats that hit hard, a punch on the strongest accent inside a slot, and
    a drift across a still shot where nothing moves, never two drifts in a row; and what was
    added. `refine` then fills in their details."""
    notes: list[str] = []
    slots: list[SlotPlan] = []
    previous = ""
    drifting = False
    for i, slot in enumerate(plan.slots):
        effects = list(slot.fx)
        kinds = {f.kind for f in effects}
        opens = slot.section == "chorus" and previous != "chorus"
        previous = slot.section
        downbeat = slot.section == "chorus" and _on_bar(analysis, slot.start)
        if "flash" not in kinds and (
            opens or (downbeat and _strength(analysis, slot.start) >= STRONG_ACCENT)
        ):
            effects.append(Fx("flash"))
            notes.append(f"slot {i}: flash on the chorus {'entry' if opens else 'downbeat'}")
        inside = strong_accents(
            analysis,
            slot.start + AUTO_PUNCH_MARGIN,
            slot.end - AUTO_PUNCH_MARGIN,
            minimum=AUTO_PUNCH,
        )
        if "punch" not in kinds and inside:
            at, strength = max(inside, key=lambda a: a[1])
            effects.append(Fx("punch", at=at))
            notes.append(f"slot {i}: punch on the accent at {at:.2f}s ({strength:.2f})")
        shot = motion.get(slot.shot) if slot.shot is not None else None
        calm = slot.duration >= AUTO_DRIFT_SLOT and _calm(slot, shot)
        if "drift" not in kinds and calm and not drifting:
            effects.append(Fx("drift"))
            notes.append(f"slot {i}: drift, nothing moves in shot {slot.shot}")
        drifting = any(f.kind == "drift" for f in effects)
        slots.append(replace(slot, fx=effects))
    return replace(plan, slots=slots), notes


def describe(effects: Sequence[Fx]) -> str:
    """The effects in a few words, for the report."""
    words: list[str] = []
    for f in effects:
        if f.kind == "punch" and f.at is not None:
            words.append(f"punch {f.zoom or PUNCH_ZOOM:.2f}x at {f.at:.2f}s")
        elif f.kind == "drift":
            words.append(f"drift to {f.zoom or DRIFT_ZOOM:.2f}x")
        else:
            words.append(f.kind)
    return ", ".join(words)
