"""受控基线（回应 AE-3 / R2-4）。"""
from .tracklet_baseline import TrackletBaseline, build  # noqa: F401

__all__ = ["TrackletBaseline", "build"]
