"""Compatibility shim for OmniDocBench v1_5 and modern scikit-image.

OmniDocBench v1_5 calls skimage.measure.ransac(random_state=...), while
scikit-image versions used by the evaluator expose the same RNG parameter as
rng. Mapping the keyword preserves the seed and the RANSAC algorithm; it does
not alter the scoring implementation.
"""
from __future__ import annotations

from functools import wraps
import sys


def _wrap(current):
    if getattr(current, "_omnidocbench_random_state_compat", False):
        return current

    @wraps(current)
    def ransac_compat(*args, random_state=None, **kwargs):
        if random_state is not None:
            if "rng" in kwargs:
                raise TypeError("ransac received both random_state and rng")
            kwargs["rng"] = random_state
        return current(*args, **kwargs)

    ransac_compat._omnidocbench_random_state_compat = True
    return ransac_compat


def install() -> None:
    # Use only the public API. The private implementation module changed
    # names across scikit-image releases (_fit, fit, or package-local), while
    # skimage.measure.ransac remains the supported entry point.
    import skimage.measure as measure

    wrapped = _wrap(measure.ransac)
    measure.ransac = wrapped

    # visual_matcher.py imports ransac by value, so patch that binding too when
    # the official evaluator has already imported the module.
    visual_matcher = sys.modules.get("metrics.cdm.modules.visual_matcher")
    if visual_matcher is not None and hasattr(visual_matcher, "ransac"):
        visual_matcher.ransac = wrapped


install()
