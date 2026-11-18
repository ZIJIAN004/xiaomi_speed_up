"""Compatibility shim for OmniDocBench v1_5 and modern scikit-image.

OmniDocBench v1_5 calls skimage.measure.ransac(random_state=...), while
scikit-image versions used by the evaluator expose the same RNG parameter as
rng. Mapping the keyword preserves the seed and the RANSAC algorithm; it does
not alter the scoring implementation.
"""
from __future__ import annotations

from functools import wraps


def install() -> None:
    import skimage.measure as measure

    current = measure.ransac
    if getattr(current, "_omnidocbench_random_state_compat", False):
        return

    @wraps(current)
    def ransac_compat(*args, random_state=None, **kwargs):
        if random_state is not None:
            if "rng" in kwargs:
                raise TypeError("ransac received both random_state and rng")
            kwargs["rng"] = random_state
        return current(*args, **kwargs)

    ransac_compat._omnidocbench_random_state_compat = True
    measure.ransac = ransac_compat


install()
