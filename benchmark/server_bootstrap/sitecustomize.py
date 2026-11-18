"""Only placed on server PYTHONPATH by the benchmark launcher."""
import os
import sys

# Preserve any environment-provided sitecustomize instead of silently replacing it.
import importlib.machinery
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_paths = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _paths)
if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)

if os.environ.get("VLLM_BENCHMARK_PROCESSOR_DIR"):
    try:
        from benchmark.server_processor import install
        install()
    except Exception as exc:
        print("BENCHMARK PROCESSOR BOOTSTRAP FAILED: " + repr(exc), file=sys.stderr)
