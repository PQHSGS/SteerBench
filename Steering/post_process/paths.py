import os
from pathlib import Path

_DATA_ROOT = Path("/data/caotue")


def get_data_root() -> Path:
    """Return the base data root for GLP outputs.

    The data root is fixed to `/data/caotue`. If `GLP_DATA_DIR` is set,
    it must resolve to that exact directory or an error is raised.
    """
    env = os.environ.get("GLP_DATA_DIR")
    if env:
        resolved = Path(env).expanduser().resolve()
        if resolved != _DATA_ROOT.resolve():
            raise ValueError(
                f"GLP_DATA_DIR must be /data/caotue, got {resolved}."
            )
    return _DATA_ROOT
