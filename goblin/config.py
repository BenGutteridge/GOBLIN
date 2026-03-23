"""Global configuration for GOBLIN.

To override the data cache location, set the GOBLIN_DATA_CACHE environment variable
before importing goblin, or set goblin.config.DATA_CACHE directly:

    import goblin.config as goblin_cfg
    from pathlib import Path
    goblin_cfg.DATA_CACHE = Path("/path/to/my/data_cache")
"""
import os
from pathlib import Path

DATA_CACHE: Path = Path(os.environ.get("GOBLIN_DATA_CACHE", "data_cache"))
