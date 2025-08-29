import os
import sys
import asyncio
import numpy as np
from typing import Optional

# Compute project root (repo root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure vendored LightRAG is importable
LIGHTRAG_PATH = os.path.join(PROJECT_ROOT, "external", "LightRAG")
if LIGHTRAG_PATH not in sys.path:
    sys.path.append(LIGHTRAG_PATH)

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

# Central artifacts directory for caches and tmp outputs
RUN_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "run_artifacts")
os.makedirs(RUN_ARTIFACTS_DIR, exist_ok=True)

# Singleton cache instance
_llm_cache_instance: Optional[JsonKVStorage] = None

_shared_init_done = False

def _create_cache() -> JsonKVStorage:
    global_config = {"working_dir": RUN_ARTIFACTS_DIR}
    return JsonKVStorage(
        namespace="llm_response_cache",
        workspace="",
        global_config=global_config,
        embedding_func=EmbeddingFunc(embedding_dim=1, func=lambda texts: np.zeros((len(texts), 1))),
    )

def bootstrap_llm_cache() -> JsonKVStorage:
    global _llm_cache_instance, _shared_init_done
    if not _shared_init_done:
        initialize_share_data(workers=1)        # Add this first
        import asyncio
        asyncio.run(initialize_pipeline_status())
        _shared_init_done = True
    if _llm_cache_instance is None:
        _llm_cache_instance = _create_cache()
        import asyncio
        asyncio.run(_llm_cache_instance.initialize())
    return _llm_cache_instance

def get_llm_cache() -> JsonKVStorage:
    global _llm_cache_instance, _shared_init_done
    if _llm_cache_instance is not None:
        return _llm_cache_instance
    # If an event loop is already running (e.g., inside LangGraph), require prior bootstrap
    try:
        asyncio.get_running_loop()
        if not _shared_init_done or _llm_cache_instance is None:
            raise RuntimeError("LLM cache not bootstrapped. Call bootstrap_llm_cache() before starting async workflows.")
    except RuntimeError:
        # no running loop; safe to bootstrap here
        return bootstrap_llm_cache()
    return _llm_cache_instance 