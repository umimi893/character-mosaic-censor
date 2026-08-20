"""Public batch-processing API.

Implementation is split into focused modules so GUI, CLI, review persistence,
and file I/O can evolve independently while keeping the historical import
surface stable.
"""

from .pipeline_config import PipelineConfig
from .pipeline_logging import JsonlRunLogger, write_jsonl_log
from .pipeline_processor import BatchProcessor, discover_images, validate_processing_paths
from .pipeline_review import write_review_html

__all__ = [
    "BatchProcessor",
    "PipelineConfig",
    "JsonlRunLogger",
    "discover_images",
    "validate_processing_paths",
    "write_jsonl_log",
    "write_review_html",
]
