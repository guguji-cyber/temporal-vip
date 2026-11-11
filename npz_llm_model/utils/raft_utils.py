"""Helpers to download RAFT checkpoints and manage default paths."""
import os
import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = {
    'raft_things': 'https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth',
    'raft_sintel': 'https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-sintel.pth',
    'raft_small': 'https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-small.pth'
}


def download_raft_checkpoint(name: str = 'raft_things', dest_dir: Optional[str] = None, url: Optional[str] = None) -> Optional[str]:
    """Download a RAFT checkpoint from a default location or a provided URL.

    Args:
        name: one of keys in DEFAULT_CHECKPOINTS (used for filename) if url not given
        dest_dir: directory to save the checkpoint (defaults to project's models/raft)
        url: optional explicit URL to download from (takes precedence)

    Returns:
        local_path or None if failed
    """
    if dest_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../npz_llm_model/utils -> ../npz_llm_model
        dest_dir = os.path.join(base_dir, 'models', 'raft')
    os.makedirs(dest_dir, exist_ok=True)

    if url is None:
        if name not in DEFAULT_CHECKPOINTS:
            logger.error(f"Unknown RAFT checkpoint name: {name}")
            return None
        url = DEFAULT_CHECKPOINTS[name]

    # derive filename
    filename = os.path.basename(url)
    if not filename.endswith('.pth'):
        # fall back to name-based filename
        filename = (name + '.pth') if not name.endswith('.pth') else name
    dest_path = os.path.join(dest_dir, filename)

    # If already exists, return
    if os.path.exists(dest_path):
        logger.info(f"RAFT checkpoint already exists at {dest_path}")
        return dest_path

    try:
        logger.info(f"Downloading RAFT checkpoint from {url} to {dest_path}")
        torch.hub.download_url_to_file(url, dest_path, progress=True)
        logger.info(f"Downloaded RAFT checkpoint to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to download RAFT checkpoint from {url}: {e}")
        logger.error("You can provide a custom URL using the `url` argument or place a checkpoint manually in code/src/models/raft and set --raft_model_path to its path.")
        return None
