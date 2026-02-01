"""
Embedding cache service for IP-Adapter CLIP embeddings.
Handles caching, invalidation, and retrieval of pre-computed embeddings.
"""
import hashlib
import pickle
from io import BytesIO

import torch
from django.core.files.base import ContentFile


def compute_image_hash(image_paths: list) -> str:
    """Compute hash of sorted image paths for cache invalidation."""
    sorted_paths = sorted(str(p) for p in image_paths)
    combined = '|'.join(sorted_paths)
    return hashlib.sha256(combined.encode()).hexdigest()


def get_cached_embeddings(character, model_type: str = 'sdxl'):
    """
    Get cached embeddings for a character if valid.

    Returns None if:
    - No cache exists
    - Model type mismatch
    - Images have changed (hash mismatch)
    """
    from img2img_app.models import CharacterEmbeddingCache

    try:
        cache = character.embedding_cache
    except CharacterEmbeddingCache.DoesNotExist:
        return None

    # Check model type matches
    if cache.model_type != model_type:
        return None

    # Check if images have changed
    current_hash = compute_image_hash(character.all_image_paths)
    if cache.image_hash != current_hash:
        return None

    # Load and return embeddings
    try:
        with cache.embedding_file.open('rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def cache_embeddings(character, embeddings: torch.Tensor, model_type: str):
    """
    Cache embeddings for a character.
    Replaces existing cache if present.
    """
    from img2img_app.models import CharacterEmbeddingCache

    image_hash = compute_image_hash(character.all_image_paths)

    # Serialize embeddings
    buffer = BytesIO()
    pickle.dump(embeddings, buffer)
    buffer.seek(0)

    # Delete existing cache if present
    CharacterEmbeddingCache.objects.filter(character=character).delete()

    # Create new cache entry
    cache = CharacterEmbeddingCache(
        character=character,
        model_type=model_type,
        image_hash=image_hash,
    )
    cache.embedding_file.save(
        f'embeddings_{character.id}_{model_type}.pkl',
        ContentFile(buffer.read())
    )
    cache.save()

    return cache


def invalidate_cache(character):
    """Invalidate (delete) cached embeddings for a character."""
    from img2img_app.models import CharacterEmbeddingCache
    CharacterEmbeddingCache.objects.filter(character=character).delete()


def get_or_compute_embeddings(character, model_id: str = None):
    """
    Get cached embeddings or compute and cache them.

    Returns:
        Tuple of (embeddings_tensor, model_type)
    """
    from img2img_app.services.diffusion_engine import engine

    model_type = 'sdxl' if model_id and 'xl' in model_id.lower() else 'sd15'

    # Try cache first
    cached = get_cached_embeddings(character, model_type)
    if cached is not None:
        return cached, model_type

    # Compute embeddings
    embeddings, computed_type = engine.extract_ip_adapter_embeddings(
        character.all_image_paths,
        model_id
    )

    # Cache them
    cache_embeddings(character, embeddings, computed_type)

    return embeddings, computed_type
