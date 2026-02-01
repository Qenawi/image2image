"""Django signals for image processing and cleanup."""
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from pathlib import Path

from .models import GenerationSession, GenerationIteration, Character, CharacterImage


@receiver(pre_delete, sender=GenerationSession)
def cleanup_session_files(sender, instance, **kwargs):
    """Clean up files when a session is deleted."""
    if instance.base_image:
        try:
            path = Path(instance.base_image.path)
            if path.exists():
                path.unlink()
        except Exception:
            pass


@receiver(pre_delete, sender=GenerationIteration)
def cleanup_iteration_files(sender, instance, **kwargs):
    """Clean up files when an iteration is deleted."""
    for field in [instance.input_image, instance.output_image]:
        if field:
            try:
                path = Path(field.path)
                if path.exists():
                    path.unlink()
            except Exception:
                pass


@receiver(pre_delete, sender=CharacterImage)
def cleanup_character_image(sender, instance, **kwargs):
    """Clean up image file when a character image is deleted."""
    if instance.image:
        try:
            path = Path(instance.image.path)
            if path.exists():
                path.unlink()
        except Exception:
            pass


@receiver(pre_delete, sender=Character)
def cleanup_character(sender, instance, **kwargs):
    """Clean up all character images when character is deleted."""
    for char_image in instance.images.all():
        if char_image.image:
            try:
                path = Path(char_image.image.path)
                if path.exists():
                    path.unlink()
            except Exception:
                pass


# Embedding cache invalidation signals
@receiver(post_save, sender=CharacterImage)
def invalidate_embedding_cache_on_add(sender, instance, created, **kwargs):
    """Invalidate embedding cache when a new image is added."""
    if created:
        try:
            from .services.embedding_cache import invalidate_cache
            invalidate_cache(instance.character)
        except Exception:
            pass


@receiver(pre_delete, sender=CharacterImage)
def invalidate_embedding_cache_on_delete(sender, instance, **kwargs):
    """Invalidate embedding cache when an image is deleted."""
    try:
        from .services.embedding_cache import invalidate_cache
        invalidate_cache(instance.character)
    except Exception:
        pass
