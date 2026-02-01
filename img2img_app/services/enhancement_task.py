"""
Background task handler for enhancement generation.
Uses threading for async operation without Celery dependency.
"""
import threading
from io import BytesIO

from django.utils import timezone
from django.core.files.base import ContentFile


# Enhancement prompts for each variation type
ENHANCEMENT_PROMPTS = {
    'front_neutral': "portrait, front-facing view, neutral expression, direct eye contact, centered composition, professional photo",
    'three_quarter_smile': "portrait, three-quarter angle view, slight natural smile, warm expression, professional photo",
    'side_profile': "portrait, side profile view, looking to the side, profile angle, professional photo",
    'different_lighting': "portrait, dramatic studio lighting, rim light, soft shadows, professional photography, artistic lighting",
}

# Track running jobs
_running_jobs = {}
_jobs_lock = threading.Lock()


def start_enhancement_job(character_id: str, model_id: str = None):
    """
    Start an enhancement job for a character.
    Returns the job object immediately; generation runs in background.
    """
    from img2img_app.models import Character, EnhancementJob

    character = Character.objects.get(id=character_id)

    # Check for existing running job
    existing = EnhancementJob.objects.filter(
        character=character,
        status__in=['pending', 'processing']
    ).first()
    if existing:
        return existing

    # Calculate total variations: 4 per image
    total_images = character.images.count()
    total_variations = total_images * len(ENHANCEMENT_PROMPTS)

    # Create job record
    job = EnhancementJob.objects.create(
        character=character,
        status='pending',
        total_variations=total_variations,
    )

    # Start background thread
    thread = threading.Thread(
        target=_run_enhancement_job,
        args=(str(job.id), str(character_id), model_id),
        daemon=True
    )

    with _jobs_lock:
        _running_jobs[str(job.id)] = thread

    thread.start()

    return job


def _run_enhancement_job(job_id: str, character_id: str, model_id: str):
    """Background worker function for enhancement generation."""
    import django
    django.setup()

    from img2img_app.models import Character, EnhancementJob, EnhancedCharacterImage
    from img2img_app.services.diffusion_engine import engine
    from img2img_app.services.embedding_cache import get_or_compute_embeddings

    try:
        job = EnhancementJob.objects.get(id=job_id)
        character = Character.objects.get(id=character_id)

        job.status = 'processing'
        job.started_at = timezone.now()
        job.save()

        # Get or compute cached embeddings
        embeddings, model_type = get_or_compute_embeddings(character, model_id)

        # Generate variations for each source image
        for source_image in character.images.all():
            for var_type, prompt in ENHANCEMENT_PROMPTS.items():
                try:
                    # Check if this variation already exists
                    existing = EnhancedCharacterImage.objects.filter(
                        job=job,
                        source_image=source_image,
                        variation_type=var_type
                    ).exists()
                    if existing:
                        continue

                    # Generate with cached embeddings
                    result_image, gen_time = engine.generate_with_cached_embeddings(
                        prompt=prompt,
                        ip_adapter_image_embeds=embeddings,
                        model_id=model_id,
                        ip_adapter_scale=0.7,
                        strength=0.85,
                        guidance_scale=7.5,
                        num_inference_steps=30,
                    )

                    # Save enhanced image
                    buffer = BytesIO()
                    result_image.save(buffer, format='PNG')
                    buffer.seek(0)

                    enhanced = EnhancedCharacterImage(
                        job=job,
                        source_image=source_image,
                        variation_type=var_type,
                        generation_time=gen_time,
                    )
                    enhanced.image.save(
                        f'enhanced_{source_image.id}_{var_type}.png',
                        ContentFile(buffer.read())
                    )
                    enhanced.save()

                    # Update progress
                    job.completed_variations += 1
                    job.save()

                except Exception as e:
                    print(f"Error generating {var_type} for {source_image.id}: {e}")
                    import traceback
                    traceback.print_exc()

        job.status = 'completed'
        job.completed_at = timezone.now()
        job.save()

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            job = EnhancementJob.objects.get(id=job_id)
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save()
        except Exception:
            pass

    finally:
        with _jobs_lock:
            _running_jobs.pop(str(job_id), None)


def get_job_status(job_id: str) -> dict:
    """Get current status of an enhancement job."""
    from img2img_app.models import EnhancementJob

    try:
        job = EnhancementJob.objects.get(id=job_id)
        return {
            'id': str(job.id),
            'status': job.status,
            'total': job.total_variations,
            'completed': job.completed_variations,
            'progress_percent': job.progress_percent,
            'error': job.error_message if job.status == 'failed' else None,
        }
    except EnhancementJob.DoesNotExist:
        return {'error': 'Job not found'}


def is_job_running(job_id: str) -> bool:
    """Check if a job is currently running."""
    with _jobs_lock:
        return str(job_id) in _running_jobs
