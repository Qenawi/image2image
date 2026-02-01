import uuid
from django.db import models
from django.core.validators import FileExtensionValidator


class Character(models.Model):
    """Represents a character with multiple reference images."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(
        blank=True,
        help_text='Describe the character appearance, style, etc.'
    )
    trigger_word = models.CharField(
        max_length=100,
        blank=True,
        help_text='Optional trigger word to use in prompts (e.g., "CHARNAME")'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Character'
        verbose_name_plural = 'Characters'

    def __str__(self):
        return self.name

    @property
    def image_count(self):
        return self.images.count()

    @property
    def primary_image(self):
        return self.images.filter(is_primary=True).first() or self.images.first()

    @property
    def all_image_paths(self):
        return [img.image.path for img in self.images.all() if img.image]


class CharacterImage(models.Model):
    """Reference image for a character."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    character = models.ForeignKey(
        Character,
        on_delete=models.CASCADE,
        related_name='images'
    )
    image = models.ImageField(
        upload_to='character_images/%Y/%m/',
        validators=[FileExtensionValidator(['jpg', 'jpeg', 'png', 'webp'])]
    )
    original_filename = models.CharField(max_length=255, blank=True)
    is_primary = models.BooleanField(
        default=False,
        help_text='Primary image used as main reference'
    )
    caption = models.CharField(
        max_length=500,
        blank=True,
        help_text='Optional caption describing this specific image'
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-is_primary', 'uploaded_at']
        verbose_name = 'Character Image'
        verbose_name_plural = 'Character Images'

    def __str__(self):
        return f"{self.character.name} - Image {str(self.id)[:8]}"

    def save(self, *args, **kwargs):
        if self.is_primary:
            # Ensure only one primary image per character
            CharacterImage.objects.filter(
                character=self.character, is_primary=True
            ).exclude(pk=self.pk).update(is_primary=False)
        super().save(*args, **kwargs)


class ModelConfig(models.Model):
    """Stores model configurations for image generation."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    model_id = models.CharField(
        max_length=255,
        default='stabilityai/stable-diffusion-xl-refiner-1.0',
        help_text='Hugging Face model ID'
    )
    strength = models.FloatField(
        default=0.75,
        help_text='Transformation strength (0.0-1.0). Higher = more change'
    )
    guidance_scale = models.FloatField(
        default=7.5,
        help_text='How closely to follow the prompt (1-20)'
    )
    num_inference_steps = models.IntegerField(
        default=50,
        help_text='Number of denoising steps (20-100)'
    )
    negative_prompt = models.TextField(
        blank=True,
        default='blurry, low quality, distorted, deformed',
        help_text='What to avoid in generation'
    )
    is_active = models.BooleanField(
        default=False,
        help_text='Use this config for next generation'
    )
    use_refiner = models.BooleanField(
        default=False,
        help_text='Apply SDXL Refiner after SDXL Base for enhanced details'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Model Configuration'
        verbose_name_plural = 'Model Configurations'

    def __str__(self):
        return f"{self.name} ({'Active' if self.is_active else 'Inactive'})"

    def save(self, *args, **kwargs):
        if self.is_active:
            # Deactivate other configs when this one is activated
            ModelConfig.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)


class GenerationSession(models.Model):
    """Represents a generation session with iterative improvements."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, blank=True)
    base_image = models.ImageField(
        upload_to='uploaded_images/%Y/%m/',
        validators=[FileExtensionValidator(['jpg', 'jpeg', 'png', 'webp'])],
        null=True,
        blank=True,
        help_text='Optional base image for img2img'
    )
    character = models.ForeignKey(
        Character,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='sessions',
        help_text='Character to use for consistent generation'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Generation Session'
        verbose_name_plural = 'Generation Sessions'

    def __str__(self):
        return self.name or f"Session {str(self.id)[:8]}"

    @property
    def iteration_count(self):
        return self.iterations.count()

    @property
    def latest_iteration(self):
        return self.iterations.order_by('-created_at').first()


class GenerationIteration(models.Model):
    """Single iteration within a generation session."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        GenerationSession,
        on_delete=models.CASCADE,
        related_name='iterations'
    )
    iteration_number = models.PositiveIntegerField(default=1)
    prompt = models.TextField(help_text='Description of changes to apply')
    input_image = models.ImageField(
        upload_to='generated_images/%Y/%m/',
        null=True,
        blank=True,
        help_text='Input image for this iteration'
    )
    output_image = models.ImageField(
        upload_to='generated_images/%Y/%m/',
        null=True,
        blank=True,
        help_text='Generated output image'
    )
    model_config = models.ForeignKey(
        ModelConfig,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='iterations'
    )
    character = models.ForeignKey(
        Character,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='iterations',
        help_text='Character used for this generation'
    )
    # Store config values at generation time for history
    config_snapshot = models.JSONField(default=dict, blank=True)
    generation_time = models.FloatField(null=True, blank=True, help_text='Time in seconds')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['session', 'iteration_number']
        verbose_name = 'Generation Iteration'
        verbose_name_plural = 'Generation Iterations'

    def __str__(self):
        return f"{self.session} - Iteration {self.iteration_number}"

    def save(self, *args, **kwargs):
        if not self.iteration_number:
            last_iteration = GenerationIteration.objects.filter(
                session=self.session
            ).order_by('-iteration_number').first()
            self.iteration_number = (last_iteration.iteration_number + 1) if last_iteration else 1
        super().save(*args, **kwargs)


class EnhancementJob(models.Model):
    """Tracks enhancement generation jobs for a character."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    character = models.ForeignKey(
        Character,
        on_delete=models.CASCADE,
        related_name='enhancement_jobs'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_variations = models.IntegerField(default=0)
    completed_variations = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Enhancement Job'
        verbose_name_plural = 'Enhancement Jobs'

    def __str__(self):
        return f"Enhancement for {self.character.name} ({self.status})"

    @property
    def progress_percent(self):
        if self.total_variations == 0:
            return 0
        return int(self.completed_variations / self.total_variations * 100)


class EnhancedCharacterImage(models.Model):
    """Generated enhanced variation of a character image."""
    VARIATION_TYPES = [
        ('front_neutral', 'Front-facing, neutral expression'),
        ('three_quarter_smile', '3/4 angle, slight smile'),
        ('side_profile', 'Side profile'),
        ('different_lighting', 'Different lighting condition'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        EnhancementJob,
        on_delete=models.CASCADE,
        related_name='enhanced_images'
    )
    source_image = models.ForeignKey(
        CharacterImage,
        on_delete=models.CASCADE,
        related_name='enhancements'
    )
    variation_type = models.CharField(max_length=30, choices=VARIATION_TYPES)
    image = models.ImageField(upload_to='enhanced_images/%Y/%m/')
    generation_time = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['source_image', 'variation_type']
        verbose_name = 'Enhanced Character Image'
        verbose_name_plural = 'Enhanced Character Images'

    def __str__(self):
        return f"{self.source_image.character.name} - {self.get_variation_type_display()}"


class CharacterEmbeddingCache(models.Model):
    """Cached IP-Adapter CLIP embeddings for a character."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    character = models.OneToOneField(
        Character,
        on_delete=models.CASCADE,
        related_name='embedding_cache'
    )
    model_type = models.CharField(
        max_length=20,
        help_text='sdxl or sd15 - determines embedding dimension'
    )
    embedding_file = models.FileField(
        upload_to='embeddings_cache/%Y/%m/',
        help_text='Pickled tensor file containing embeddings'
    )
    image_hash = models.CharField(
        max_length=64,
        help_text='Hash of all source image paths to detect changes'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Character Embedding Cache'
        verbose_name_plural = 'Character Embedding Caches'

    def __str__(self):
        return f"Embeddings for {self.character.name} ({self.model_type})"
