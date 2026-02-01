"""
Diffusion Engine - Singleton service for image-to-image generation.
Uses Hugging Face Diffusers library with lazy model loading.
Supports IP-Adapter for character-consistent generation.

Models must be downloaded manually using Hugging Face CLI:
    huggingface-cli download <model_id> --local-dir models_cache/<model_name>

Example:
    huggingface-cli download stabilityai/stable-diffusion-xl-refiner-1.0 --local-dir models_cache/stabilityai/stable-diffusion-xl-refiner-1.0
"""
import os
import gc
import time
import threading
from pathlib import Path
from PIL import Image
from io import BytesIO

from django.conf import settings

# Set cache directory before importing diffusers
_cache_dir = str(getattr(settings, 'MODELS_CACHE_DIR',
                         os.path.join(settings.BASE_DIR, 'models_cache')))
os.environ.setdefault('HF_HOME', _cache_dir)
os.environ.setdefault('TRANSFORMERS_CACHE', _cache_dir)

# Apply memory limits from settings
_max_ram_gb = getattr(settings, 'MAX_RAM_GB', 0)
if _max_ram_gb and _max_ram_gb > 0:
    try:
        import resource
        # Convert GB to bytes
        max_bytes = int(_max_ram_gb * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        print(f"Memory limit set to {_max_ram_gb} GB")
    except (ImportError, ValueError) as e:
        print(f"Could not set memory limit: {e}")


class ModelNotFoundError(Exception):
    """Raised when a required model is not found in the local cache."""
    pass


class DiffusionEngine:
    """
    Singleton engine for image-to-image generation using Hugging Face Diffusers.
    Implements lazy loading and thread-safe model access.
    Supports IP-Adapter for character-consistent generation with multiple reference images.
    """
    _instance = None
    _lock = threading.Lock()

    # Model instances (lazy-loaded)
    _pipelines = {}
    _pipeline_locks = {}
    _downloading = {}
    _ip_adapter_loaded = {}

    # Generation progress tracking
    _generation_progress = {
        'is_generating': False,
        'current_step': 0,
        'total_steps': 0,
        'progress_percent': 0,
        'started_at': None,
        'eta_seconds': None,
    }
    _progress_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._cache_dir = Path(_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hf_token = getattr(settings, 'HF_TOKEN', '') or os.getenv('HF_TOKEN', '')

        # Memory management settings
        self._auto_unload = getattr(settings, 'AUTO_UNLOAD_MODELS', False)
        self._max_loaded_models = getattr(settings, 'MAX_LOADED_MODELS', 1)
        self._clear_cache_after_gen = getattr(settings, 'CLEAR_CACHE_AFTER_GENERATION', True)
        self._model_load_order = []  # Track order for LRU unloading

        self._initialized = True

    def cleanup_resources(self):
        """Clean up GPU memory and run garbage collection."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't have empty_cache, but we can still gc
                pass
        except Exception as e:
            print(f"Error during GPU cleanup: {e}")

        # Force garbage collection
        gc.collect()

    def unload_model(self, model_id: str):
        """Unload a specific model from memory."""
        keys_to_remove = [k for k in self._pipelines.keys() if k.startswith(model_id)]

        for key in keys_to_remove:
            try:
                pipeline = self._pipelines.pop(key, None)
                if pipeline is not None:
                    del pipeline
                self._ip_adapter_loaded.pop(model_id, None)
            except Exception as e:
                print(f"Error unloading model {key}: {e}")

        if model_id in self._model_load_order:
            self._model_load_order.remove(model_id)

        self.cleanup_resources()
        print(f"Unloaded model: {model_id}")

    def unload_all_models(self):
        """Unload all models from memory."""
        model_ids = list(set(k.split('_')[0] for k in self._pipelines.keys()))
        for model_id in model_ids:
            self.unload_model(model_id)

        self._pipelines.clear()
        self._ip_adapter_loaded.clear()
        self._model_load_order.clear()
        self.cleanup_resources()
        print("All models unloaded")

    def _enforce_model_limit(self):
        """Unload oldest models if we exceed the limit."""
        if self._max_loaded_models <= 0:
            return

        # Count unique loaded models
        loaded_models = list(set(k.split('_')[0] for k in self._pipelines.keys()))

        while len(loaded_models) > self._max_loaded_models and self._model_load_order:
            oldest_model = self._model_load_order[0]
            if oldest_model in loaded_models:
                print(f"Unloading oldest model to stay within limit: {oldest_model}")
                self.unload_model(oldest_model)
                loaded_models = list(set(k.split('_')[0] for k in self._pipelines.keys()))

    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()

        result = {
            'process_rss_gb': round(mem_info.rss / (1024**3), 2),
            'process_vms_gb': round(mem_info.vms / (1024**3), 2),
            'system_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'system_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'system_percent_used': psutil.virtual_memory().percent,
            'loaded_models': list(set(k.split('_')[0] for k in self._pipelines.keys())),
            'max_loaded_models': self._max_loaded_models,
        }

        try:
            import torch
            if torch.cuda.is_available():
                result['gpu_allocated_gb'] = round(torch.cuda.memory_allocated() / (1024**3), 2)
                result['gpu_reserved_gb'] = round(torch.cuda.memory_reserved() / (1024**3), 2)
                result['gpu_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except Exception:
            pass

        return result

    def _progress_callback(self, step: int, timestep: int, latents):
        """Callback function for tracking generation progress."""
        with self._progress_lock:
            self._generation_progress['current_step'] = step + 1
            total = self._generation_progress['total_steps']
            if total > 0:
                self._generation_progress['progress_percent'] = int((step + 1) / total * 100)
                # Calculate ETA
                if self._generation_progress['started_at']:
                    elapsed = time.time() - self._generation_progress['started_at']
                    if step > 0:
                        time_per_step = elapsed / (step + 1)
                        remaining_steps = total - (step + 1)
                        self._generation_progress['eta_seconds'] = int(time_per_step * remaining_steps)

    def get_generation_progress(self) -> dict:
        """Get current generation progress."""
        with self._progress_lock:
            return self._generation_progress.copy()

    def _check_model_exists(self, model_id: str) -> bool:
        """Check if a model exists in the local cache."""
        # Check for model in cache directory (HuggingFace cache structure)
        # e.g., models_cache/models--stabilityai--stable-diffusion-xl-refiner-1.0
        model_path = self._cache_dir / f"models--{model_id.replace('/', '--')}"
        if model_path.exists():
            return True

        # Also check for direct path structure (from huggingface-cli download)
        direct_path = self._cache_dir / model_id
        if direct_path.exists():
            return True

        # Check hub cache structure
        hub_path = self._cache_dir / "hub" / f"models--{model_id.replace('/', '--')}"
        if hub_path.exists():
            return True

        return False

    def _get_pipeline(self, model_id: str, load_ip_adapter: bool = False):
        """Get or create a pipeline for the specified model.

        Raises:
            ModelNotFoundError: If the model is not found in the local cache.
        """
        if model_id not in self._pipeline_locks:
            self._pipeline_locks[model_id] = threading.Lock()

        with self._pipeline_locks[model_id]:
            pipeline_key = f"{model_id}_ipadapter" if load_ip_adapter else model_id

            if pipeline_key in self._pipelines:
                # Update LRU order
                if model_id in self._model_load_order:
                    self._model_load_order.remove(model_id)
                self._model_load_order.append(model_id)
                return self._pipelines[pipeline_key]

            # Enforce model limit before loading new model
            self._enforce_model_limit()

            # Check if model exists locally
            if not self._check_model_exists(model_id):
                raise ModelNotFoundError(
                    f"Model '{model_id}' not found in local cache.\n"
                    f"Please download it manually using Hugging Face CLI:\n\n"
                    f"    huggingface-cli download {model_id} --local-dir {self._cache_dir}/{model_id}\n\n"
                    f"Or download to HuggingFace cache:\n\n"
                    f"    HF_HOME={self._cache_dir} huggingface-cli download {model_id}\n\n"
                    f"Available models:\n"
                    f"  - stabilityai/stable-diffusion-xl-refiner-1.0\n"
                    f"  - runwayml/stable-diffusion-v1-5\n"
                    f"  - stabilityai/stable-diffusion-2-1\n"
                    f"  - timbrooks/instruct-pix2pix"
                )

            # Mark as loading
            self._downloading[model_id] = True
            lock_file = self._cache_dir / f'.{model_id.replace("/", "_")}.lock'
            lock_file.touch()

            try:
                import torch
                from diffusers import (
                    StableDiffusionImg2ImgPipeline,
                    StableDiffusionXLImg2ImgPipeline,
                    StableDiffusionInstructPix2PixPipeline,
                )

                # Determine device
                if torch.cuda.is_available():
                    device = "cuda"
                    dtype = torch.float16
                elif torch.backends.mps.is_available():
                    device = "mps"
                    dtype = torch.float16
                else:
                    device = "cpu"
                    dtype = torch.float32

                # Determine the actual path to load from
                # Check for direct download structure first (models--org--name/)
                direct_cache_path = self._cache_dir / f"models--{model_id.replace('/', '--')}"
                if direct_cache_path.exists() and (direct_cache_path / 'model_index.json').exists():
                    # Model was downloaded with --local-dir, use direct path
                    model_path = str(direct_cache_path)
                    kwargs = {
                        'torch_dtype': dtype,
                        'local_files_only': True,
                    }
                else:
                    # Try standard HuggingFace cache structure
                    model_path = model_id
                    kwargs = {
                        'cache_dir': str(self._cache_dir),
                        'torch_dtype': dtype,
                        'local_files_only': True,
                    }

                if self._hf_token:
                    kwargs['token'] = self._hf_token

                try:
                    if 'instruct-pix2pix' in model_id.lower():
                        # Skip safety_checker for InstructPix2Pix (not needed, may block swimwear)
                        kwargs['safety_checker'] = None
                        kwargs['requires_safety_checker'] = False
                        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                            model_path, **kwargs
                        )
                    elif 'xl' in model_id.lower() or 'sdxl' in model_id.lower():
                        # Disable safety checker for SDXL (may block swimwear)
                        kwargs['safety_checker'] = None
                        kwargs['requires_safety_checker'] = False
                        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                            model_path, **kwargs
                        )
                    else:
                        # Disable safety checker for SD 1.5/2.1 (may block swimwear)
                        kwargs['safety_checker'] = None
                        kwargs['requires_safety_checker'] = False
                        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                            model_path, **kwargs
                        )
                except OSError as e:
                    raise ModelNotFoundError(
                        f"Failed to load model '{model_id}' from local cache.\n"
                        f"The model files may be incomplete or corrupted.\n"
                        f"Please re-download using Hugging Face CLI:\n\n"
                        f"    HF_HOME={self._cache_dir} huggingface-cli download {model_id}\n\n"
                        f"Original error: {e}"
                    )

                pipeline = pipeline.to(device)

                # Enable memory optimizations
                if device == "cuda":
                    pipeline.enable_attention_slicing()
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass

                # Load IP-Adapter if requested and supported
                if load_ip_adapter:
                    try:
                        self._load_ip_adapter(pipeline, model_id, device)
                        self._ip_adapter_loaded[model_id] = True
                    except Exception as e:
                        print(f"Could not load IP-Adapter: {e}")
                        self._ip_adapter_loaded[model_id] = False

                self._pipelines[pipeline_key] = pipeline

                # Track load order for LRU
                if model_id in self._model_load_order:
                    self._model_load_order.remove(model_id)
                self._model_load_order.append(model_id)

                return pipeline

            finally:
                self._downloading[model_id] = False
                if lock_file.exists():
                    lock_file.unlink()

    def _load_ip_adapter(self, pipeline, model_id: str, device: str):
        """Load IP-Adapter for the pipeline.

        IP-Adapter must be downloaded manually:
            HF_HOME=models_cache huggingface-cli download h94/IP-Adapter
        """
        try:
            # IP-Adapter model selection based on base model
            if 'xl' in model_id.lower() or 'sdxl' in model_id.lower():
                ip_adapter_id = "h94/IP-Adapter"
                subfolder = "sdxl_models"
                weight_name = "ip-adapter_sdxl.bin"
            else:
                ip_adapter_id = "h94/IP-Adapter"
                subfolder = "models"
                weight_name = "ip-adapter_sd15.bin"

            pipeline.load_ip_adapter(
                ip_adapter_id,
                subfolder=subfolder,
                weight_name=weight_name,
                cache_dir=str(self._cache_dir),
                local_files_only=True,  # Prevent auto-download
            )
            print(f"IP-Adapter loaded successfully for {model_id}")
        except OSError as e:
            raise ModelNotFoundError(
                f"IP-Adapter not found in local cache.\n"
                f"Please download it manually using Hugging Face CLI:\n\n"
                f"    HF_HOME={self._cache_dir} huggingface-cli download h94/IP-Adapter\n\n"
                f"Original error: {e}"
            )
        except Exception as e:
            print(f"Failed to load IP-Adapter: {e}")
            raise

    def _prepare_character_images(self, image_paths: list, target_size: int = 224) -> list:
        """Prepare character reference images for IP-Adapter."""
        prepared_images = []

        for path in image_paths:
            try:
                img = Image.open(path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize while maintaining aspect ratio
                img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

                # Create square image with padding
                square_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
                offset = ((target_size - img.size[0]) // 2, (target_size - img.size[1]) // 2)
                square_img.paste(img, offset)

                prepared_images.append(square_img)
            except Exception as e:
                print(f"Error preparing image {path}: {e}")
                continue

        return prepared_images

    def generate(
        self,
        prompt: str,
        image: Image.Image = None,
        model_id: str = None,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        negative_prompt: str = "",
        character_images: list = None,
        ip_adapter_scale: float = 0.6,
        use_refiner: bool = False,
    ) -> tuple[Image.Image, float]:
        """
        Generate an image from prompt and optional input image.

        Args:
            prompt: Text description of desired output
            image: Optional input image for img2img
            model_id: Hugging Face model ID
            strength: How much to transform the input (0-1)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            negative_prompt: What to avoid in generation
            character_images: List of image paths for character consistency
            ip_adapter_scale: Weight of IP-Adapter influence (0-1)
            use_refiner: If True and using SDXL Base, apply SDXL Refiner after

        Returns:
            Tuple of (generated_image, generation_time_seconds)
        """
        if model_id is None:
            model_id = getattr(settings, 'DEFAULT_MODEL_ID',
                              'stabilityai/stable-diffusion-xl-refiner-1.0')

        # Load IP-Adapter if character images provided
        use_ip_adapter = character_images and len(character_images) > 0
        pipeline = self._get_pipeline(model_id, load_ip_adapter=use_ip_adapter)

        start_time = time.time()

        # Initialize progress tracking
        with self._progress_lock:
            self._generation_progress = {
                'is_generating': True,
                'current_step': 0,
                'total_steps': num_inference_steps,
                'progress_percent': 0,
                'started_at': start_time,
                'eta_seconds': None,
            }

        # Prepare character reference images
        ip_adapter_images = None
        if use_ip_adapter and self._ip_adapter_loaded.get(model_id, False):
            ip_adapter_images = self._prepare_character_images(character_images)
            if ip_adapter_images:
                pipeline.set_ip_adapter_scale(ip_adapter_scale)

        # Prepare image if provided
        if image is not None:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize to model's expected size if needed
            max_size = 1024 if 'xl' in model_id.lower() else 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                # Make dimensions divisible by 8
                new_size = (new_size[0] - new_size[0] % 8, new_size[1] - new_size[1] % 8)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Generate
        with self._pipeline_locks.get(model_id, threading.Lock()):
            gen_kwargs = {
                'prompt': prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
            }

            if negative_prompt:
                gen_kwargs['negative_prompt'] = negative_prompt

            # Add IP-Adapter images if available
            if ip_adapter_images and self._ip_adapter_loaded.get(model_id, False):
                # For single IP-Adapter with multiple reference images, wrap in list
                # so diffusers knows all images are for the one adapter
                if len(ip_adapter_images) == 1:
                    gen_kwargs['ip_adapter_image'] = ip_adapter_images[0]
                else:
                    gen_kwargs['ip_adapter_image'] = [ip_adapter_images]

            # Add progress callback
            gen_kwargs['callback'] = self._progress_callback
            gen_kwargs['callback_steps'] = 1

            try:
                if 'instruct-pix2pix' in model_id.lower():
                    gen_kwargs['image'] = image
                    # image_guidance_scale: lower = more change, higher = preserve original
                    # Use strength to control: strength 0.8 -> image_guidance_scale 1.2 (more change)
                    # strength 0.5 -> image_guidance_scale 1.5 (less change)
                    image_guidance = 1.0 + (1.0 - strength)  # Range: 1.0-2.0 based on strength
                    gen_kwargs['image_guidance_scale'] = image_guidance
                    result = pipeline(**gen_kwargs).images[0]

                elif image is not None:
                    gen_kwargs['image'] = image
                    gen_kwargs['strength'] = strength
                    result = pipeline(**gen_kwargs).images[0]

                else:
                    # Text-to-image fallback
                    # For img2img pipelines, we need to provide an image
                    import numpy as np

                    size = 1024 if 'xl' in model_id.lower() else 512
                    noise_image = Image.fromarray(
                        np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                    )
                    gen_kwargs['image'] = noise_image
                    gen_kwargs['strength'] = 0.99
                    result = pipeline(**gen_kwargs).images[0]
            finally:
                # Clear progress tracking
                with self._progress_lock:
                    self._generation_progress['is_generating'] = False
                    self._generation_progress['progress_percent'] = 100

        # Apply SDXL Refiner if requested and using SDXL Base
        if use_refiner and 'xl-base' in model_id.lower():
            refiner_id = 'stabilityai/stable-diffusion-xl-refiner-1.0'
            if self._check_model_exists(refiner_id):
                try:
                    # Update progress for refiner stage
                    with self._progress_lock:
                        self._generation_progress['is_generating'] = True
                        self._generation_progress['current_step'] = 0
                        self._generation_progress['total_steps'] = 20
                        self._generation_progress['progress_percent'] = 0

                    refiner_pipeline = self._get_pipeline(refiner_id)
                    refiner_kwargs = {
                        'prompt': prompt,
                        'image': result,
                        'strength': 0.3,  # Light refinement
                        'guidance_scale': guidance_scale,
                        'num_inference_steps': 20,
                        'callback': self._progress_callback,
                        'callback_steps': 1,
                    }
                    if negative_prompt:
                        refiner_kwargs['negative_prompt'] = negative_prompt

                    result = refiner_pipeline(**refiner_kwargs).images[0]
                except Exception as e:
                    print(f"Refiner failed, using base result: {e}")
                finally:
                    with self._progress_lock:
                        self._generation_progress['is_generating'] = False
                        self._generation_progress['progress_percent'] = 100

        generation_time = time.time() - start_time

        # Cleanup resources after generation
        if self._clear_cache_after_gen:
            self.cleanup_resources()

        # Auto-unload models if enabled
        if self._auto_unload:
            self.unload_all_models()

        return result, generation_time

    def extract_ip_adapter_embeddings(
        self,
        image_paths: list,
        model_id: str = None,
    ):
        """
        Extract IP-Adapter CLIP embeddings from images.

        Args:
            image_paths: List of paths to character reference images
            model_id: Model ID to determine SDXL vs SD1.5

        Returns:
            Tuple of (embeddings_tensor, model_type)
            - SDXL: shape (batch, 1024)
            - SD1.5: shape (batch, 768)
        """
        import torch

        if model_id is None:
            model_id = getattr(settings, 'DEFAULT_MODEL_ID',
                              'stabilityai/stable-diffusion-xl-base-1.0')

        # Load pipeline with IP-Adapter to get image_encoder
        pipeline = self._get_pipeline(model_id, load_ip_adapter=True)

        if not hasattr(pipeline, 'image_encoder') or pipeline.image_encoder is None:
            raise RuntimeError("Pipeline does not have image_encoder loaded")

        # Determine device and dtype
        device = next(pipeline.image_encoder.parameters()).device
        dtype = next(pipeline.image_encoder.parameters()).dtype

        # Prepare images using existing method
        prepared_images = self._prepare_character_images(image_paths)

        if not prepared_images:
            raise ValueError("No valid images to process")

        # Use pipeline's feature_extractor to preprocess
        pixel_values = pipeline.feature_extractor(
            prepared_images,
            return_tensors="pt"
        ).pixel_values.to(device=device, dtype=dtype)

        # Extract embeddings
        with torch.no_grad():
            image_embeds = pipeline.image_encoder(pixel_values).image_embeds

        model_type = 'sdxl' if 'xl' in model_id.lower() else 'sd15'
        return image_embeds.cpu(), model_type

    def generate_with_cached_embeddings(
        self,
        prompt: str,
        ip_adapter_image_embeds,
        image: Image.Image = None,
        model_id: str = None,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        negative_prompt: str = "",
        ip_adapter_scale: float = 0.6,
    ) -> tuple[Image.Image, float]:
        """
        Generate image using pre-computed IP-Adapter embeddings.

        Args:
            prompt: Text description of desired output
            ip_adapter_image_embeds: Pre-computed CLIP embeddings tensor
            image: Optional input image for img2img
            model_id: Hugging Face model ID
            strength: How much to transform the input (0-1)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            negative_prompt: What to avoid in generation
            ip_adapter_scale: Weight of IP-Adapter influence (0-1)

        Returns:
            Tuple of (generated_image, generation_time_seconds)
        """
        import torch

        if model_id is None:
            model_id = getattr(settings, 'DEFAULT_MODEL_ID',
                              'stabilityai/stable-diffusion-xl-base-1.0')

        # Load pipeline with IP-Adapter
        pipeline = self._get_pipeline(model_id, load_ip_adapter=True)

        start_time = time.time()

        # Initialize progress tracking
        with self._progress_lock:
            self._generation_progress = {
                'is_generating': True,
                'current_step': 0,
                'total_steps': num_inference_steps,
                'progress_percent': 0,
                'started_at': start_time,
                'eta_seconds': None,
            }

        # Set IP-Adapter scale
        pipeline.set_ip_adapter_scale(ip_adapter_scale)

        # Prepare embeddings for pipeline
        # Move to device and prepare for classifier-free guidance
        device = pipeline.device
        dtype = next(pipeline.image_encoder.parameters()).dtype

        if isinstance(ip_adapter_image_embeds, torch.Tensor):
            embeds = ip_adapter_image_embeds.to(device=device, dtype=dtype)
        else:
            embeds = torch.tensor(ip_adapter_image_embeds).to(device=device, dtype=dtype)

        # Ensure embeddings are 2D: (num_images, embed_dim)
        if embeds.dim() == 1:
            embeds = embeds.unsqueeze(0)

        # Average embeddings if multiple images to get single reference
        if embeds.shape[0] > 1:
            embeds = embeds.mean(dim=0, keepdim=True)

        # Create uncond embeddings (zeros) for classifier-free guidance
        uncond_embeds = torch.zeros_like(embeds)

        # Stack for CFG: shape becomes (2, 1, embed_dim) -> need (2, num_images, embed_dim)
        # Then we need to make it 3D for the pipeline
        combined_embeds = torch.cat([uncond_embeds, embeds], dim=0)  # (2, embed_dim)

        # Add batch dimension to make it 3D: (batch*2, 1, embed_dim)
        # The pipeline expects: list of tensors with shape (batch_size * 2, num_images, embed_dim)
        combined_embeds = combined_embeds.unsqueeze(1)  # (2, 1, embed_dim)

        # Prepare image if provided
        if image is not None:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            max_size = 1024 if 'xl' in model_id.lower() else 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                new_size = (new_size[0] - new_size[0] % 8, new_size[1] - new_size[1] % 8)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Generate
        with self._pipeline_locks.get(model_id, threading.Lock()):
            gen_kwargs = {
                'prompt': prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'ip_adapter_image_embeds': [combined_embeds],
            }

            if negative_prompt:
                gen_kwargs['negative_prompt'] = negative_prompt

            gen_kwargs['callback'] = self._progress_callback
            gen_kwargs['callback_steps'] = 1

            try:
                if image is not None:
                    gen_kwargs['image'] = image
                    gen_kwargs['strength'] = strength
                    result = pipeline(**gen_kwargs).images[0]
                else:
                    # Text-to-image with noise
                    import numpy as np
                    size = 1024 if 'xl' in model_id.lower() else 512
                    noise_image = Image.fromarray(
                        np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                    )
                    gen_kwargs['image'] = noise_image
                    gen_kwargs['strength'] = 0.99
                    result = pipeline(**gen_kwargs).images[0]
            finally:
                with self._progress_lock:
                    self._generation_progress['is_generating'] = False
                    self._generation_progress['progress_percent'] = 100

        generation_time = time.time() - start_time

        # Cleanup resources after generation
        if self._clear_cache_after_gen:
            self.cleanup_resources()

        # Auto-unload models if enabled
        if self._auto_unload:
            self.unload_all_models()

        return result, generation_time

    def status(self) -> dict:
        """Get current engine status."""
        loading = [k for k, v in self._downloading.items() if v]
        loaded = list(set(k.split('_')[0] for k in self._pipelines.keys()))

        # Check which models are downloaded
        available_models = self.get_available_models()
        downloaded = [m['id'] for m in available_models if self._check_model_exists(m['id'])]
        missing = [m['id'] for m in available_models if not self._check_model_exists(m['id'])]

        # Get memory usage
        memory = self.get_memory_usage()

        # Get generation progress
        progress = self.get_generation_progress()

        return {
            'ready': len(loading) == 0 and not progress['is_generating'],
            'loading': loading,
            'loaded_models': loaded,
            'downloaded_models': downloaded,
            'missing_models': missing,
            'ip_adapter_supported': list(self._ip_adapter_loaded.keys()),
            'cache_dir': str(self._cache_dir),
            'has_token': bool(self._hf_token),
            'generation': progress,
            'memory': memory,
            'settings': {
                'auto_unload_models': self._auto_unload,
                'max_loaded_models': self._max_loaded_models,
                'clear_cache_after_generation': self._clear_cache_after_gen,
            },
        }

    def get_available_models(self) -> list:
        """Get list of available models with download status."""
        models = [
            {
                'id': 'stabilityai/stable-diffusion-xl-base-1.0',
                'name': 'SDXL Base 1.0',
                'description': 'Best quality model. 1024x1024 resolution, excellent anatomy and faces, best prompt understanding. Ideal for professional advertising and swimwear. Supports character consistency.',
                'supports_ip_adapter': True
            },
            {
                'id': 'stabilityai/stable-diffusion-xl-refiner-1.0',
                'name': 'SDXL Refiner 1.0',
                'description': 'Refinement model only - use after SDXL Base to enhance details. NOT for standalone generation.',
                'supports_ip_adapter': True
            },
            {
                'id': 'runwayml/stable-diffusion-v1-5',
                'name': 'Stable Diffusion 1.5',
                'description': 'General-purpose image generation. Fast with good quality balance. Supports character consistency.',
                'supports_ip_adapter': True
            },
            {
                'id': 'stabilityai/stable-diffusion-2-1',
                'name': 'Stable Diffusion 2.1',
                'description': 'Improved version of SD 1.5 with better quality and detail. Supports character consistency.',
                'supports_ip_adapter': True
            },
            {
                'id': 'timbrooks/instruct-pix2pix',
                'name': 'InstructPix2Pix',
                'description': 'Edit existing images using natural language instructions. Best for "make it X" style edits like "make it sunset", "turn into watercolor", "add snow". Does NOT support character consistency.',
                'supports_ip_adapter': False
            },
        ]

        # Add download status to each model
        for model in models:
            model['downloaded'] = self._check_model_exists(model['id'])

        return models


# Global singleton instance
engine = DiffusionEngine()
