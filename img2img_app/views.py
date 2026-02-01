import json
import uuid
from io import BytesIO
from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.views import View
from django.urls import reverse_lazy
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image

from .models import (
    ModelConfig, GenerationSession, GenerationIteration,
    Character, CharacterImage, EnhancementJob, EnhancedCharacterImage
)
from .forms import ModelConfigForm, GenerationForm, SessionForm, CharacterForm, CharacterImageForm
from .services.diffusion_engine import engine, ModelNotFoundError


class IndexView(View):
    """Main generation interface."""
    template_name = 'img2img_app/index.html'

    def get(self, request):
        active_config = ModelConfig.objects.filter(is_active=True).first()
        sessions = GenerationSession.objects.all()[:10]
        characters = Character.objects.all()
        form = GenerationForm()
        available_models = engine.get_available_models()

        # Get info about active model (for IP-Adapter support check)
        active_model_info = None
        if active_config:
            for m in available_models:
                if m['id'] == active_config.model_id:
                    active_model_info = m
                    break

        return render(request, self.template_name, {
            'form': form,
            'active_config': active_config,
            'active_model_info': active_model_info,
            'sessions': sessions,
            'characters': characters,
            'available_models': available_models,
        })


# Configuration Views
class ConfigListView(ListView):
    """List all model configurations."""
    model = ModelConfig
    template_name = 'img2img_app/config_list.html'
    context_object_name = 'configs'
    paginate_by = 10

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        available_models = engine.get_available_models()
        # Create lookup dict for quick access in template
        context['model_status'] = {
            m['id']: {
                'downloaded': m['downloaded'],
                'supports_ip_adapter': m['supports_ip_adapter'],
                'name': m['name']
            }
            for m in available_models
        }
        return context


class ConfigCreateView(CreateView):
    """Create a new model configuration."""
    model = ModelConfig
    form_class = ModelConfigForm
    template_name = 'img2img_app/config_form.html'
    success_url = reverse_lazy('img2img_app:config_list')

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['available_models'] = engine.get_available_models()
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['available_models'] = engine.get_available_models()
        return context


class ConfigUpdateView(UpdateView):
    """Update an existing model configuration."""
    model = ModelConfig
    form_class = ModelConfigForm
    template_name = 'img2img_app/config_form.html'
    success_url = reverse_lazy('img2img_app:config_list')

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['available_models'] = engine.get_available_models()
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['available_models'] = engine.get_available_models()
        return context


class ConfigDeleteView(DeleteView):
    """Delete a model configuration."""
    model = ModelConfig
    template_name = 'img2img_app/config_confirm_delete.html'
    success_url = reverse_lazy('img2img_app:config_list')


# Character Views
class CharacterListView(ListView):
    """List all characters."""
    model = Character
    template_name = 'img2img_app/character_list.html'
    context_object_name = 'characters'
    paginate_by = 12


class CharacterCreateView(CreateView):
    """Create a new character with images."""
    model = Character
    form_class = CharacterForm
    template_name = 'img2img_app/character_form.html'
    success_url = reverse_lazy('img2img_app:character_list')

    def form_valid(self, form):
        response = super().form_valid(form)
        # Handle multiple image uploads
        images = self.request.FILES.getlist('images')
        for i, image_file in enumerate(images):
            CharacterImage.objects.create(
                character=self.object,
                image=image_file,
                original_filename=image_file.name,
                is_primary=(i == 0)  # First image is primary
            )
        return response


class CharacterDetailView(DetailView):
    """View character details and images."""
    model = Character
    template_name = 'img2img_app/character_detail.html'
    context_object_name = 'character'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['images'] = self.object.images.all()
        context['image_form'] = CharacterImageForm()
        return context


class CharacterUpdateView(UpdateView):
    """Update character details."""
    model = Character
    form_class = CharacterForm
    template_name = 'img2img_app/character_form.html'

    def get_success_url(self):
        return reverse_lazy('img2img_app:character_detail', kwargs={'pk': self.object.pk})

    def form_valid(self, form):
        response = super().form_valid(form)
        # Handle additional image uploads
        images = self.request.FILES.getlist('images')
        for image_file in images:
            CharacterImage.objects.create(
                character=self.object,
                image=image_file,
                original_filename=image_file.name
            )
        return response


class CharacterDeleteView(DeleteView):
    """Delete a character."""
    model = Character
    template_name = 'img2img_app/character_confirm_delete.html'
    success_url = reverse_lazy('img2img_app:character_list')


# Session Views
class SessionListView(ListView):
    """List all generation sessions."""
    model = GenerationSession
    template_name = 'img2img_app/session_list.html'
    context_object_name = 'sessions'
    paginate_by = 12


class SessionDetailView(DetailView):
    """View a generation session with all iterations."""
    model = GenerationSession
    template_name = 'img2img_app/session_detail.html'
    context_object_name = 'session'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['iterations'] = self.object.iterations.order_by('iteration_number')
        context['form'] = GenerationForm(initial={'session_id': self.object.id})
        context['active_config'] = ModelConfig.objects.filter(is_active=True).first()
        return context


class SessionDeleteView(DeleteView):
    """Delete a generation session."""
    model = GenerationSession
    template_name = 'img2img_app/session_confirm_delete.html'
    success_url = reverse_lazy('img2img_app:session_list')


# API Endpoints

@require_http_methods(["POST"])
def generate_image(request):
    """Generate an image based on prompt and optional input image."""
    try:
        prompt = request.POST.get('prompt', '').strip()
        if not prompt:
            return JsonResponse({'error': 'Prompt is required'}, status=400)

        session_id = request.POST.get('session_id')
        use_previous = request.POST.get('use_previous_output') == 'on'
        uploaded_image = request.FILES.get('base_image')
        character_id = request.POST.get('character')

        # Get character if specified
        character = None
        character_images = []
        if character_id:
            try:
                character = Character.objects.get(id=character_id)
                character_images = character.all_image_paths
            except Character.DoesNotExist:
                pass

        # Get or create session
        session = None
        input_image = None
        input_pil_image = None

        if session_id:
            try:
                session = GenerationSession.objects.get(id=session_id)
                if use_previous and session.latest_iteration and session.latest_iteration.output_image:
                    input_image = session.latest_iteration.output_image
                    input_pil_image = Image.open(input_image.path)
                elif session.base_image:
                    input_image = session.base_image
                    input_pil_image = Image.open(session.base_image.path)
            except GenerationSession.DoesNotExist:
                pass

        if uploaded_image:
            input_pil_image = Image.open(uploaded_image)
            if not session:
                session = GenerationSession.objects.create(
                    name=f"Session {timezone.now().strftime('%Y-%m-%d %H:%M')}",
                    character=character
                )
                session.base_image.save(uploaded_image.name, uploaded_image)

        if not session:
            session = GenerationSession.objects.create(
                name=f"Session {timezone.now().strftime('%Y-%m-%d %H:%M')}",
                character=character
            )

        # Update session character if changed
        if character and session.character != character:
            session.character = character
            session.save()

        # Get active config
        config = ModelConfig.objects.filter(is_active=True).first()
        config_data = {}

        if config:
            model_id = config.model_id
            strength = config.strength
            guidance_scale = config.guidance_scale
            num_inference_steps = config.num_inference_steps
            negative_prompt = config.negative_prompt
            use_refiner = config.use_refiner
            config_data = {
                'model_id': model_id,
                'strength': strength,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_inference_steps,
                'negative_prompt': negative_prompt,
                'use_refiner': use_refiner,
            }
        else:
            from django.conf import settings
            model_id = settings.DEFAULT_MODEL_ID
            strength = 0.75
            guidance_scale = 7.5
            num_inference_steps = 50
            negative_prompt = 'blurry, low quality, distorted'
            use_refiner = False
            config_data = {
                'model_id': model_id,
                'strength': strength,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_inference_steps,
                'negative_prompt': negative_prompt,
                'use_refiner': use_refiner,
            }

        # Add character info to config snapshot
        if character:
            config_data['character'] = {
                'id': str(character.id),
                'name': character.name,
                'image_count': len(character_images)
            }

        # Enhance prompt with character description if available
        enhanced_prompt = prompt
        if character:
            if character.trigger_word:
                enhanced_prompt = f"{character.trigger_word}, {prompt}"
            if character.description:
                enhanced_prompt = f"{enhanced_prompt}. Character: {character.description}"

        # Generate image
        output_pil, gen_time = engine.generate(
            prompt=enhanced_prompt,
            image=input_pil_image,
            model_id=model_id,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            character_images=character_images if character_images else None,
            use_refiner=use_refiner,
        )

        # Save iteration
        iteration = GenerationIteration(
            session=session,
            prompt=prompt,
            model_config=config,
            character=character,
            config_snapshot=config_data,
            generation_time=gen_time,
        )

        # Save input image if new
        if uploaded_image and not input_image:
            iteration.input_image.save(
                f"input_{uuid.uuid4().hex[:8]}.png",
                uploaded_image
            )

        # Save output image
        output_buffer = BytesIO()
        output_pil.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        iteration.output_image.save(
            f"output_{uuid.uuid4().hex[:8]}.png",
            ContentFile(output_buffer.read())
        )
        iteration.save()

        return JsonResponse({
            'success': True,
            'session_id': str(session.id),
            'iteration_id': str(iteration.id),
            'iteration_number': iteration.iteration_number,
            'output_url': iteration.output_image.url,
            'generation_time': round(gen_time, 2),
            'character_used': character.name if character else None,
        })

    except ModelNotFoundError as e:
        return JsonResponse({
            'error': 'Model not downloaded',
            'message': str(e),
            'type': 'model_not_found'
        }, status=503)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def engine_status(request):
    """Get current engine status."""
    return JsonResponse(engine.status())


@require_http_methods(["POST"])
def unload_models(request):
    """Unload all models from memory."""
    try:
        engine.unload_all_models()
        return JsonResponse({
            'success': True,
            'message': 'All models unloaded',
            'memory': engine.get_memory_usage()
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["POST"])
def cleanup_memory(request):
    """Run garbage collection and clear GPU cache."""
    try:
        engine.cleanup_resources()
        return JsonResponse({
            'success': True,
            'message': 'Memory cleanup completed',
            'memory': engine.get_memory_usage()
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def memory_status(request):
    """Get detailed memory status."""
    try:
        return JsonResponse(engine.get_memory_usage())
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["POST"])
def set_active_config(request, pk):
    """Set a configuration as active."""
    config = get_object_or_404(ModelConfig, pk=pk)

    # Check if model is downloaded
    available_models = engine.get_available_models()
    model_downloaded = any(
        m['id'] == config.model_id and m['downloaded']
        for m in available_models
    )

    if not model_downloaded:
        return JsonResponse({
            'success': False,
            'error': 'Cannot activate config - model not downloaded',
            'model_id': config.model_id
        }, status=400)

    config.is_active = True
    config.save()
    return JsonResponse({'success': True, 'config_id': str(config.id)})


@require_http_methods(["POST"])
def clear_sessions(request):
    """Clear all generation sessions."""
    GenerationSession.objects.all().delete()
    return JsonResponse({'success': True})


@require_http_methods(["GET"])
def get_session_data(request, pk):
    """Get session data as JSON."""
    session = get_object_or_404(GenerationSession, pk=pk)
    iterations = []

    for it in session.iterations.order_by('iteration_number'):
        iterations.append({
            'id': str(it.id),
            'number': it.iteration_number,
            'prompt': it.prompt,
            'input_url': it.input_image.url if it.input_image else None,
            'output_url': it.output_image.url if it.output_image else None,
            'generation_time': it.generation_time,
            'config': it.config_snapshot,
            'character': it.character.name if it.character else None,
            'created_at': it.created_at.isoformat(),
        })

    return JsonResponse({
        'id': str(session.id),
        'name': session.name,
        'base_image_url': session.base_image.url if session.base_image else None,
        'character': session.character.name if session.character else None,
        'iterations': iterations,
        'created_at': session.created_at.isoformat(),
    })


# Character API endpoints

@require_http_methods(["POST"])
def add_character_image(request, pk):
    """Add an image to a character."""
    character = get_object_or_404(Character, pk=pk)
    image_file = request.FILES.get('image')

    if not image_file:
        return JsonResponse({'error': 'No image provided'}, status=400)

    caption = request.POST.get('caption', '')
    is_primary = request.POST.get('is_primary') == 'on'

    char_image = CharacterImage.objects.create(
        character=character,
        image=image_file,
        original_filename=image_file.name,
        caption=caption,
        is_primary=is_primary
    )

    return JsonResponse({
        'success': True,
        'image_id': str(char_image.id),
        'image_url': char_image.image.url
    })


@require_http_methods(["POST"])
def delete_character_image(request, pk):
    """Delete a character image."""
    char_image = get_object_or_404(CharacterImage, pk=pk)
    character_id = char_image.character.id
    char_image.delete()
    return JsonResponse({'success': True, 'character_id': str(character_id)})


@require_http_methods(["POST"])
def set_primary_image(request, pk):
    """Set an image as primary for its character."""
    char_image = get_object_or_404(CharacterImage, pk=pk)
    char_image.is_primary = True
    char_image.save()
    return JsonResponse({'success': True})


@require_http_methods(["GET"])
def get_character_data(request, pk):
    """Get character data as JSON."""
    character = get_object_or_404(Character, pk=pk)
    images = []

    for img in character.images.all():
        images.append({
            'id': str(img.id),
            'url': img.image.url,
            'caption': img.caption,
            'is_primary': img.is_primary,
        })

    return JsonResponse({
        'id': str(character.id),
        'name': character.name,
        'description': character.description,
        'trigger_word': character.trigger_word,
        'images': images,
        'image_count': len(images),
    })


# Enhancement views
@require_http_methods(["POST"])
def start_character_enhancement(request, pk):
    """Start enhancement generation for a character."""
    from .services.enhancement_task import start_enhancement_job

    character = get_object_or_404(Character, pk=pk)

    if character.images.count() == 0:
        return JsonResponse({
            'error': 'Character has no reference images'
        }, status=400)

    # Get active model config
    config = ModelConfig.objects.filter(is_active=True).first()
    model_id = config.model_id if config else None

    try:
        job = start_enhancement_job(str(character.id), model_id)
        return JsonResponse({
            'success': True,
            'job_id': str(job.id),
            'status': job.status,
            'total_variations': job.total_variations,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_enhancement_status(request, pk):
    """Get status of an enhancement job."""
    from .services.enhancement_task import get_job_status

    status = get_job_status(str(pk))
    return JsonResponse(status)


@require_http_methods(["GET"])
def get_enhanced_images(request, pk):
    """Get all enhanced images for a character."""
    character = get_object_or_404(Character, pk=pk)

    # Get latest completed job
    job = EnhancementJob.objects.filter(
        character=character,
        status='completed'
    ).first()

    if not job:
        # Check for running job
        running_job = EnhancementJob.objects.filter(
            character=character,
            status__in=['pending', 'processing']
        ).first()
        return JsonResponse({
            'enhanced_images': [],
            'has_enhancements': False,
            'job_running': running_job is not None,
            'running_job_id': str(running_job.id) if running_job else None,
        })

    images = []
    for enhanced in job.enhanced_images.all():
        images.append({
            'id': str(enhanced.id),
            'source_image_id': str(enhanced.source_image.id),
            'variation_type': enhanced.variation_type,
            'variation_label': enhanced.get_variation_type_display(),
            'url': enhanced.image.url,
            'generation_time': enhanced.generation_time,
        })

    return JsonResponse({
        'enhanced_images': images,
        'has_enhancements': True,
        'job_id': str(job.id),
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
    })
