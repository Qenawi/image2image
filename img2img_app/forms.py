from django import forms
from django.forms import ClearableFileInput
from .models import ModelConfig, GenerationSession, Character, CharacterImage


class MultipleFileInput(ClearableFileInput):
    """Custom widget for multiple file upload."""
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    """Custom field for multiple file upload."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result


class ModelConfigForm(forms.ModelForm):
    """Form for creating/editing model configurations."""

    def __init__(self, *args, available_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._available_models = available_models or []

        # Build dynamic choices from available models
        if available_models:
            choices = []
            for model in available_models:
                label = model['name']
                if not model['downloaded']:
                    label += ' (Not Downloaded)'
                choices.append((model['id'], label))
            self.fields['model_id'].widget.choices = choices

    def clean(self):
        cleaned_data = super().clean()
        model_id = cleaned_data.get('model_id')
        is_active = cleaned_data.get('is_active')

        # If setting as active, verify model is downloaded
        if is_active and model_id and self._available_models:
            model_info = next(
                (m for m in self._available_models if m['id'] == model_id),
                None
            )
            if model_info and not model_info.get('downloaded', False):
                self.add_error(
                    'is_active',
                    'Cannot set as active: model is not downloaded. '
                    'Download the model first or choose a different model.'
                )

        return cleaned_data

    class Meta:
        model = ModelConfig
        fields = [
            'name', 'model_id', 'strength', 'guidance_scale',
            'num_inference_steps', 'negative_prompt', 'is_active', 'use_refiner'
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Configuration name'
            }),
            'model_id': forms.Select(attrs={
                'class': 'form-select',
                'id': 'id_model_id'
            }),
            'strength': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 0.0,
                'max': 1.0,
                'step': 0.05
            }),
            'guidance_scale': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1.0,
                'max': 20.0,
                'step': 0.5
            }),
            'num_inference_steps': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 20,
                'max': 100,
                'step': 5
            }),
            'negative_prompt': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'What to avoid in generation...'
            }),
            'is_active': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'use_refiner': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }


class CharacterForm(forms.ModelForm):
    """Form for creating/editing characters."""
    images = MultipleFileField(
        required=False,
        widget=MultipleFileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/png,image/webp',
            'multiple': True
        }),
        help_text='Upload multiple reference images (3-10 recommended for best results)'
    )

    class Meta:
        model = Character
        fields = ['name', 'description', 'trigger_word']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Character name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe the character: appearance, style, distinctive features...'
            }),
            'trigger_word': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., MYCHAR (optional)'
            }),
        }


class CharacterImageForm(forms.ModelForm):
    """Form for adding images to a character."""

    class Meta:
        model = CharacterImage
        fields = ['image', 'caption', 'is_primary']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/jpeg,image/png,image/webp'
            }),
            'caption': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Optional caption for this image'
            }),
            'is_primary': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }


class GenerationForm(forms.Form):
    """Form for image generation."""
    prompt = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Describe the changes you want to apply...'
        }),
        max_length=2000,
        help_text='Describe what you want to create or modify'
    )
    base_image = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/png,image/webp'
        }),
        help_text='Upload a base image (optional for txt2img, required for img2img)'
    )
    character = forms.ModelChoiceField(
        queryset=Character.objects.all(),
        required=False,
        empty_label='-- No Character --',
        widget=forms.Select(attrs={
            'class': 'form-select'
        }),
        help_text='Select a character for consistent generation'
    )
    session_id = forms.UUIDField(
        required=False,
        widget=forms.HiddenInput()
    )
    use_previous_output = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        help_text='Use previous output as input for iterative refinement'
    )


class SessionForm(forms.ModelForm):
    """Form for creating a new generation session."""

    class Meta:
        model = GenerationSession
        fields = ['name', 'base_image', 'character']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Session name (optional)'
            }),
            'base_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/jpeg,image/png,image/webp'
            }),
            'character': forms.Select(attrs={
                'class': 'form-select'
            }),
        }
