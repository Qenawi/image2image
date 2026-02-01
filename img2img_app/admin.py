from django.contrib import admin
from .models import ModelConfig, GenerationSession, GenerationIteration, Character, CharacterImage


@admin.register(ModelConfig)
class ModelConfigAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_id', 'is_active', 'strength', 'guidance_scale', 'updated_at']
    list_filter = ['is_active', 'model_id']
    search_fields = ['name']
    ordering = ['-is_active', '-updated_at']


class CharacterImageInline(admin.TabularInline):
    model = CharacterImage
    extra = 1
    readonly_fields = ['uploaded_at']
    fields = ['image', 'caption', 'is_primary', 'uploaded_at']


@admin.register(Character)
class CharacterAdmin(admin.ModelAdmin):
    list_display = ['name', 'trigger_word', 'image_count', 'created_at', 'updated_at']
    search_fields = ['name', 'description', 'trigger_word']
    inlines = [CharacterImageInline]
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(CharacterImage)
class CharacterImageAdmin(admin.ModelAdmin):
    list_display = ['character', 'is_primary', 'caption', 'uploaded_at']
    list_filter = ['character', 'is_primary']
    search_fields = ['caption', 'character__name']


class GenerationIterationInline(admin.TabularInline):
    model = GenerationIteration
    extra = 0
    readonly_fields = ['iteration_number', 'generation_time', 'created_at']
    fields = ['iteration_number', 'prompt', 'character', 'input_image', 'output_image', 'generation_time', 'created_at']


@admin.register(GenerationSession)
class GenerationSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'id', 'character', 'iteration_count', 'created_at', 'updated_at']
    list_filter = ['character']
    search_fields = ['name']
    inlines = [GenerationIterationInline]
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(GenerationIteration)
class GenerationIterationAdmin(admin.ModelAdmin):
    list_display = ['session', 'iteration_number', 'character', 'prompt', 'generation_time', 'created_at']
    list_filter = ['session', 'model_config', 'character']
    search_fields = ['prompt']
    readonly_fields = ['id', 'iteration_number', 'config_snapshot', 'created_at']
