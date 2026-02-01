from django.urls import path
from . import views

app_name = 'img2img_app'

urlpatterns = [
    # Main pages
    path('', views.IndexView.as_view(), name='index'),

    # Model configurations
    path('configs/', views.ConfigListView.as_view(), name='config_list'),
    path('configs/create/', views.ConfigCreateView.as_view(), name='config_create'),
    path('configs/<uuid:pk>/edit/', views.ConfigUpdateView.as_view(), name='config_update'),
    path('configs/<uuid:pk>/delete/', views.ConfigDeleteView.as_view(), name='config_delete'),
    path('configs/<uuid:pk>/activate/', views.set_active_config, name='config_activate'),

    # Characters
    path('characters/', views.CharacterListView.as_view(), name='character_list'),
    path('characters/create/', views.CharacterCreateView.as_view(), name='character_create'),
    path('characters/<uuid:pk>/', views.CharacterDetailView.as_view(), name='character_detail'),
    path('characters/<uuid:pk>/edit/', views.CharacterUpdateView.as_view(), name='character_update'),
    path('characters/<uuid:pk>/delete/', views.CharacterDeleteView.as_view(), name='character_delete'),
    path('characters/<uuid:pk>/data/', views.get_character_data, name='character_data'),
    path('characters/<uuid:pk>/add-image/', views.add_character_image, name='add_character_image'),
    path('characters/images/<uuid:pk>/delete/', views.delete_character_image, name='delete_character_image'),
    path('characters/images/<uuid:pk>/set-primary/', views.set_primary_image, name='set_primary_image'),

    # Character Enhancement
    path('characters/<uuid:pk>/enhance/', views.start_character_enhancement, name='start_enhancement'),
    path('characters/<uuid:pk>/enhanced-images/', views.get_enhanced_images, name='enhanced_images'),
    path('enhancement-jobs/<uuid:pk>/status/', views.get_enhancement_status, name='enhancement_status'),

    # Sessions
    path('sessions/', views.SessionListView.as_view(), name='session_list'),
    path('sessions/<uuid:pk>/', views.SessionDetailView.as_view(), name='session_detail'),
    path('sessions/<uuid:pk>/delete/', views.SessionDeleteView.as_view(), name='session_delete'),
    path('sessions/<uuid:pk>/data/', views.get_session_data, name='session_data'),
    path('sessions/clear/', views.clear_sessions, name='clear_sessions'),

    # API
    path('api/generate/', views.generate_image, name='generate'),
    path('api/status/', views.engine_status, name='status'),
    path('api/memory/', views.memory_status, name='memory_status'),
    path('api/memory/cleanup/', views.cleanup_memory, name='cleanup_memory'),
    path('api/models/unload/', views.unload_models, name='unload_models'),
]
