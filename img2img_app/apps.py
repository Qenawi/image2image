from django.apps import AppConfig


class Img2ImgAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'img2img_app'
    verbose_name = 'Image to Image Generator'

    def ready(self):
        import img2img_app.signals  # noqa
