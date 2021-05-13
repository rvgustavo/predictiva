from django.urls import path
from ModelAssistant import settings
from django.conf.urls.static import static
from assistant import views

urlpatterns = [
    path('', views.home, name='home'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)