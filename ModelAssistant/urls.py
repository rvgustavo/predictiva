from django.urls import path
from ModelAssistant import settings
from django.conf.urls.static import static
from assistant import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home', views.home, name='home'),
    path('assistant/<int:step>/', views.step_assistant, name='assisntant'),    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)