from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('test/', views.prediction, name='prediction'),
    path('about/', views.about, name='about'),
    path('upload/', views.upload_and_predict, name='upload_and_predict'),
   
]
