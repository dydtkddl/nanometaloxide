from django.urls import path
from . import views

urlpatterns = [
    path('load_data/', views.load_data, name='load_data'),
    path('label_titles/', views.label_titles, name='label_titles'),
    path('get_labeling_info/', views.get_labeling_info, name='get_labeling_info'),
]