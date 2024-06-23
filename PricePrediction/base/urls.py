
from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('laptop/', laptop, name='laptop'),
    path('mobile/', mobile, name='mobile'),
    path('result/', laptop, name='result'),
    path('result_mobile/', mobile, name='result_mobile'),
]
