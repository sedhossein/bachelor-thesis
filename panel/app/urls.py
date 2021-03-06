# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from app import views

urlpatterns = [
    # The home page
    path('', views.index, name='home'),

    path('<slug:username>/bio', views.bio, name='bio'),
    path('<slug:username>/wcloud', views.wcloud, name='wcloud'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),
]
