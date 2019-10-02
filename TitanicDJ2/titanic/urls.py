from django.urls import path, include
from django.views.generic.base import RedirectView
from rest_framework import routers
from titanic import views

router = routers.DefaultRouter()
router.register(r'titanic', views.TitanicViewSet)

urlpatterns = [
    path(r'api/', include(router.urls)),
    path('', RedirectView.as_view(url="api/")),
]

