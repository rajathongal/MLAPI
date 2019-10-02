from django.shortcuts import render
from django.http.response import HttpResponse
from rest_framework import viewsets
from rest_framework.response import Response
from titanic import TiML
from titanic.models import Titanic
from titanic.myserializer import TitanicSerializer


# Create your views here.

class TitanicViewSet(viewsets.ModelViewSet):
    queryset = Titanic.objects.order_by("-id")
    serializer_class = TitanicSerializer

    def create(self, request, *args, **kwargs):
        viewsets.ModelViewSet.create(self, request, *args, **kwargs)
        ob = Titanic.objects.latest("id")
        sur = TiML.pred(ob)
        return Response({"status": "Sucess", "Survived": sur, 'tmp': args})
