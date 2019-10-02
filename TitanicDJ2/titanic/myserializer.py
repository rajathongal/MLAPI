from titanic.models import Titanic
from rest_framework import serializers


class TitanicSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Titanic
        fields = "__all__"
