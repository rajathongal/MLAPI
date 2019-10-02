from django.db import models


# Create your models here.
class Titanic(models.Model):
    Pclass = models.IntegerField(default=0)
    Sex = models.CharField(max_length=10)
    Age = models.IntegerField()
    SibSp = models.IntegerField()
    Parch = models.IntegerField()
    Embarked = models.CharField(max_length=10, null=True, blank=True)
    Survived = models.IntegerField(null=True, blank=True)

    def to_dict(self):
        return {
            'Survived': self.Survived,
            'Pclass': self.Pclass,
            'Sex': self.Sex,
            'Age': self.Age,
            'SibSp': self.SibSp,
            'Parch': self.Parch,
            'Embarked': self.Embarked
        }
