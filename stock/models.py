from django.db import models


# Create your models here.

class User(models.Model):
    name = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.name + " , " + self.password


class Stock(models.Model):
    date = models.DateTimeField()
    data = models.FloatField()

    def __str__(self):
        return str(self.data)



