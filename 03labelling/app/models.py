from django.db import models

class Title(models.Model):
    keyword = models.TextField()
    title = models.TextField()
    translated_title = models.TextField()

class LabeledTitle(models.Model):
    title = models.ForeignKey(Title, on_delete=models.CASCADE)
    oxide = models.CharField(max_length=50)
    is_nanoparticle = models.BooleanField(default=True)