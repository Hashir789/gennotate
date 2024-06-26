# Generated by Django 5.0.4 on 2024-05-01 08:53

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gennotate', '0007_delete_testmodel'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RemoveField(
            model_name='generatedimage',
            name='addToGallery',
        ),
        migrations.AddField(
            model_name='generatedimage',
            name='generated',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='segmentedimage',
            name='type',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='segmentedimage',
            name='userId',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='generatedimage',
            name='type',
            field=models.IntegerField(default=0),
        ),
    ]
