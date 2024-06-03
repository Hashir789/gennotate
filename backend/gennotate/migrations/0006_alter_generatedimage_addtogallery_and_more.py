# Generated by Django 5.0.2 on 2024-02-25 05:31

import cloudinary.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gennotate', '0005_testmodel'),
    ]

    operations = [
        migrations.AlterField(
            model_name='generatedimage',
            name='addToGallery',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='generatedimage',
            name='annotations',
            field=models.TextField(default='', max_length=1000),
        ),
        migrations.AlterField(
            model_name='generatedimage',
            name='link',
            field=cloudinary.models.CloudinaryField(max_length=255, verbose_name='image'),
        ),
        migrations.AlterField(
            model_name='segmentedimage',
            name='annotations',
            field=models.TextField(default='', max_length=1000),
        ),
        migrations.AlterField(
            model_name='segmentedimage',
            name='link',
            field=cloudinary.models.CloudinaryField(max_length=255, verbose_name='image'),
        ),
        migrations.AlterField(
            model_name='testmodel',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]
