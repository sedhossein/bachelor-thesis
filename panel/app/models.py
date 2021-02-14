# -*- encoding: utf-8 -*-
from django.db import models


class Tweet(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=200)
    short_description = models.TextField()
    predict = models.CharField(max_length=200)

    class Meta:
        db_table = "tweets"

    def __str__(self):
        return self.name


class User(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=200)
    followers = models.IntegerField()
    followings = models.IntegerField()
    favorites = models.IntegerField()
    tweets_count = models.IntegerField()
    profile_pic = models.CharField(max_length=200)
    cover_pic = models.CharField(max_length=200)
    wcloud_pic = models.CharField(max_length=200)
    name = models.CharField(max_length=200)
    bio = models.CharField(max_length=200)
    location = models.CharField(max_length=200)
    website = models.CharField(max_length=200)
    join_at = models.CharField(max_length=200)

    class Meta:
        db_table = "users"

    def __str__(self):
        return self.username

