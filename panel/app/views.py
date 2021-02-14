# -*- encoding: utf-8 -*-
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
from app.models import Tweet, User
from django.db.models import Count


@login_required(login_url="/login/")
def index(request):
    context = {'users': User.objects.all()}
    html_template = loader.get_template("index.html")

    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    try:
        load_template = request.path.split('/')[-1]
        html_template = loader.get_template(load_template)

        return HttpResponse(html_template.render(context, request))
    except template.TemplateDoesNotExist:
        html_template = loader.get_template('page-404.html')

        return HttpResponse(html_template.render(context, request))
    except:
        html_template = loader.get_template('page-500.html')

        return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def bio(request, username):
    tweets = Tweet.objects.filter(username=username).values('predict').annotate(total=Count('id'))
    user = User.objects.get(username=username)
    html_template = loader.get_template("profile.html")

    tweets_topic = {}
    total = 0
    for p in tweets:
        tweets_topic[p['predict']] = p['total']
        total += p['total']

    for name in tweets_topic:
        tweets_topic[name] = int(tweets_topic[name] * 100 / total)

    context = {
        'user': user,
        'total': total,
        'tweets_topic': tweets_topic,
        'tweets': tweets,
    }

    return HttpResponse(html_template.render(context, request))


def wcloud(request, username):
    tweets = Tweet.objects.filter(username=username).all()

    txt = ""
    for tweet in tweets:
        txt += " " + tweet.short_description

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wordcloud = WordCloud(repeat=False,
                          collocations=False,
                          margin=3,
                          width=1600,
                          height=800,
                          contour_color="black",
                          background_color="white",
                          relative_scaling=False,
                          ).generate(txt)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("core/static/wordcloud/" + username + "-word-cloud.png")

    path = "/static/wordcloud/" + username + "-word-cloud.png"

    User.objects.filter(username=username).update(wcloud_pic=path)

    return HttpResponse(txt)
