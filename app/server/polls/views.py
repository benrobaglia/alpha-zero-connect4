from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render


def index(request):
    template = loader.get_template('polls/index.html')
    context = {

    }
    return HttpResponse(template.render(context, request))


def p4(request):
    return render(request, 'polls/index.html')
