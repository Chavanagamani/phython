from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
def index(request):
    return HttpResponse("Hello, World")


def user_details(request, user_id):
    response = "User id : %s"
    return HttpResponse(response % user_id)
