from django.shortcuts import render
from django.http import HttpResponse
from transformers import pipeline

# Create your views here.

stance_pipeline = pipeline("text-classification", model="nharrel/Valuesnet_DeBERTa_v3")

def index(request):
    return render(request, "index.html")

def stance(request):
    return render(request, "Stance.html")

def predict_stance(post, comment):
    input_text = f"Post: {post}\nComment: {comment}"
    result = stance_pipeline(input_text)
    return result[0]['label']

def stance(request):
    results = None
    post = ''
    comment = ''

    if request.method == "POST":
        post = request.POST.get("post")
        comment = request.POST.get("comment")

        results = {
            "Your Model": predict_stance(post, comment),
        }

    return render(request, "stance.html", {"results": results})