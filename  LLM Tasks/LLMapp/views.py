from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


task = pipeline(task='text-classification', model='papluca/xlm-roberta-base-language-detection', device=0)

@csrf_exempt
def detect_language_view(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        text = data.get('text', '')

        if text:
            output = task(text)
            detected_language = output[0]['label']
            return JsonResponse({'detected_language': detected_language})

    return JsonResponse({'detected_language': ''})

task1 = pipeline(task='sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment', device=0)

@csrf_exempt
def analyze_sentiment_view(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        text = data.get('text', '')

        if text:
            output1 = task1(text)
            sentiment = output1[0]['label']
            return JsonResponse({'sentiment': sentiment})

    return JsonResponse({'sentiment': ''})

task2 = pipeline(task='ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', device=0)

@csrf_exempt
def analyze_entities_view(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        text = data.get('text', '')

        if text:
            output2 = task2(text)
            entities = [entity['word'] for entity in output2]
            return JsonResponse({'entities': entities})

    return JsonResponse({'entities': []})

translation_pipeline = pipeline(task='translation_en_to_de', model='Helsinki-NLP/opus-mt-en-de', device=0)

@csrf_exempt
def translate_view(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        text = data.get('text', '')

        if text:
            output3 = translation_pipeline(text)
            translation = output3[0]['translation_text']
            return JsonResponse({'translation': translation})

    return JsonResponse({'translation': ''})

text_generation_pipeline = pipeline(task='text-generation', model='gpt2', device=0, truncation=True)

@csrf_exempt
def generate_text_view(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        text = data.get('text', '')

        if text:
            output4 = text_generation_pipeline(text, max_length=50, num_return_sequences=1)
            generated_text = output4[0]['generated_text']
            return JsonResponse({'generated_text': generated_text})

    return JsonResponse({'generated_text': ''})

classifier = pipeline(task="image-classification")

@csrf_exempt
def classify_image(request):
    if request.method == 'POST' and request.FILES.get('image-upload'):
        image_file = request.FILES['image-upload']
        try:
            # Convert the uploaded image to a PIL image
            image = Image.open(io.BytesIO(image_file.read()))

            # Perform classification using the Hugging Face pipeline
            results = classifier(image)
            label = results[0]['label']

            # Return the classification label as a JSON response
            return JsonResponse({'label': label})

        except Exception as e:
            # Handle errors in processing the image
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def image_cap(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image-upload')
        if not image_file:
            return JsonResponse({'error': 'No image file provided'}, status=400)
        
        try:
            image = Image.open(image_file).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            return JsonResponse({'label': caption})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def main(request):
    return render(request,'main.html')
def nlp(request):
    return render(request,'NLP.html')
def images(request):
    return render(request,'Images.html')
def ic(request):
    return render(request,'IC.html')
def iti(request):
    return render(request,'ITI.html')
def ise(request):
    return render(request,'IS.html')
def tti(request):
    return render(request,'TTI.html')
def od(request):
    return render(request,'OD.html')
def tc(request):
    return render(request,'TC.html')
def sa(request):
    return render(request,'SA.html')
def ner(request):
    return render(request,'NER.html')
def lt(request):
    return render(request,'LT.html')
def tg(request):
    return render(request,'TG.html')



