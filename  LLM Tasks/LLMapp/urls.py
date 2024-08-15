from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name = 'main'),
    path('NLP',views.nlp, name = 'NLP'),
    path('Image',views.images, name = 'Image'),
    path('detect-language/', views.detect_language_view, name='detect_language'),
    path('analyze-sentiment/', views.analyze_sentiment_view, name='analyze_sentiment'),
    path('analyze-entities/', views.analyze_entities_view, name='analyze_entities'),
    path('translate/', views.translate_view, name='translate'),
    path('generate/', views.generate_text_view, name='generate'),
    path('classify/', views.classify_image, name='classify_image'),
    path('classify/', views.image_cap, name='classify_image'),
    path('IC',views.ic, name = 'IC'),
    path('ITI',views.iti, name = 'ITI'),
    path('TC',views.tc, name = 'TC'),
    path('SA',views.sa, name = 'SA'),
    path('NER',views.ner, name = 'NER'),
    path('LT',views.lt, name = 'LT'),
    path('TG',views.tg, name = 'TG'),
]
