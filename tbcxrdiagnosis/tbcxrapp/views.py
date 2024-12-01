from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from .models import predict
import os


def prediction(request):
    return render(request, 'prediction.html')

def welcome(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def upload_and_predict(request):
    if request.method == 'POST' and request.FILES['image']:
        # Save the uploaded image
        uploaded_image = request.FILES['image']
        
        image_path =  os.path.join('media/uploads', uploaded_image.name)
        os.makedirs(os.path.dirname(os.path.join(settings.MEDIA_ROOT, image_path)), exist_ok=True)
        with open(os.path.join(settings.MEDIA_ROOT, image_path), 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Predict using the model
        prediction = predict(image_path)

        # Return the prediction result
        imgresult = predict(os.path.join(settings.MEDIA_ROOT, image_path))
        result = 'Tuberculosis' if prediction == 1 else 'Normal'
        return render(request, 'prediction.html', {'result': result,
                                              'image_path': image_path,
                                              'imgresult': imgresult,
                'image_path': image_path,  # Relative path to media/
                'MEDIA_URL': settings.MEDIA_URL, })

    return JsonResponse({'error': 'No image uploaded'})
