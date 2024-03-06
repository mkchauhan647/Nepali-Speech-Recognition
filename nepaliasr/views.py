from django.shortcuts import render
import librosa
from pydub import AudioSegment
from .models import predict_text  # assuming you have a prediction function
from jiwer import wer
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os
import json
from django.http import JsonResponse, HttpResponse
from django.conf import settings

def index(request):
    
 return render(request, 'index.html')

@csrf_exempt
# def process_audio(request):
#     if request.method == "POST":
#         try:
#             audio_file = request.FILES.get("audio")
#             original_text = request.POST.get("original_text")

#             if not audio_file or not original_text:
#                 return JsonResponse({"error": "Missing audio file or original text"}, status=400)

#             # Create a temporary file with manual deletion
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#                 for chunk in audio_file.chunks():
#                     tmp_file.write(chunk)
#                 temp_audio_path = tmp_file.name

#             # Ensure the file is closed before attempting to load it with LibROSA
#             # signal, sr = librosa.load(temp_audio_path, sr=16000)
#             librosa.

#             # Process the signal as needed...

#             predicted_text = predict_text(signal, sr)


#             # Delete the temporary file manually
#             os.remove(temp_audio_path)

#             return JsonResponse({
#                 "predicted_text": "your_predicted_text_here",
#                 "word_error_rate": "your_error_rate_here"
#             })

#         except Exception as e:
#             # Attempt to clean up the temporary file in case of error
#             if 'temp_audio_path' in locals():
#                 os.remove(temp_audio_path)
#             return JsonResponse({"error": str(e)}, status=500)

#     return JsonResponse({"error": "Request method not allowed"}, status=405)

def process_audio(request):
    if request.method == "POST":
        try:
            audio_file = request.FILES.get("audio")
            original_text = request.POST.get("original_text")
            modelname = request.POST.get("modelname")

            print("This is selected modelname ",modelname)

            if not audio_file or not original_text:
                return JsonResponse({"error": "Missing audio file or original text"}, status=400)

            # Process audio file
            print("This is ",audio_file)
            audio = AudioSegment.from_file(audio_file)
            audio = audio.set_channels(1)

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                temp_audio_path = tmp_file.name
                audio.export(temp_audio_path, format="wav")

                # Load with librosa
                signal, sr = librosa.load(temp_audio_path, sr=16000)

            # Model prediction
            os.remove(temp_audio_path)
            predicted_text = predict_text(signal, sr,modelname)

            error_rate = wer(original_text, predicted_text)

            return JsonResponse({
                "predicted_text": predicted_text,
                "word_error_rate": error_rate
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Request method not allowed"}, status=405)

@csrf_exempt
# def save_json(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         # with open(os.path.join(settings.STATIC_ROOT, 'data.json'), 'w') as f:
#         with open('data.json', 'a') as f:
#             json.dump(data, f)
#         return JsonResponse({"message": "JSON data saved successfully."})
#     else:
#         return HttpResponse(status=405)
# @csrf_exempt
def save_json(request):
    if request.method == 'POST':
        # Load existing data from data.json if it exists
        existing_data = []
        file_path = 'data.json'
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as existing_file:
                existing_data = json.load(existing_file)

        # Load new data from the POST request
        new_data = json.loads(request.body)

        # Append new data to the existing data array
        # existing_data.append(new_data)
        existing_data.insert(0,new_data)

        # Write the updated data array back to the file
        with open(file_path, 'w') as f:
            json.dump(existing_data, f)

        return JsonResponse({"message": "JSON data saved successfully."})
    else:
        return HttpResponse(status=405)
@csrf_exempt
def get_json(request):
    if request.method == 'GET':
        # with open(os.path.join(settings.STATIC_ROOT, 'data.json'), 'r') as f:
        with open('data.json', 'r') as f:
            data = json.load(f)
        return JsonResponse(data, safe=False)
    else:
        return HttpResponse(status=405)

def get_models(request):
    # model_files = ModelFile.objects.values_list('filename', flat=True)
    print("hello from get_models")
    with open('modelsname.json','r') as f:
      model_files = json.load(f)
      print("Model name",model_files)
      

    return JsonResponse({"models":list(model_files)})