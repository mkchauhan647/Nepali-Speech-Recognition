from django.shortcuts import render
import librosa
from pydub import AudioSegment
from django.http import JsonResponse
from .models import predict_text  # assuming you have a prediction function
from jiwer import wer
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os


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
            predicted_text = predict_text(signal, sr)

            error_rate = wer(original_text, predicted_text)

            return JsonResponse({
                "predicted_text": predicted_text,
                "word_error_rate": error_rate
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Request method not allowed"}, status=405)
