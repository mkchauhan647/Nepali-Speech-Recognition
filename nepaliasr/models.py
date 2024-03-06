import tensorflow as tf 

import librosa
import json
import numpy as np
import requests
import os
def CTCLoss(y_true,y_pred):
  # print("This is y_true:",y_true)
  # print("This is y_pred:",y_pred)
  batch_len = tf.cast(tf.shape(y_true)[0],dtype='int64')
  input_length = tf.cast(tf.shape(y_pred)[1],dtype='int64')
  label_length = tf.cast(tf.shape(y_true)[1],dtype='int64')
  input_length = input_length * tf.ones(shape=(batch_len,1),dtype='int64')
  label_length = label_length * tf.ones(shape=(batch_len,1),dtype='int64')

  # print(f"batch_len{batch_len}, input_length {input_length}, label_length {label_length}")
  loss = tf.keras.backend.ctc_batch_cost(y_true,y_pred,input_length,label_length)
  return loss

with open('vocab.json','r',encoding='utf-8') as f:
        vocabulary = json.load(f)
        # print(vocabulary)
        characters = list(vocabulary.keys())
        characters.extend([' ', ',', ':', '-', '_'])
        print(characters)
#String to Number Mapping using tensorflow LookupString method
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters,oov_token='')
num_to_char = tf.keras.layers.StringLookup(vocabulary= char_to_num.get_vocabulary(),oov_token='',invert=True)

#Audio Framing
frame_length = 256
frame_step = 160
fft_length = 384
target_sample_rate = 16000


def decode_batch_predictions(pred):
  input_len = np.ones(pred.shape[0])*pred.shape[1]
  results = tf.keras.backend.ctc_decode(pred,input_length = input_len , greedy = True)[0][0]

  print("This is resultssss :",results)

  output_text = []
  for result in results:
    result = tf.strings.reduce_join(num_to_char(result)).numpy().decode('utf-8')
    print("This is result:",result)
    output_text.append(result)
  return output_text

def encode_audio(audio,sample_rate,label='dummy'):

  # audio, sample_rate = librosa.load(audio_path, sr=None)

  # Resampling audio while Testing in live environment
  if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)

    # Convert audio array back to TensorFlow tensor
  audio = tf.convert_to_tensor(audio, dtype=tf.float32)

  #  Resampling audio while Testing in live environment
  print(audio.shape)
  audio = tf.reshape(audio,[-1,1])
  print(audio.shape)
  audio = tf.squeeze(audio, axis=-1)
  print("This is sample_rate",sample_rate)

  # computing spectrogram
  spectrogram = tf.signal.stft(audio,frame_length,frame_step,fft_length) # // getting the spectrogram
  spectrogram = tf.abs(spectrogram)  # spectrogram Magnitude as this is in x + ij format.
  spectrogram = tf.math.pow(spectrogram, 0.5) # scaling

  # Normalizing spectrogram
  means = tf.math.reduce_mean(spectrogram,axis=1,keepdims=True)
  std   = tf.math.reduce_std(spectrogram,axis=1,keepdims=True)
  spectrogram = (spectrogram-means)/(std + 1e-10)

  # Tokenizing and encoding labels
  label = tf.strings.unicode_split(label,input_encoding='UTF-8')
  label = char_to_num(label)
  label = tf.cast(label,tf.int32)

  return spectrogram,label



def predict_text(audio,sample_rate,modelname):
    print("Hello World from predict")
    print("version",tf.__version__)

    # file_name = '384-40k-v2-extended-pro.h5'
    # file_name = 'final_model.h5'
    # file_name = 'final_latest_v1.h5'

  # Check if the file exists in the current directory
    if os.path.exists(modelname):
      print(f"File '{modelname}' already exists. Using the existing file.")
    else:
    # If the file doesn't exist, proceed to download it
    # Define the URL of the file
      url = 'https://drive.usercontent.google.com/download?id=1-1ewihGdTYGkW18rStn2dA_mqLdAN0Su&export=download&confirm=t&uuid=0481515c-a1f0-4159-9e9c-3a2750f3c659'

    # Send a GET request to download the file
      response = requests.get(url)

    # Save the downloaded file locally
      with open(modelname, 'wb') as file:
        file.write(response.content)

      print(f"File '{modelname}' downloaded successfully.")


    # model = tf.keras.models.load_model('final_model.h5',custom_objects={"CTCLoss":CTCLoss})
    model = tf.keras.models.load_model(modelname,custom_objects={"CTCLoss":CTCLoss})
    # encode_audio(audio=audio,sample_rate=sample_rate)
    # preprocessed_audio, label= encode_audio('record (2).wav')
    preprocessed_audio, label= encode_audio(audio,sample_rate)
    preprocessed_audio = tf.expand_dims(preprocessed_audio, axis=0)
    
    print(preprocessed_audio.shape)
    
    # Making predictions
    predictions = model.predict(preprocessed_audio)
    decoded_prediction = decode_batch_predictions(predictions)
    label = tf.strings.reduce_join(num_to_char(label)).numpy().decode('utf-8')

    # wer_rate = wer(label,decoded_prediction)

    # Display or use the predictions
    print("Label: ",label)
    print("Predictions: ", decoded_prediction)
    return decoded_prediction

# predict_text()
