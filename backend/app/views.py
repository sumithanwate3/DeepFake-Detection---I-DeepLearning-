import os
from io import BytesIO

import cv2
import numpy as np
import requests
from django.conf import settings
from django.http import JsonResponse
from PIL import Image
from tensorflow.keras.models import load_model

model_path = os.path.join(settings.BASE_DIR, 'model', 'Best_Model2.keras')
best_model = load_model(model_path, safe_mode=False)


def extract_frames_from_video(video_content):
    temp_filename = 'temp_video.mp4'
    with open(temp_filename, 'wb') as f:
        f.write(video_content)

    cap = cv2.VideoCapture(temp_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % fps == 0:
            frames.append(frame)
        frame_num += 1

    cap.release()

    os.remove(temp_filename)

    return frames

def process_image(image):
    resized_image = image.resize((300, 300))
    normalized_image = np.array(resized_image) / 255.0
    processed_image = np.expand_dims(normalized_image, axis=0)

    prediction = best_model.predict(processed_image)
    print(prediction)
    result_label = "Fake" if prediction < 0.5 else "Real"
    return result_label

def process_video(video_frames):
    predictions = []
    for frame in video_frames:
        frame_copy = frame.copy()
        resized_frame = cv2.resize(frame_copy, (300, 300))  
        normalized_frame = np.array(resized_frame) / 255.0 
        processed_frame = np.expand_dims(normalized_frame, axis=0)  
 
        prediction = best_model.predict(processed_frame)
        predictions.append(prediction)
    avg_prediction = np.mean(predictions)
    result_label = "Fake" if avg_prediction < 0.5 else "Real"
    return result_label

def process_media(request):
    if request.method == 'GET':
        media_url = request.GET.get('url')
        print(media_url)
        response = requests.get(media_url)
        if response.status_code == 200:
            content_type = response.headers['content-type']
            if 'image' in content_type:
                image = Image.open(BytesIO(response.content))
                result_label = process_image(image)
                return JsonResponse({'result': result_label})
            elif 'video' in content_type:
                video_frames = extract_frames_from_video(response.content)
                if video_frames:
                    result_label = process_video(video_frames)
                    return JsonResponse({'result': result_label})
                else:
                    return JsonResponse({'error': 'Failed to extract frames from video'})
            else:
                return JsonResponse({'error': 'Unsupported media type'})
        else:
            return JsonResponse({'error': 'Failed to download media'})
    else:
        return JsonResponse({'error': 'Invalid request method'})