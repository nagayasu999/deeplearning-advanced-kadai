import base64
from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path
from io import BytesIO
import numpy as np

# モデルを事前にロード
model = VGG16(weights='imagenet')

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']

            # InMemoryUploadedFileをBytesIOに変換
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))  # 画像を読み込み
            img_array = img_to_array(img)                       # 画像を配列に変換
            img_array = np.expand_dims(img_array, axis=0)      # バッチ次元を追加
            img_array = preprocess_input(img_array)             # 前処理
            
             # 予測を行う
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=5)[0]  # 上位5件
            
            # 画像をBase64にエンコード
            img_file.seek(0)  # BytesIOのポインタを先頭に戻す
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')  # Base64エン

            # カテゴリと確率を取得
            predictions_with_percentage = [(pred[1], f"{pred[2] * 100:.2f}%") for pred in decoded_predictions]
            
            return render(request, 'home.html', {
                'form': form,
                'predictions_with_percentage': predictions_with_percentage,
                'img_data': img_base64,   # Base64エンコードされた画像データ
            })
        else:
            return render(request, 'home.html', {'form': form})
