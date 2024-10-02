from django import forms
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import save_model
model = VGG16(weights='imagenet')
save_model(model, 'vgg16.h5')

class ImageUploadForm(forms.Form):
    image = forms.ImageField()
