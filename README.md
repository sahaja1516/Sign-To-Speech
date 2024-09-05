!pip install gtts
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
# Define dataset directory and image dimensions
dataset_directory = '/content/drive/MyDrive/Indian/1/Reverse sign images'
img_height, img_width = 224, 224
# Define batch size
batch_size = 32
# Define number of classes
num_classes = 36  # Assuming you have 10 classes (0-9 for digits)
# Define ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    dataset_directory,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
# Load pre-trained MobileNetV2 model without top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
# Add new classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
# Combine base model and new classification head
model = Model(inputs=base_model.input, outputs=predictions)
# Freeze layers of base model
for layer in base_model.layers:
    layer.trainable = False
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fine-tune the model
# Fine-tune the model
epochs = 20
model.fit(train_generator, epochs=epochs)
# Save the fine-tuned model in the native Keras format
model.save('fine_tuned_model.h5')
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from gtts import gTTS
import IPython.display as ipd
import numpy as np
# Load the fine-tuned model
model = tf.keras.models.load_model('/content/fine_tuned_model.h5')
# Function to predict the class of an image and generate text/audio output
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    # Map predicted class label to text
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 
   '9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']  
   # Update with your class  labels
   predicted_text = class_labels[predicted_class]
   # Generate audio output
   tts = gTTS(text=predicted_text, lang='en')
    tts.save('output.mp3')
   # Display the predicted text and play audio
   print("Predicted Label:", predicted_text)
    print("Audio:")
    ipd.display(ipd.Audio('output.mp3'))
    image_path = '/content/drive/MyDrive/Indian/1/Reverse sign images/t/t.jpg'
    predict_image(image_path)
    from google.colab import drive
   drive.mount('/content/drive')
