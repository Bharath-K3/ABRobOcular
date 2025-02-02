import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from train import ArcFace
from dataset_loader import load_config

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess a single image for prediction."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_single_image(image_path):
    """Make prediction for a single image."""
    config = load_config()
    
    # Load model
    model = load_model(
        os.path.join(config['paths']['model_save_dir'], config['paths']['model_name']),
        custom_objects={'ArcFace': ArcFace}
    )
    
    # Preprocess image
    img_array = preprocess_image(
        image_path,
        target_size=tuple(config['data']['target_size'])
    )
    
    # Create dummy label (required for ArcFace)
    dummy_label = np.zeros((1, config['model']['n_classes']))
    
    # Make prediction
    predictions = model.predict([img_array, dummy_label])
    predicted_class = np.argmax(predictions[0])
    
    # Get class labels from test generator
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        config['data']['test_dir'],
        target_size=tuple(config['data']['target_size']),
        batch_size=1,
        class_mode=config['data']['class_mode']
    )
    
    class_labels = list(test_generator.class_indices.keys())
    predicted_class_label = class_labels[predicted_class]
    
    return predicted_class_label, predictions[0][predicted_class]

if __name__ == "__main__":
    # Example usage
    image_path = 'ocular_recognition/test/1141/1141_l_1.png'
    label, confidence = predict_single_image(image_path)
    print(f'Predicted class: {label}')
    print(f'Confidence: {confidence:.4f}')