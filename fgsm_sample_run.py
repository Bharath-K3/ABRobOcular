import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from dataset_loader import load_config, create_test_generator
from train import ArcFace
from AB_Rob_Ocular.AB_Rob_Ocular_Tensor import AdversarialEvaluationMetrics
from AB_Rob_Ocular import OcularAdversarialAttacks as OAA

def main():
    # Load configuration from config.yml
    config = load_config()

    # Extract directories and parameters from config
    test_dir = config['data']['test_dir']
    adv_dir = config['paths']['adversarial_dir']
    model_save_dir = config['paths']['model_save_dir']
    model_name = config['paths']['model_name']
    model_path = os.path.join(model_save_dir, model_name)
    target_size = tuple(config['data']['target_size'])
    n_classes = config['model']['n_classes']

    # Create the test data generator
    test_generator = create_test_generator()

    # Load the trained model.
    # Ensure that the custom_objects dictionary includes your ArcFace layer.
    model = load_model(model_path, custom_objects={'ArcFace': ArcFace})
    print(f"Loaded model from: {model_path}")

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # ----------------------------
    # Single Image Prediction Demo
    # ----------------------------
    # For demonstration, pick one sample image from the test directory.
    # This assumes that test_dir has subdirectories for each class.
    class_folders = os.listdir(test_dir)
    if not class_folders:
        raise ValueError(f"No subdirectories found in the test directory: {test_dir}")
    sample_class = class_folders[0]
    sample_class_dir = os.path.join(test_dir, sample_class)
    sample_images = os.listdir(sample_class_dir)
    if not sample_images:
        raise ValueError(f"No images found in the directory: {sample_class_dir}")
    sample_image_path = os.path.join(sample_class_dir, sample_images[0])
    print(f"Making a prediction on sample image: {sample_image_path}")

    # Load and preprocess the sample image
    img = image.load_img(sample_image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, height, width, channels)
    img_array = img_array / 255.0  # Normalize

    # Since the model expects two inputs (image and label), create a dummy label vector
    dummy_label = np.zeros((1, n_classes), dtype=np.float32)

    # Get predictions from the model
    predictions = model.predict([img_array, dummy_label])
    predicted_index = np.argmax(predictions[0])

    # Retrieve the mapping of class names from the test generator and invert it for lookup
    class_indices = test_generator.generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = index_to_class.get(predicted_index, "Unknown")
    print(f"Predicted class for the sample image: {predicted_class}")

    # ----------------------------
    # Adversarial Attack Evaluation
    # ----------------------------
    # Create an instance of AdversarialEvaluationMetrics using FGSM attack.

    # Attack method name
    attack_name = 'fgsm'

    # Directory to save the adversarial images
    adv_dir = 'Adversarial_Images/'

    # Create an instance of AdversarialEvaluationMetrics
    metrics = AdversarialEvaluationMetrics(model, 
                                           OAA.fgsm_attack_face, 
                                           attack_name, 
                                           num_classes=n_classes,
                                           image_format = '.png',
                                           )

    # Compute the attack success rate and (optionally) save adversarial images.
    attack_success_rate = metrics.compute_attack_success_rate(test_dir, 
                                                              epsilon=0.01, 
                                                              adv_dir=adv_dir, 
                                                              save_image=True)
    
    print(f"FGSM Attack Success Rate: {attack_success_rate:.4f}")


if __name__ == "__main__":
    main()