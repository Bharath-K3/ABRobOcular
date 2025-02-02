import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset_loader import create_test_generator, load_config
from train import ArcFace
import os

def evaluate():
    """Evaluate the trained model on test data."""
    config = load_config()
    
    # Load the saved model
    model = load_model(
        os.path.join(config['paths']['model_save_dir'], config['paths']['model_name']),
        custom_objects={'ArcFace': ArcFace}
    )
    
    # Create test generator
    test_generator = create_test_generator()
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')

if __name__ == "__main__":
    evaluate()