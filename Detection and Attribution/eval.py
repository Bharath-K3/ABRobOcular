import tensorflow as tf
import yaml
from dataset_loader import DataLoader
from utils import ArcFace

def load_model(config_path='config.yml'):
    """Load the trained model."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_path = f"{config['paths']['model_save_dir']}/{config['paths']['model_name']}"
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'ArcFace': ArcFace}
    )

def evaluate_model(config_path='config.yml'):
    """Evaluate the model on test data."""
    # Load model
    model = load_model(config_path)
    
    # Initialize data loader
    data_loader = DataLoader(config_path)
    
    # Get test generator
    test_generator = data_loader.get_test_data()
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(
        test_generator,
        steps=10000  # You might want to adjust this based on your test set size
    )
    
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    return test_loss, test_acc

if __name__ == '__main__':
    evaluate_model()