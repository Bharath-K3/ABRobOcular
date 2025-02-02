import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

def load_config():
    """Load configuration from yaml file."""
    with open('config.yml', 'r') as file:
        return yaml.safe_load(file)

class CustomDataGen(tf.keras.utils.Sequence):
    """Custom data generator for handling paired inputs (images and labels)."""
    def __init__(self, directory, datagen, batch_size=32, target_size=(256, 256), class_mode='categorical'):
        self.generator = datagen.flow_from_directory(
            directory, 
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode
        )
        
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, index):
        x, y = self.generator[index]
        return [x, y], y

def create_data_generators():
    """Create train and validation data generators with augmentation."""
    config = load_config()
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = CustomDataGen(
        config['data']['train_dir'],
        train_datagen,
        batch_size=config['training']['batch_size'],
        target_size=tuple(config['data']['target_size']),
        class_mode=config['data']['class_mode']
    )
    
    val_generator = CustomDataGen(
        config['data']['val_dir'],
        val_datagen,
        batch_size=config['training']['batch_size'],
        target_size=tuple(config['data']['target_size']),
        class_mode=config['data']['class_mode']
    )
    
    return train_generator, val_generator

def create_test_generator():
    """Create test data generator."""
    config = load_config()
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return CustomDataGen(
        config['data']['test_dir'],
        test_datagen,
        batch_size=config['training']['batch_size'],
        target_size=tuple(config['data']['target_size']),
        class_mode=config['data']['class_mode']
    )