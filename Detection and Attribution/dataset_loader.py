import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

class DataLoader:
    def __init__(self, config_path='config.yml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.datagen = ImageDataGenerator(rescale=1./255)
        
    def _create_generator(self, directory):
        """Create a data generator for a specific directory."""
        return self.datagen.flow_from_directory(
            directory,
            target_size=tuple(self.config['data']['target_size']),
            batch_size=self.config['training']['batch_size'],
            class_mode='sparse'
        )
    
    def arcface_generator(self, generator):
        """Wrap the regular generator to provide ArcFace compatible format."""
        num_classes = self.config['model']['num_classes']
        while True:
            data = next(generator)
            yield [data[0], tf.one_hot(data[1], depth=num_classes)], tf.one_hot(data[1], depth=num_classes)
    
    def get_train_data(self):
        """Get training data generator."""
        train_data = self._create_generator(self.config['data']['train_dir'])
        return self.arcface_generator(train_data), train_data.class_indices
    
    def get_validation_data(self):
        """Get validation data generator."""
        val_data = self._create_generator(self.config['data']['validation_dir'])
        return self.arcface_generator(val_data)
    
    def get_test_data(self):
        """Get test data generator."""
        test_data = self._create_generator(self.config['data']['test_dir'])
        return self.arcface_generator(test_data)
    
    def get_steps_per_epoch(self):
        """Calculate steps per epoch for training."""
        train_data = self._create_generator(self.config['data']['train_dir'])
        return len(train_data)
    
    def get_validation_steps(self):
        """Calculate validation steps."""
        val_data = self._create_generator(self.config['data']['validation_dir'])
        return len(val_data)