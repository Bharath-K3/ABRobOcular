import tensorflow as tf
import yaml
from dataset_loader import DataLoader
from utils import setup_model, setup_callbacks

def train_model(config_path='config.yml'):
    """Train the model using configuration from config.yml."""
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize data loader
    data_loader = DataLoader(config_path)
    
    # Get data generators
    train_generator, _ = data_loader.get_train_data()
    validation_generator = data_loader.get_validation_data()
    
    # Setup model
    model = setup_model(config_path)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['training']['learning_rate']
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(config_path)
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=data_loader.get_steps_per_epoch(),
        validation_steps=data_loader.get_validation_steps(),
        epochs=config['training']['epochs'],
        callbacks=callbacks
    )
    
    return history

if __name__ == '__main__':
    history = train_model()