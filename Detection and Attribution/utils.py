import tensorflow as tf
from tensorflow.keras import layers
import os
import yaml

class ArcFace(layers.Layer):
    """ArcFace layer implementation."""
    def __init__(self, n_classes=5, s=64.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizer

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        theta = tf.acos(tf.clip_by_value(logits, -1.0 + tf.keras.backend.epsilon(), 
                                       1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        return tf.nn.softmax(logits)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer,
        })
        return config

def setup_model(config_path='config.yml'):
    """Setup and return the model architecture."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load base model
    base_model = tf.keras.applications.Xception(
        input_shape=tuple(config['model']['input_shape']),
        include_top=False,
        weights='imagenet'
    )
    
    # Configure fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:config['model']['fine_tune_at']]:
        layer.trainable = False
    
    # Build the complete model
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    y = tf.keras.Input(shape=(config['model']['num_classes'],))
    output = ArcFace(
        n_classes=config['model']['num_classes'],
        s=config['arcface']['scale'],
        m=config['arcface']['margin']
    )([x, y])
    
    model = tf.keras.Model(inputs=[base_model.input, y], outputs=output)
    
    return model

def setup_callbacks(config_path='config.yml'):
    """Setup and return training callbacks."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience']
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'],
            min_lr=config['training']['min_lr']
        )
    ]
    
    # Setup model checkpoint
    save_path = os.path.join(
        config['paths']['model_save_dir'],
        config['paths']['model_name']
    )
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
    
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            save_path,
            save_best_only=True
        )
    )
    
    return callbacks