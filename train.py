import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import yaml
from dataset_loader import create_data_generators, load_config

class ArcFace(layers.Layer):
    """ArcFace layer implementation."""
    def __init__(self, n_classes=300, s=64.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizer

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[0][-1], self.n_classes),
            initializer='glorot_uniform',
            trainable=True,
            regularizer=self.regularizer
        )

    def call(self, inputs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        theta = tf.acos(tf.clip_by_value(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
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

def create_model(config):
    """Create and compile the model."""
    base_model = tf.keras.applications.Xception(
        input_shape=tuple(config['model']['input_shape']),
        include_top=False,
        weights='imagenet'
    )
    
    # Fine-tuning setup
    base_model.trainable = True
    for layer in base_model.layers[:config['model']['fine_tune_at']]:
        layer.trainable = False
    
    # Build model architecture
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(config['model']['dense_units'], activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    y = tf.keras.Input(shape=(config['model']['n_classes'],))
    output = ArcFace(
        n_classes=config['model']['n_classes'],
        s=config['arcface']['scale'],
        m=config['arcface']['margin']
    )([x, y])
    
    model = Model(inputs=[base_model.input, y], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

def train():
    """Main training function."""
    config = load_config()
    
    # Create data generators
    train_generator, val_generator = create_data_generators()
    
    # Create model
    model = create_model(config)
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stop_patience']
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'],
            min_lr=config['training']['min_lr']
        ),
        ModelCheckpoint(
            os.path.join(config['paths']['model_save_dir'], config['paths']['model_name']),
            save_best_only=True
        )
    ]
    
    # Create model save directory if it doesn't exist
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
    
    # Train the model
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['training']['epochs'],
        callbacks=callbacks
    )

if __name__ == "__main__":
    train()