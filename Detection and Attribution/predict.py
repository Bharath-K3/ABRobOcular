import numpy as np
from tensorflow.keras.preprocessing import image
import yaml
from eval import load_model
from dataset_loader import DataLoader

class Predictor:
    def __init__(self, config_path='config.yml'):
        self.model = load_model(config_path)
        self.config = self._load_config(config_path)
        self.class_indices = self._get_class_indices()
    
    def _load_config(self, config_path):
        """Load configuration from yaml file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _get_class_indices(self):
        """Get class indices from training data."""
        data_loader = DataLoader(self.config)
        _, class_indices = data_loader.get_train_data()
        return {v: k for k, v in class_indices.items()}
    
    def predict_image(self, img_path):
        """Make prediction for a single image."""
        # Load and preprocess the image
        img = image.load_img(
            img_path,
            target_size=tuple(self.config['data']['target_size'])
        )
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        
        # Prepare dummy label input for ArcFace layer
        dummy_label_input = np.zeros((1, self.config['model']['num_classes']))
        
        # Make prediction
        predictions = self.model.predict([x, dummy_label_input])
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_indices[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        return {
            'class_name': predicted_class,
            'class_index': predicted_class_idx,
            'confidence': float(confidence),
            'probabilities': {
                self.class_indices[i]: float(prob)
                for i, prob in enumerate(predictions[0])
            }
        }

if __name__ == '__main__':
    # Example usage
    predictor = Predictor()
    
    # Example paths for different types of images
    test_images = [
        'multi-detector/test/No Attack/1141/1141_l_1.png',
        'multi-detector/test/PGD/1141/1141_l_1_pgd_01.png',
        'multi-detector/test/MIM/3265/3265_l_1_mim_001.png'
    ]
    
    for img_path in test_images:
        result = predictor.predict_image(img_path)
        print(f"\nPrediction for {img_path}:")
        print(f"Predicted class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Class probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")