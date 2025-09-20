
import torch
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random

class CattleBreedPredictor:
    def __init__(self, model_path='models/cattle_breed_classifier_complete.pth'):
        """
        Initialize the cattle breed classifier
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the saved model package
        model_package = torch.load(model_path, map_location=self.device)
        
        # Extract components
        self.class_names = model_package['class_names']
        self.model_config = model_package['model_config']
        
        # Initialize model and processor
        self.processor = ViTImageProcessor.from_pretrained(self.model_config['model_name'])
        self.model = ViTForImageClassification.from_pretrained(
            self.model_config['model_name'],
            num_labels=self.model_config['num_classes'],
            ignore_mismatched_sizes=True
        )
        
        # Load trained weights
        self.model.load_state_dict(model_package['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully with {len(self.class_names)} classes")
        print("Available breeds:", ", ".join(self.class_names))
    
    def predict(self, image_path, show_results=True):
        """
        Predict breed for a single image
        """
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                return None, 0.0, None
            
            # Open image
            image = Image.open(image_path).convert('RGB')
            
            if show_results:
                # Display original image
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                plt.title("Input Image", fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # Preprocess for ViT
            preprocess_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            
            image_tensor = preprocess_transform(image)
            image_pil = transforms.ToPILImage()(image_tensor)
            
            # Process with ViT processor
            inputs = self.processor(images=image_pil, return_tensors="pt")
            processed_image = inputs['pixel_values'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(processed_image)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted.item()]
            confidence_val = confidence.item()
            all_probs = probabilities.cpu().numpy()[0]
            
            # Display results
            print("\n" + "="*50)
            print("CATTLE/BUFFALO BREED PREDICTION RESULTS")
            print("="*50)
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Predicted Breed: {predicted_class}")
            print(f"Confidence: {confidence_val:.4f}")
            
            # Show top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            print("\nTop 3 Predictions:")
            for i in range(3):
                prob = top3_probs[0][i].item()
                class_idx = top3_indices[0][i].item()
                print(f"   {i+1}. {self.class_names[class_idx]}: {prob:.4f}")
            
            if show_results:
                # Plot confidence distribution
                plt.figure(figsize=(12, 6))
                colors = ['red' if i == predicted.item() else 'blue' for i in range(len(self.class_names))]
                bars = plt.bar(range(len(self.class_names)), all_probs, color=colors, alpha=0.7)
                plt.xlabel('Cattle/Buffalo Breeds', fontsize=12)
                plt.ylabel('Confidence Score', fontsize=12)
                plt.title(f'Prediction Confidence: {predicted_class} ({confidence_val:.4f})', fontsize=14)
                plt.xticks(range(len(self.class_names)), self.class_names, rotation=45, ha='right')
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                
                # Add value labels on top bars
                for bar, prob in zip(bars, all_probs):
                    if prob > 0.1:
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                plt.show()
            
            return predicted_class, confidence_val, all_probs
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, 0.0, None
    
    def predict_multiple(self, image_folder, num_images=5):
        """
        Predict breeds for multiple images in a folder
        """
        image_files = []
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print("No images found in the specified folder.")
            return []
        
        print(f"Found {len(image_files)} images in folder")
        
        # Select random images
        selected_images = image_files[:num_images] if len(image_files) <= num_images else random.sample(image_files, num_images)
        
        results = []
        print(f"\nPredicting {len(selected_images)} images...")
        
        for i, img_path in enumerate(selected_images):
            print(f"\n{'='*40}")
            print(f"Image {i+1}/{len(selected_images)}")
            print(f"{'='*40}")
            
            pred_class, confidence, _ = self.predict(img_path, show_results=True)
            
            if pred_class:
                results.append({
                    'image_path': img_path,
                    'predicted_class': pred_class,
                    'confidence': confidence
                })
        
        return results

def main():
    """
    Example usage of the cattle breed predictor
    """
    # Initialize predictor
    print("Initializing Cattle Breed Predictor...")
    predictor = CattleBreedPredictor('models/cattle_breed_classifier_complete.pth')
    
    # Example: Predict a single image
    image_path = input("\nEnter the path to a cattle image (or press Enter for demo): ").strip()
    
    if image_path and os.path.exists(image_path):
        predicted_breed, confidence, _ = predictor.predict(image_path, show_results=True)
        if predicted_breed:
            print(f"\nFinal Result: {predicted_breed} ({confidence:.2%} confidence)")
    else:
        print("No valid image path provided. Ready to predict when you provide an image path.")
        print("\nUsage example:")
        print("predictor = CattleBreedPredictor('models/cattle_breed_classifier_complete.pth')")
        print("predicted_breed, confidence, probs = predictor.predict('path/to/your/image.jpg')")

if __name__ == "__main__":
    main()
