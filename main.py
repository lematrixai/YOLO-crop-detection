from ultralytics import YOLO
from PIL import Image
import torch

# Define valid disease classes
VALID_DISEASES = {
    'maize_streak_virus': 'Maize Streak Virus (MSV)',
    'maize_lethal_necrosis': 'Maize Lethal Necrosis (MLN)',
    'healthy': 'Healthy Corn'
}

# ✅ STEP 1: Load the YOLO model
try:
    model = YOLO("Corn-Disease50Epoch.pt")
    print("Model loaded ✅")
    
    # Print model's classes
    print("\nModel Classes:")
    for idx, class_name in model.names.items():
        print(f"{idx}: {class_name}")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit()

def analyze_image(image_path):
    try:
        # Load image
        image = Image.open(image_path)
        print(f"\nImage loaded ✅ | Size: {image.size}")

        # Run inference
        results = model(image)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get confidence score
                conf = box.conf[0].item()
                # Get class name
                cls = int(box.cls[0].item())
                cls_name = model.names[cls]
                
                # Only include if confidence is above threshold
                if conf >= 0.5:  # 50% confidence threshold
                    detections.append({
                        'disease': cls_name,
                        'confidence': conf,
                        'box': box.xyxy[0].tolist()
                    })

        if not detections:
            return {
                'status': 'rejected',
                'message': 'No corn leaf or disease detected in the image. Please upload a clear image of a corn leaf.',
                'confidence': 0.0
            }

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        best_detection = detections[0]

        # Check if detected disease is in our valid list
        if best_detection['disease'] not in VALID_DISEASES:
            return {
                'status': 'rejected',
                'message': f'Detected disease ({best_detection["disease"]}) is not in our supported list. Please upload an image of corn leaves with MSV, MLN, or healthy corn.',
                'confidence': best_detection['confidence']
            }

        # Return successful detection
        return {
            'status': 'success',
            'disease': VALID_DISEASES[best_detection['disease']],
            'confidence': best_detection['confidence'],
            'message': f'Detected {VALID_DISEASES[best_detection["disease"]]} with {best_detection["confidence"]*100:.1f}% confidence'
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error processing image: {str(e)}',
            'confidence': 0.0
        }

# Test the function
if __name__ == "__main__":
    result = analyze_image("test-1.jpg")
    print("\nAnalysis Results:")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if result['status'] == 'success':
        print(f"Disease: {result['disease']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
