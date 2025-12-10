from ultralytics import YOLO
import os

def main():
    # 1. Load your BEST trained model
    # After training, YOLO saves weights in 'runs/detect/space_station_model/weights/best.pt'
    # MAKE SURE THIS PATH IS CORRECT based on your actual folder structure
    model_path = 'runs/detect/space_station_model/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained model at {model_path}. Did you run train.py?")
        return

    model = YOLO(model_path)

    # 2. Validate on the Test Set (Calculate mAP)
    # We use mode='val' but point it to the 'test' split to get accurate metrics for the report
    print("Running evaluation on Test set...")
    metrics = model.val(
        data='data.yaml', 
        split='test',       # Important: Use the test split!
        name='final_evaluation'
    )

    # 3. Print Results
    print(f"mAP@0.5: {metrics.box.map50}")
    print(f"mAP@0.5-0.95: {metrics.box.map}")
    
    # 4. Run visual predictions on a few images (Optional, for the report images)
    # This saves images with boxes drawn on them to 'runs/detect/predict_visuals'
    model.predict(
        source='./dataset/test/images', # Point to test images
        conf=0.25,                      # Confidence threshold
        save=True, 
        name='predict_visuals'
    )
    print("Visual predictions saved to runs/detect/predict_visuals")

if __name__ == '__main__':
    main()