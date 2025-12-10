from ultralytics import YOLO

def main():
    # 1. Load the model (Nano version is fastest)
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt') 

    # 2. Train the model
    # We point to 'data.yaml' which is in the same folder as this script
    print("Starting training...")
    results = model.train(
        data='data.yaml',
        epochs=50,       # You can reduce this to 20-30 if you are in a hurry
        imgsz=640,       # Standard image size
        batch=8,         # Set to 8 or 4 to save memory on your laptop
        name='space_station_run',
        device='0' if 0 else 'cpu' # Use GPU if available, else CPU
    )
    
    print("Training finished! Results saved in runs/detect/space_station_run")

if __name__ == '__main__':
    main()