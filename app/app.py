import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
# This is where uploaded images will be saved
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD THE AI MODEL ---
# Make sure 'best.pt' is in the same folder as this script!
print("Loading Space Station Model...")
try:
    model = YOLO('best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Did you copy 'best.pt' from your runs/ folder to here?")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if user uploaded a file
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        if file:
            # 1. Save the Original Image
            filename = "input_image.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 2. Run Object Detection
            # conf=0.25: Only detect objects if 25% sure
            results = model(filepath, conf=0.25)

            # 3. Draw Boxes & Save Result
            # results[0].plot() returns the image with boxes drawn on it
            result_img = results[0].plot()
            
            output_filename = "output_image.jpg"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, result_img)

            # 4. Show Results to User
            return render_template('index.html', 
                                   original=filename, 
                                   result=output_filename)

    return render_template('index.html')

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(debug=True, port=5000)