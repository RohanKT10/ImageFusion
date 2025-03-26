from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return redirect(url_for('process_image', filename=filename))

@app.route('/process/<filename>', methods=['GET', 'POST'])
def process_image(filename):
    if request.method == 'POST':
        operation = request.form.get('operation')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = cv2.imread(filepath)

        if image is None:
            return "Error loading image."

        # Flag to track if image has already been saved
        saved_by_pil = False

        # Operation: Resize
        if operation == 'resize':
            width = int(request.form.get('width'))
            height = int(request.form.get('height'))
            processed = cv2.resize(image, (width, height))
            output_filename = 'resized_' + filename
            result_width, result_height = width, height

        # Compress
        elif operation == 'compress':
            # Get compression quality; default to 50 if not provided
            compression_quality = int(request.form.get('compression_quality', 50))
            # Ensure the quality is within 1-100
            if compression_quality < 1:
                compression_quality = 1
            elif compression_quality > 100:
                compression_quality = 100
            # Determine output filename:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.jpg', '.jpeg']:
                output_filename = 'compressed_' + os.path.splitext(filename)[0] + '.jpg'
            else:
                output_filename = 'compressed_' + filename
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            # Save image using JPEG quality parameter.
            cv2.imwrite(output_filepath, image, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
            processed = image
            result_width, result_height = image.shape[1], image.shape[0]
            saved_by_pil = True



        # Operation: Color Scale (Grayscale, Red, Green, Blue)
        elif operation == 'color_scale':
            color_scale = request.form.get('color_scale')
            keep_original_size = request.form.get('keep_original_size') == 'on'
            if color_scale == 'gray':
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                output_filename = 'gray_' + filename
            elif color_scale == 'red':
                processed = image.copy()
                processed[:, :, 1] = 0
                processed[:, :, 0] = 0
                output_filename = 'red_' + filename
            elif color_scale == 'green':
                processed = image.copy()
                processed[:, :, 0] = 0
                processed[:, :, 2] = 0
                output_filename = 'green_' + filename
            elif color_scale == 'blue':
                processed = image.copy()
                processed[:, :, 1] = 0
                processed[:, :, 2] = 0
                output_filename = 'blue_' + filename

            if not keep_original_size:
                width = int(request.form.get('width', image.shape[1]))
                height = int(request.form.get('height', image.shape[0]))
                processed = cv2.resize(processed, (width, height))
            else:
                width, height = image.shape[1], image.shape[0]
            result_width, result_height = width, height

        # Operation: Edge Detection using Canny
        elif operation == 'edge_detection':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processed = cv2.Canny(gray, 100, 200)
            output_filename = 'edges_' + filename
            result_width, result_height = processed.shape[1], processed.shape[0]

        # Operation: Face Blurring (Privacy Filter)
        elif operation == 'face_blur':
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            processed = image.copy()
            for (x, y, w, h) in faces:
                face_region = processed[y:y + h, x:x + w]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                processed[y:y + h, x:x + w] = blurred_face
            output_filename = 'face_blurred_' + filename
            result_width, result_height = processed.shape[1], processed.shape[0]

        else:
            return "Unknown operation selected."

        # Save the image only if PIL didn’t already save it
        if not saved_by_pil:
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_filepath, processed)

        return render_template('result.html', filename=output_filename, width=result_width, height=result_height)

    return render_template('resize.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
