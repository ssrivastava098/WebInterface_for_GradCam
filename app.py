import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from prediction import predict

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

latest_uploaded_image = None
latest_processed_image_gradCAM = None
latest_processed_image_Saliency = None
predictions = []

@app.route('/')
def index():
    global latest_uploaded_image, latest_processed_image_gradCAM, latest_processed_image_Saliency, predictions
    return render_template('index.html', latest_uploaded_image=latest_uploaded_image, latest_processed_image_gradCAM=latest_processed_image_gradCAM, latest_processed_image_Saliency=latest_processed_image_Saliency,predictions=predictions)

@app.route('/upload', methods=['POST'])
def upload_file():
    global latest_uploaded_image, latest_processed_image_gradCAM, latest_processed_image_Saliency, predictions
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        latest_uploaded_image = filename  

        output_filename = 'processed_gradCAM' + filename
        outpath_gradCAM = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        # process_image(input_path, output_path)

        latest_processed_image_gradCAM = output_filename 

        output_filename = 'processed_Saliency' + filename
        outpath_Saliency = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        # generate_saliency_map(input_path, output_path)

        latest_processed_image_Saliency = output_filename 

        predictions = predict(input_path,outpath_gradCAM,outpath_Saliency)
        return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
