from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from face_swap_utils import face_swap
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/result'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file1 = request.files['image1']
        file2 = request.files['image2']

        if file1 and file2:
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
            file1.save(path1)
            file2.save(path2)

            result_path = os.path.join(app.config['RESULT_FOLDER'], 'swapped.jpg')
            face_swap(path1, path2, result_path)

            return render_template('index.html', result_image=result_path)

    return render_template('index.html', result_image=None)

if __name__ == '__main__':
    app.run(debug=True)
