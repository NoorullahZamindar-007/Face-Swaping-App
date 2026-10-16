from flask import Flask, render_template, request, redirect, url_for, flash
import os
from face_swap_utils import face_swap
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Session-signing key comes from the environment; the fallback is only for
# local development. Set SECRET_KEY in production - the default key lets
# anyone forge session cookies.
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
# Reject oversized uploads before they are read into memory (16 MB default cap).
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "16")) * 1024 * 1024

# Absolute folders next to this file, created on startup, so uploads never
# fail with FileNotFoundError on a fresh clone (old code used CWD-relative
# paths that were never created).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'result')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def upload_too_large(_error):
    """Oversized upload: explain the limit instead of a bare 413 page."""
    max_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    flash(f"Image too large - please keep each upload under {max_mb} MB.")
    return redirect(url_for("index"))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # .get() instead of [] so a malformed form cannot raise KeyError
        file1 = request.files.get('image1')
        file2 = request.files.get('image2')

        if not file1 or not file1.filename:
            flash('Please choose the first image.')
            return redirect(url_for('index'))
        if not file2 or not file2.filename:
            flash('Please choose the second image.')
            return redirect(url_for('index'))
        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            flash('Both uploads must be images (png, jpg, jpeg, bmp, webp).')
            return redirect(url_for('index'))

        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        if not filename1 or not filename2:
            flash('Invalid file name, please rename the images and try again.')
            return redirect(url_for('index'))

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(path1)
        file2.save(path2)

        result_path = os.path.join(app.config['RESULT_FOLDER'], 'swapped.jpg')
        try:
            face_swap(path1, path2, result_path)
        except Exception as exc:
            flash(f'Face swap failed: {exc}')
            return redirect(url_for('index'))

        return render_template('index.html', result_image=result_path)

    return render_template('index.html', result_image=None)


if __name__ == "__main__":
    # Debug mode only when explicitly enabled; never in production.
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
