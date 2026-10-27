from flask import Flask, render_template, request, redirect, url_for, flash
import os
import time
import uuid
from face_swap_utils import face_swap
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Session-signing key comes from the environment; the fallback is only for
# local development. Set SECRET_KEY in production - the default key lets
# anyone forge session cookies.
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")

# Absolute folders next to this file, created on startup, so uploads never
# fail with FileNotFoundError on a fresh clone (old code used CWD-relative
# paths that were never created).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'result')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Uploaded face photos are sensitive and pile up forever on a
# long-running server. Delete files older than a day on startup so the
# app does not hoard user photos (or fill the disk). The window is
# configurable via UPLOAD_RETENTION_SECONDS; the current result image is
# always fresh, so it is never touched by the prune.
def _env_int(name, default):
    """Read an integer environment variable, falling back to *default* when
    unset or malformed instead of crashing the app at startup."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Reject oversized uploads before they are read into memory (16 MB default
# cap). Read through _env_int (not a bare int() cast) so a malformed
# MAX_UPLOAD_MB value falls back to 16 instead of crashing startup.
app.config["MAX_CONTENT_LENGTH"] = _env_int("MAX_UPLOAD_MB", 16) * 1024 * 1024


STALE_FILE_MAX_AGE_SECONDS = _env_int("UPLOAD_RETENTION_SECONDS", 24 * 3600)


def _prune_stale_files(folder, max_age_seconds):
    """Delete files in *folder* older than *max_age_seconds*; return the count."""
    # A negative retention window is invalid config, not "delete
    # everything": now - mtime is always greater than a negative age,
    # so without this guard a typo like -1 would wipe the folders
    # on startup instead of pruning nothing.
    if max_age_seconds is None or max_age_seconds < 0:
        return 0
    removed = 0
    now = time.time()
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    for name in names:
        path = os.path.join(folder, name)
        try:
            if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
                os.remove(path)
                removed += 1
        except OSError:
            continue
    return removed


_prune_stale_files(UPLOAD_FOLDER, STALE_FILE_MAX_AGE_SECONDS)
_prune_stale_files(RESULT_FOLDER, STALE_FILE_MAX_AGE_SECONDS)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_filename(filename):
    """Prefix the sanitized name with a short random id.

    Two users uploading "photo.jpg" at the same time must not overwrite
    each other's in-progress files, and the stale-file prune must not
    delete a file another request is still writing.
    """
    return f"{uuid.uuid4().hex[:8]}_{filename}"


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

        # Unique names so concurrent uploads with the same original
        # filename (e.g. two users uploading "photo.jpg") do not
        # overwrite each other.
        # unique_filename() always prefixes a random id, so the sanitized
        # base name must be validated first - otherwise this check could
        # never trigger (e.g. for names werkzeug strips to nothing).
        base1 = secure_filename(file1.filename)
        base2 = secure_filename(file2.filename)
        if not base1 or not base2:
            flash('Invalid file name, please rename the images and try again.')
            return redirect(url_for('index'))
        filename1 = unique_filename(base1)
        filename2 = unique_filename(base2)

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(path1)
        file2.save(path2)

        result_path = os.path.join(app.config['RESULT_FOLDER'], unique_filename('swapped.jpg'))
        try:
            face_swap(path1, path2, result_path)
        except Exception as exc:
            # A failed swap must not leave orphan uploads behind for the
            # startup prune to clean up hours later.
            for stale in (path1, path2):
                try:
                    os.remove(stale)
                except OSError:
                    pass
            flash(f'Face swap failed: {exc}')
            return redirect(url_for('index'))

        # Pass a static-relative path; the template used to split the
        # absolute filesystem path on 'static/', which breaks on Windows
        # (backslashes) and leaks the server directory layout.
        result_image = 'result/' + os.path.basename(result_path)
        return render_template('index.html', result_image=result_image)

    return render_template('index.html', result_image=None)


if __name__ == "__main__":
    # Debug mode only when explicitly enabled; never in production.
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
