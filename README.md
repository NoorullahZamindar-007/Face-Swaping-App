# Face Swap using Dlib and OpenCV

A Flask web app that detects faces with `dlib`'s 68-point facial landmark model
and swaps faces between two uploaded images: it Delaunay-triangulates the
first image's landmarks, warps each triangle onto the matching triangle of
the second face, and seamless-clones the result onto the second photo —
the same pipeline demonstrated in `Face_Swap.ipynb`. The first image's
face ends up on the second image.

## Project structure

```
Face-Swaping-App/
├── app.py                    # Thin shim importing face_swap_app/app.py
│                             # (run from the repo root or from
│                             # face_swap_app/ - both serve the same app)
├── face_swap_utils.py        # Thin shim re-exporting
│                             # face_swap_app/face_swap_utils.py
├── face_swap_app/
│   ├── app.py                # Flask app: upload form, validation, swap route
│   ├── face_swap_utils.py    # Face detection, landmark triangulation,
│   │                         # triangle warp and seamless-clone swap
│   ├── requirements.txt      # Python dependencies
│   ├── templates/
│   │   ├── index.html        # Upload form and result page
│   │   └── error.html        # Branded 404/405/500 error page (oversized
│   │                         # uploads flash a message instead)
│   └── Face_Swap.ipynb       # Jupyter Notebook demonstrating the technique
└── README.md
```

## Setup

1. Install dependencies:

   ```bash
   pip install -r face_swap_app/requirements.txt
   ```

2. Download the landmark model (not shipped with the repo, ~100 MB):

   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

   Extract it and place `shape_predictor_68_face_landmarks.dat` next to
   `face_swap_app/face_swap_utils.py`.

3. Run the app (from the repo root, or from `face_swap_app/` — both
   serve the same app):

   ```bash
   python app.py
   ```

   Open http://127.0.0.1:5000, upload two face photos, and submit.

## Configuration (environment variables)

| Variable                   | Default | Purpose                                          |
|----------------------------|---------|--------------------------------------------------|
| `SECRET_KEY`               | dev key | Flask session signing key (set a real one in production) |
| `PORT`                     | `5000`  | Port the app listens on (invalid or out-of-range values fall back to 5000) |
| `FLASK_HOST`               | `127.0.0.1` | Host interface to bind                       |
| `MAX_UPLOAD_MB`            | `16`    | Maximum size per uploaded image                  |
| `UPLOAD_RETENTION_SECONDS` | `86400` | Uploads/results older than this are deleted (on startup and on each request) |
| `FLASK_DEBUG`              | `0`     | Set to `1` to enable Flask debug mode locally    |

## Running tests

```bash
pip install pytest
python -m pytest tests/
```

Or run a single file directly:

```bash
python tests/test_app_helpers.py      # no pytest needed
python tests/test_face_swap_utils.py  # needs pytest and OpenCV installed
```

The suite covers the swap pipeline (triangulation, warping, blending),
upload validation, unique filenames, stale-file pruning, user-facing
error messages, and port/env parsing. `test_app_helpers.py` runs
without Flask, OpenCV, or dlib installed (they are stubbed out);
`test_face_swap_utils.py` needs OpenCV installed (dlib is stubbed).

## Features

- Detect faces using `dlib`'s HOG-based face detector
- Extract facial landmarks using the 68-point model
- Build a face mask from the convex hull of the facial landmarks
- Upload validation (image types only, size cap, safe filenames)
- Concurrent-safe uploads: files are stored under unique names
- Uploaded and result images older than 24 h are pruned on startup and on each request

## Limitations

- Only the largest (most prominent) detected face in each photo is swapped.
- Photos larger than 4000 px on the long edge are downscaled before
  swapping, so the result is at most 4000 px on a side.
- Works best on frontal, well-lit faces: the dlib HOG detector can miss
  faces that are turned away, partly occluded, or very small.
- The landmark model (~100 MB) must be downloaded separately; without it
  the app reports a clear error instead of swapping.

## Author

**Noorullah Zamindar**
AI Developer | Python Enthusiast

## License

MIT License.
