# Face Swap using Dlib and OpenCV

A Flask web app that detects faces with `dlib`'s 68-point facial landmark model
and performs a basic face-mask transfer between two uploaded images.

## Project structure

```
Face-Swaping-App/
├── face_swap_app/
│   ├── app.py                # Flask app: upload form, validation, swap route
│   ├── face_swap_utils.py    # Face detection / landmark / mask utilities
│   ├── requirements.txt      # Python dependencies
│   ├── templates/
│   │   └── index.html        # Upload form and result page
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

3. Run the app:

   ```bash
   cd face_swap_app
   python app.py
   ```

   Open http://127.0.0.1:5000, upload two face photos, and submit.

## Configuration (environment variables)

| Variable                   | Default | Purpose                                          |
|----------------------------|---------|--------------------------------------------------|
| `SECRET_KEY`               | dev key | Flask session signing key (set a real one in production) |
| `MAX_UPLOAD_MB`            | `16`    | Maximum size per uploaded image                  |
| `UPLOAD_RETENTION_SECONDS` | `86400` | Uploads/results older than this are deleted on startup |
| `FLASK_DEBUG`              | `0`     | Set to `1` to enable Flask debug mode locally    |

## Features

- Detect faces using `dlib`'s HOG-based face detector
- Extract facial landmarks using the 68-point model
- Build a face mask from the convex hull of the facial landmarks
- Upload validation (image types only, size cap, safe filenames)
- Concurrent-safe uploads: files are stored under unique names
- Uploaded and result images older than 24 h are pruned on startup

## Limitations

- The current version only performs partial face transfer (a masked face
  image), not a full warp-and-blend swap onto the second image.
- The output is built from the first image only — the second image is
  validated but not used in the current implementation.

## Author

**Noorullah Zamindar**
AI Developer | Python Enthusiast

## License

MIT License.
