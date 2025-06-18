   # Face-Swaping-App
face_swap_app using flask               
Here’s a professional and clean `README.md` file for your **Face Swap** project to upload to GitHub:

---            
                       
``markdown                                           
# Face Swap using Dlib and OpenCV                    

This project demonstrates a basic **face swapping** application using Python, OpenCV, and Dlib. It detects facial landmarks and performs a basic face mask transfer between two input images.                   
                                                                                    
                                                                                                                                                                                                 
``                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                    
├── Face_Swap.ipynb           # Jupyter Notebook demonstrating face swapping                                                                               
                                                                                                                                       
├── face_swap_utils.py        # Python utility functions for face detection and swapping                                                                                                                                                                                                                                                                     
                                                                                      
├── Documentation.docx        # Brief overview of the methodology                                                                                                                                                         
                                                                                                                                                       
└── shape_predictor_68_face_landmarks.dat  # Required model (Not included due to size)                                                                           
                                    
``                                   
                                       
## ⚙️ Features

- Detect faces using `dlib`'s HOG-based face detector   
- Extract facial landmarks using the 68-point model
- Perform Delaunay triangulation for accurate face mapping
- Apply a face mask from one image to another
- Output a partially swapped face image

## 🧠 Requirements

- Python 3.7+
- OpenCV
- Dlib
- NumPy
                   
### Install dependencies

``bash
pip install -r requirements.txt
``

> Note: You need to download the `shape_predictor_68_face_landmarks.dat` file from [dlib's official model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

## 📝 Usage

### Run from script
``python
from face_swap_utils import face_swap

face_swap("image1.jpg", "image2.jpg", "output.jpg")
``

### Or use the Jupyter Notebook
Open the notebook `Face_Swap.ipynb` and run the cells to see the step-by-step process.

## 📚 Documentation

Included in `Documentation.docx`:
- Library Imports
- Image loading and preprocessing
- Landmark detection using Dlib
- Face triangulation and masking

## 🚧 Limitations

- The current version only performs partial face transfer (masked image).
- The complete warping and blending process can be implemented in future versions.

## 👨‍💻 Author

**Noorullah Zamindar**  
Freelancer | AI Developer | Python Enthusiast

## 📝 License

This project is open-source and free to use under the MIT License.
``

---

