== Plant Classification System ==

A comprehensive deep learning-based plant species classification system with multiple model architectures and a user-friendly Gradio interface.

This system classifies plant species using multiple advanced deep learning models, including domain adaptation techniques and hybrid architectures. The GUI allows users to upload plant images and get predictions with corresponding herbarium reference images.

== Project Structure ==
gui/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── species_list.txt      # Species name mappings
├── herbarium/            # Herbarium reference images
│   ├── 12254/
│   ├── 12255/
│   └── ...
└── saved_models/         # Pre-trained model weights (download separately)
    ├── Baseline1/
    ├── Baseline2/
    ├── Novel1/
    ├── Novel2/
    └── Novel3/

== Installation ==
Step 1: Clone/Download the project and Navigate to /gui Directory

Step 2: Install Dependencies
Make sure you're in the correct directory (the one with requirements.txt), then run:
pip install -r requirements.txt

Step 3: Download Model Weights
Download Link:
https://drive.google.com/file/d/1v5iyANRjlwtQ0aKuA94kmoLA2gdb5ZZT/view?usp=sharing
Download Instructions:
- Download the entire saved_models folder
- Extract the zip file
- Place the saved_models folder in the same directory as app.py
- Your folder structure should look like this:

gui/
├── app.py
├── requirements.txt
├── species_list.txt
├── herbarium/
└── saved_models/    ← This should be added here

Step 4: Verify File Structure
Before running, ensuring your folder contains all these files and folders:

# In your terminal, run this command in the gui folder:
ls -la  # On Windows: dir

# You should see:
# app.py
# requirements.txt
# species_list.txt
# herbarium/ (folder)
# saved_models/ (folder) - this is the one you downloaded

== Running the Application ==
Make sure you're still in the gui folder, then run:
python app.py

The application will start and provide a local URL (typically http://127.0.0.1:7860). Open this URL in your web browser.



