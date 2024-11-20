# Remove Background on Image & Video Using RMBG-2.0
##### RMBG v2.0 Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/BRIA-RMBG-2.0-Demo/blob/main/rmbg_2.0.ipynb) <br>

## Support

- **Remove background from a single image or multiple images**
- **Remove background from video (create a green screen effect)**
  <br>
![image](https://github.com/user-attachments/assets/53654f9e-7387-4e5a-a273-c09fddf4d687)
![rmbg_video](https://github.com/user-attachments/assets/991951eb-7a8e-47f3-8751-989daed37d1e)


https://github.com/user-attachments/assets/4996ce13-24f9-484d-8ebd-7ba8e29f879d


https://github.com/user-attachments/assets/1d12f8eb-9824-41ad-94f1-b4e0cfac00a3

[Video Credit](https://pixabay.com/videos/woman-hair-drying-bathroom-bathe-37325/)


##### RMBG v2.0 Google Colab All code in single notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/BRIA-RMBG-2.0-Demo/blob/main/rmbg_2.0_embedded.ipynb) <br>

## Local Set Up

### Step 1: Create and activate a virtual environment
### Using virtualenv
##### Create a virtual environment
```
python -m venv RMBG
```
##### Activate the virtual environment (Linux/Mac)
```
source RMBG/bin/activate
```
##### Activate the virtual environment (Windows)
```
RMBG\Scripts\activate
```
## Using conda
##### Create a python 3.10 conda env (you could also use virtualenv)
```
conda create -n RMBG python=3.10
conda activate RMBG
```
### Step 2: Git Clone BRIA-RMBG-2.0
```
git clone https://github.com/NeuralFalconYT/BRIA-RMBG-2.0-Demo.git
```
### Step 3: Move to BRIA-RMBG-2.0-Demo
```
cd BRIA-RMBG-2.0-Demo
```
### Step 4: Install dependencies
```
pip install -r requirements.txt
```
### Step 5: Check CUDA version (if needed)
```
nvcc --version
```
### Step 6: Install PyTorch and Torchvision with CUDA [pytorch.org](https://pytorch.org/get-started/locally/) 
Download according to your CUDA version.
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


### Step 7: Run the Application

#### Launch the Gradio Web Interface
To run the Gradio application with a local web interface, execute the following command in your terminal:

```bash
python app.py 
```
#### Enable Debug Mode or Share Mode
You can enable **debug mode** and **share mode** by using the following options:

1. **Debug Mode**: Enables debug output for easier debugging.
   ```bash
   python app.py --debug
   ```

2. **Share Mode**: Enables a public link to share your app with others.
   ```bash
   python app.py --debug --share
   ```

   The `--share` flag will generate a public URL that you can share with others, allowing them to access the app remotely.


#### Step 8: Deactivate the virtual environment when done
```
deactivate
```

## Credit:
[RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
**Note**:<br>
  RMBG v2.0 is available as a source-available model for non-commercial use.<br>
  Developed by: [BRIA AI](https://bria.ai/) <br> HuggingFace model page: [RMBG-2.0 Model](https://huggingface.co/briaai/RMBG-2.0) <br>License: [bria-rmbg-2.0](https://bria.ai/bria-huggingface-model-license-agreement/) 
  * The model is released under a Creative Commons license for non-commercial use.
  * Commercial use is subject to a commercial agreement with BRIA. Contact BRIA AI for more information.
