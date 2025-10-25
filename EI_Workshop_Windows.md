# Setup Guide for Windows Users

### 0. *\[OPTIONAL\] Open Terminal: Press the “Win” button and search for “Terminal*  
  - *Get the Terminal for Windows 10 through the Windows Store if not available*

## 1. Install Git for Windows (This would provide Git and Bash for Windows) - [Direct Download Link](https://github.com/git-for-windows/git/releases/download/v2.51.1.windows.1/Git-2.51.1-64-bit.exe)

NB: Using a downloader like Free Downloader Manager is advisable if download speed is slow …
  - Select “(NEW!) Add a Git Bash Profile to Windows Terminal” during Git Setup’s Select Components stage  
  - Stick to the option already provided   
  - Launch Git Bash OR Press the Windows button and search for “Git Bash

### 1a. *\[OPTIONAL\] Launch the Terminal Application*  
  - *Press the “downward-facing arrow” button beside the “plus” button and select “Git Bash”. You should now have Git Bash opened in a window of the Terminal application. Same as opening the Git Bash application directly.*

## 2. Create a working directory in your Desktop Folder and clone the repository for the workshop:  
  ```bash
  cd ~/Desktop
  mkdir ei_workshop  
  cd ei_workshop/  
  git clone https://github.com/gigwegbe/edge-computing-workshop-kigali.git
  ```

## 3. Install Python on Windows  
  - Type “python” into the Git Bash terminal  
  - Select “Get” to install Python 3.13 via Windows Store

## 4. Create a Python Environment for this demo:  
  - Navigate to the working folder if “pwd” does not return ei\_workshop  
    ```bash
    cd ~/Desktop/ei_workshop
    ```
  - Create virtual environment … this would create a folder with the venv name:  
    ```bash
    python -m venv ei_workshop_env
    ```
  - Activate virtual environment:
    ```bash
    source ei_workshop_env/Scripts/activate
    ```

## 5. Install the needed packages in the virtual environment created:  
  ```bash
  pip install -r requirements-vlm.txt
  ```
  OR
  ```bash
  pip install numpy tensorflow Pillow opencv-python
  ```
  THEN
  ```bash
  pip install accelerate transformers torch
  pip install “transformers[torch]”  
  pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

## 6. Install Microsoft Visual C++ 2015-2022 Redistributable (x64) - [Download Page](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version)

## 7. Run the Scripts for image classification, object detection, and visual description with VLM:  
NB: Change directory using the “cd” command and use the “python” alias when running scripts not “python3

