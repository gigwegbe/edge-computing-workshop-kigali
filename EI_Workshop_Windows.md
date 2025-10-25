- *\[OPTIONAL\] Open Terminal: Press the “Win” button and search for “Terminal*  
  - *Get the Terminal for Windows 10 through the Windows Store if not available*

- Install Git for Windows (This would provide Git and Bash for Windows)  
  Link:   
- [https://git-scm.com/install/windows](https://git-scm.com/install/windows)  
    
- [https://github.com/git-for-windows/git/releases/download/v2.51.1.windows.1/Git-2.51.1-64-bit.exe](https://github.com/git-for-windows/git/releases/download/v2.51.1.windows.1/Git-2.51.1-64-bit.exe) (Recommended)  
  Use a downloader is advisable if download is slow …

- Select “(NEW\!) Add a Git Bash Profile to Windows Terminal” during Git Setup’s Select Components stage  
- Stick to the option already provided   
- Launch Git Bash OR Press the Windows button and search for “Git Bash

- *\[OPTIONAL\] Launch the Terminal Application*  
  - *Press the “downward-facing arrow” button besides the “plus” button and select “Git Bash”. You should now have have Git Bash opened in a window of the Terminal application. Same as opening the Git Bash application directly.*

- Clone the repository for the workshop via HTTPS using:  
  - cd \~/Desktop  
  - mkdir ei\_workshop  
  - cd ei\_workshop/  
  - git clone https://github.com/gigwegbe/edge-computing-workshop-kigali.git

- Install Python on Windows  
  - Type “python” into the Git Bash terminal  
  - Select “Get” to install Python 3.13 via Windows Store

- Create a Python Environment for this demo:  
  - Navigate to the working folder if “pwd” does not return ei\_workshop  
    - cd \~/Desktop/ei\_workshop  
  - Create virtual environment … this would create a folder with the venv name:  
    - python \-m venv ei\_workshop\_env  
  - Activate virtual environment:  
    - source ei\_workshop\_env/Scripts/activate

- Install the needed packages:  
  - pip install \-r requirements-vlm.txt

     		OR

- pip install numpy tensorflow Pillow opencv-python  
- Install Microsoft Visual C++ 2015-2022 Redistributable (x64)  
  Link: [https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170\#latest-supported-redistributable-version](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version)  
- pip install accelerate transformers torch   
- pip install “transformers\[torch\]”  
- pip install torchvision torchaudio \--index-url https://download.pytorch.org/whl/cpu

- Run the Scripts for image classification, object detection and visual description with VLM:  
  - Change directory using the “cd” command

  NB: Use the “python” alias when running scripts not “python3

