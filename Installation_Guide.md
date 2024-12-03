# Installation Guide

## Step 1: Install Python
1. Visit the official Python website: https://www.python.org/downloads/
2. Click the big yellow "Download Python" button (this will get the latest version)
3. Run the downloaded installer
4. **Important**: On the first installation screen, check the box that says "Add Python to PATH"
5. Click "Install Now"
6. Wait for the installation to complete and click "Close"

## Step 2: Install Git
1. Download Git from: https://git-scm.com/download/windows
2. Run the installer
3. You can keep all the default options during installation - just keep clicking "Next"
4. Click "Install" when you reach that screen
5. Click "Finish" when installation is complete

## Step 3: Download This Script
1. Open Command Prompt:
   - Press Windows key + R
   - Type `cmd` and press Enter
2. Navigate to where you want to download the script:
```
cd Desktop
```
3. Clone this repository:
```
git clone https://github.com/hezy/ODR-example.git
```
4. Enter the project directory:
```
cd ODR-example
```

## Step 4: Install Dependencies
1. Make sure you're in the project directory (from Step 3)
2. Install required packages:
```
pip install -r requirements.txt  
```

You're now ready to use the script! See the Usage section for how to run it.

## Troubleshooting
- If you get "'python' is not recognized": Restart your Command Prompt and try again
- If you get "'git' is not recognized": Restart your Command Prompt and try again
- If you get errors during pip install: Try running:
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt  
```
