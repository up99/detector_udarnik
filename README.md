# 🛠️ Detector Udarnik

This repository contains the source files for the **Detector Udarnik** application.

## 🚀 How to Run the Application

There are two convenient ways to launch the program depending on your setup:

### 1. Using Python (All Platforms)

- First, install the required dependencies using the provided environment file:
  ```bash
  conda env create -f environment.yml
  ```
- Then, activate the environment:
  ```bash
  conda activate detector_udarnik
  ```
- Finally, run the app:
  ```bash
  python detector.py
  ```

### 2. Using a Standalone Executable (Windows Only)

For Windows users who prefer a standalone executable:

- Clone or download the repository.
- Build the `.exe` file using PyInstaller:
  ```bash
  pyinstaller detector.py --windowed --add-data "runs\detect\train3\weights\best.pt;runs\detect\train3\weights\best.pt" --clean
  ```

## 📁 Notes on Resources

The model file `best.pt` is included as part of the build process to ensure the application runs correctly with all necessary assets bundled.