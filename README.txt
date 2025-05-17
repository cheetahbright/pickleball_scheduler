Pickleball Scheduler Pro+ - Setup Instructions
-------------------------------------------------

To run this Pickleball Scheduler, please follow these steps:

**Step 1: Install Python (If you don't have it already)**

1.  Go to: https://www.python.org/downloads/windows/
2.  Download the latest Python 3.x installer for Windows (e.g., Python 3.11 or 3.12, 64-bit).
3.  Run the installer.
4.  **VERY IMPORTANT:** On the first screen of the installer, **CHECK THE BOX** that says "Add python.exe to PATH" or "Add Python 3.x to PATH".
5.  You can then click "Install Now" for the default installation.
6.  After installation, open Command Prompt (search "cmd" in Windows Start) and type `python --version`. You should see the Python version. If not, Python was not added to PATH correctly, and you may need to do it manually (see advanced troubleshooting).

**Step 2: Install Streamlit**

1.  Open Command Prompt (if not already open).
2.  Type the following command and press Enter:
    pip install streamlit
3.  Wait for the installation to complete.
4.  (Optional test) In Command Prompt, type `streamlit --version`. You should see the Streamlit version. If you get a "not recognized" error, Python's Scripts directory might not be in your PATH (see advanced troubleshooting).

**Step 3: Run the Scheduler**

1.  Ensure this README.txt file, `app.py`, and `start_scheduler.bat` are all in the same folder.
2.  Double-click the `start_scheduler.bat` file.
3.  A black command prompt window will appear.
4.  Your web browser should automatically open to the Pickleball Scheduler application.
    (If not, the command prompt window might show a URL like http://localhost:8501 - manually open this in your browser).
5.  **IMPORTANT:** Keep the black command prompt window open while you are using the scheduler. You can minimize it. Closing it will stop the scheduler.

**Step 4: Stopping the Scheduler**

1.  When you are finished, close the web browser tab with the scheduler.
2.  Go to the black command prompt window, click inside it, and press `Ctrl + C` (hold down the Ctrl key and press C).
3.  It might ask "Terminate batch job (Y/N)?". Type `Y` and press Enter.
4.  You can then close the command prompt window.

**Advanced Troubleshooting (If "python" or "streamlit" is not recognized):**

If you get errors like "'python' is not recognized..." or "'streamlit' is not recognized...", it usually means the necessary folders were not added to your system's PATH environment variable.

1.  Find your Python installation directory. This is often:
    *   `C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python3X\` (e.g., Python311)
    *   Or `C:\Program Files\Python3X\`
2.  You need to add TWO paths to your system PATH:
    *   The main Python directory (e.g., `C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python311\`)
    *   The Python `Scripts` directory (e.g., `C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python311\Scripts\`)
3.  Search for "environment variables" in Windows Start, click "Edit the system environment variables", click "Environment Variables...", find "Path" under "System variables" or "User variables", click "Edit...", click "New", and add each path.
4.  Close and reopen any Command Prompt windows for the changes to take effect. Then try Step 2 and 3 again.

-------------------------------------------------