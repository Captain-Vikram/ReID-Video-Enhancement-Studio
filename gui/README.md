# GUI Directory

This directory contains all the graphical user interface files for the ReID Video Enhancement Studio.

## Files:

- **app.py** - Main Streamlit application
- **launch_gui.py** - GUI launcher with dependency checking  
- **start_gui.bat** - Windows batch file launcher

## Running the GUI:

### From the project root:
```bash
python launch_gui.py
# or double-click start_gui.bat on Windows
```

### From this gui directory:
```bash
python launch_gui.py
# or
streamlit run app.py
```

## File Organization:

The GUI files have been moved to this dedicated folder to keep the project structure clean and organized. The application automatically adjusts import paths to access the main project modules (`src/`, `models/`, etc.) from the parent directory.

## Dependencies:

All GUI dependencies are automatically checked and installed when using the launcher scripts.
