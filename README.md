<p><img width="128" height="128" alt="icon" src="https://github.com/user-attachments/assets/5bb130fa-7ea5-4077-84d8-9864007bb427" /></p>
Morse Code Bitmap Burst Encoder (MCB2E)

A simple python app that is able to take any image, encode it into a grayscale bitmap and encode it digitally using ITU morse code at high speeds (optimal burst speed - 10k WPM+)

## Dependencies
 
### Running the program (`gui.py`, `core_part.py`)
 
| Dependency | Used for |
|---|---|
| Python 3.9+ | Runs both scripts. |
| [numpy](https://numpy.org/) | All signal processing and image conversion. |
| [PySide6](https://pypi.org/project/PySide6/) | The Qt6 interface. Uses QtCore, QtGui, QtWidgets and QtMultimedia (microphone / audio input). The standard `PySide6` package includes all of these. |
| [FFmpeg](https://ffmpeg.org/download.html) | Checked at startup only. If `ffmpeg` is not on `PATH` (or next to the program), the app shows a download popup and closes. The program does not call FFmpeg itself. |
 
Everything else comes from the Python standard library (`math`, `os`, `re`, `struct`, `sys`, `shutil`, `traceback`, `collections`).
 
Live listening additionally needs an audio input that delivers **stereo** audio: Left = I, Right = Q (for example SDR software in raw / binaural IQ demod mode).
 
```
pip install PySide6 numpy
```
 
### Building the exe (`build_exe.bat`)
 
| Dependency | Used for |
|---|---|
| [PyInstaller](https://pyinstaller.org/) | Packs the scripts into a single windowed `.exe`. |
| [Pillow](https://pypi.org/project/pillow/) | Lets PyInstaller turn `icon.png` into the exe icon. |
 
The built exe bundles Python and all libraries above, but still needs FFmpeg on the target PC because of the startup check.
 
### Optional
 
| Dependency | Used for |
|---|---|
| numpy | `test_core.py` round-trip tests (no other requirements). |
