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

<img width="1917" height="1021" alt="image" src="https://github.com/user-attachments/assets/20903c02-6d01-4e8d-9294-c62831c4e77c" />
<img width="1917" height="1018" alt="image" src="https://github.com/user-attachments/assets/89a6a580-7a4c-4a8c-9d32-44f981340258" />
<img width="1917" height="1017" alt="image" src="https://github.com/user-attachments/assets/fd687eb1-6335-404e-b5bf-c0114363d848" />
<img width="1917" height="1021" alt="image" src="https://github.com/user-attachments/assets/ed3f773b-9cef-4cbf-8689-ba4ef863cb0f" />
<img width="1917" height="1017" alt="image" src="https://github.com/user-attachments/assets/322ff822-abd9-4073-a6ed-52364111e402" />
<video src="https://github.com/user-attachments/assets/d876666c-d37e-4a5d-b403-7b88d2b17c89" controls width="100%"></video>

## Encode Tab

- **Image support:** Opens PNG, JPG, BMP, GIF, and WebP images. Fits the entire image into a square canvas, letterboxes it in white, and converts it to grayscale.
- **Resolutions:** Supports 20 × 20, 40 × 40, 80 × 80, 120 × 120, and 240 × 240 pixels.
- **Pixel encoding:** Each pixel is represented by one of 36 symbols (`A–Z`, `0–9`). Every symbol is transmitted five times, allowing the decoder to correct errors through majority voting.
- **Callsign:** Supports callsigns of up to 16 letters or digits, with `N0CALL` as the default.
- **Header:** Transmits the callsign three times and the resolution twice, allowing the decoder to identify the correct image dimensions automatically.
- **Synchronization markers:** Uses `=` characters, which never appear in pixel data. Markers are 6, 5, or 4 characters long depending on transmission speed. Silence before and after each burst also decreases at higher speeds.
- **Transmission speed:** Supports speeds from 5 WPM to 150,000 WPM, with the corresponding LPM displayed beneath the WPM input.
- **WAV export:** Supports speeds up to 45,000 WPM, subject to the 4 GiB file-size limit.
- **RF64 PCM export:** Removes the 4 GiB file-size limit and supports speeds up to 150,000 WPM.
- **Speed validation:** Displays a warning in the information box when a requested speed is out of range instead of silently clamping the value.
- **Automatic sample-rate adjustment:** If the WPM exceeds the configured sample rate, the sample rate is increased automatically, and a warning displays the new rate.
- **Signal quality warnings:** Warns when a Morse dot occupies only a couple of samples and when the resulting file is exceptionally large.
- **IQ output:** Exports 16-bit stereo IQ audio, with the left channel carrying I and the right channel carrying Q. Includes adjustable tone offset and edge shaping from 0 to 50 ms.
- **Transmission information:** Displays the character count, sample rate, synchronization marker length, silence duration, total transmission duration, and estimated file size.
- **Demodulation bandwidth:** Provides separate minimum, recommended, and clean-edge bandwidth estimates for the selected transmission speed.
- **Background processing:** Runs exports in the background and displays progress through a progress bar.

## Decode Tab

- **File import:** Supports stereo WAV and RF64 files containing 8-bit, 16-bit, 24-bit, or 32-bit PCM audio, as well as 32-bit or 64-bit floating-point audio.
- **Live input:** Allows selection of an audio input device and real-time monitoring of incoming data. Designed for SDR software operating in raw binaural IQ mode. Maintains a rolling 10-minute buffer and progressively reconstructs the image as data arrives.
- **Automatic detection:** Automatically detects the carrier at any frequency offset, ignores SDR DC spikes, identifies the burst start, and estimates the Morse transmission speed without requiring manual configuration.
- **Header-first decoding:** Displays the callsign, transmission speed, and image resolution as soon as the header is received, before decoding the image.
- **Image reconstruction and error report:** Reconstructs pixel data using majority voting across the five repetitions. The report identifies mismatched groups, uncertain groups, and invalid symbols.
- **Decoding options:** Allows manual resolution overrides and provides a **Save PNG** option for the decoded image.
- **Spectrum display:** Displays the detected tone and highlights the required demodulation bandwidth.
- **Bandwidth information:** Shows the required demodulation bandwidth for the detected transmission speed in the Header panel.
- **Header validation:** Rejects garbled or invalid headers instead of displaying them as legitimate callsigns.

## Interface and Packaging

- **Visual design:** Features a glossy Windows Aero-inspired interface with a dark/light mode toggle.
- **Text handling:** Truncates long text with `…` rather than allowing it to stretch the window.
- **FFmpeg detection:** Checks for FFmpeg at startup. If FFmpeg is missing, displays a popup containing the download link and closes the application when the popup is dismissed.
- **Executable packaging:** Uses `build_exe.bat` to package the application into a single, no-console executable, using `icon.png` as the application icon.

## Known Limitations

- **No resynchronization:** The decoder does not resynchronize after a dropped or inserted character. A single character error shifts the remaining image data out of alignment. This is particularly noticeable with very noisy transmissions, extremely high speeds, and large images.
- **High-speed accuracy:** At 45,000 WPM and 48 kHz, a Morse dot occupies only approximately 1.3 samples. This makes speed estimation somewhat inaccurate, although the transmission decoded cleanly in testing.
- **Untested microphone input:** Live microphone listening has not been tested because audio hardware was unavailable during development.
- **Stereo IQ requirement:** Live listening requires a stereo input with the I and Q components supplied on separate channels.






