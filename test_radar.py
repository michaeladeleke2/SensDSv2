import numpy as np
from core.radar import RadarStream
from core.processing import compute_spectrogram

def on_frame(frame):
    spect = compute_spectrogram(frame)
    print(f"Spectrogram shape={spect.shape}, min={spect.min():.1f}dB, max={spect.max():.1f}dB")

def on_error(msg):
    print(f"Error: {msg}")

stream = RadarStream(on_frame=on_frame, on_error=on_error)
stream.start()

input("Press Enter to stop...\n")
stream.stop()
print("Stopped cleanly.")