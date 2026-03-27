from core.radar import RadarStream

def on_frame(frame):
    print(f"Got frame: shape={frame.shape}, dtype={frame.dtype}")

def on_error(msg):
    print(f"Error: {msg}")

stream = RadarStream(on_frame=on_frame, on_error=on_error)
stream.start()

input("Press Enter to stop...\n")
stream.stop()
print("Stopped cleanly.")