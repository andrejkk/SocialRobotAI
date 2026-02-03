# %%
# IMPORTS

import os
from moviepy import AudioFileClip
import librosa
import numpy as np




# %%
# Convert video (.webm) to audio format (.wav)

UID = '66001' # User id, for example: 66001
DATE = '2025-12-11' # Date, for example: 2025-12-11
SID = 'S1' # Recording id, for example: S1

video_dir = f"./Data/{UID}/{DATE}/{SID}"
# Find the first .webm file in the directory
webm_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.webm')]

if not webm_files:
    print(f"No .webm file found in {video_dir}")
    
video_path = os.path.join(video_dir, webm_files[0])
wav_path = os.path.splitext(video_path)[0] + ".wav"

print(f"Converting {video_path} to {wav_path} ...")
audio_clip = AudioFileClip(video_path)
audio_clip.write_audiofile(wav_path)
print(f"Saved audio to {wav_path}")




# %%
#
AUDIO_PATH = './Data/66001/2025-12-11/S1/output.wav'
y, sr = librosa.load(AUDIO_PATH, sr=None)




# %%


times = np.arange(len(y)) / sr
times

# %%
# Segment audio into time windowss
# You never work with the full waveform directly.
frame_length = int(0.025 * sr)  # 25 ms
hop_length = int(0.010 * sr)    # 10 ms overlap

frames = librosa.util.frame(
    y,
    frame_length=frame_length,
    hop_length=hop_length
).T


# %%
