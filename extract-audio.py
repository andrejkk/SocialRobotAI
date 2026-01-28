import argparse
import os
from moviepy import AudioFileClip

def main():
    parser = argparse.ArgumentParser(description="Convert .webm video to .wav audio file")
    parser.add_argument("recording_id", help="Recording DB id")
    parser.add_argument("uID", help="User id, for example: 66001")
    parser.add_argument("date", help="Date, for example: 2025-12-11")
    parser.add_argument("sID", help="Recording id, for example: S1")
    args = parser.parse_args()

    video_dir = f"./Data/{args.uID}/{args.date}/{args.sID}"
    # Find the first .webm file in the directory
    webm_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.webm')]
    if not webm_files:
        print(f"No .webm file found in {video_dir}")
        return
    video_path = os.path.join(video_dir, webm_files[0])
    wav_path = os.path.splitext(video_path)[0] + ".wav"

    print(f"Converting {video_path} to {wav_path} ...")
    audio_clip = AudioFileClip(video_path)
    audio_clip.write_audiofile(wav_path)
    print(f"Saved audio to {wav_path}")

if __name__ == "__main__":
    main()
