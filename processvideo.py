import os
import random
import subprocess
import json

def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "format=duration", "-of", "json", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])

def run_ffmpeg(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg Error:", result.stderr.decode())
        raise RuntimeError("FFmpeg command failed.")

def process_video(input_path, replacement_path, output_path, clip_duration=8):
    video_duration = get_video_duration(input_path)

    if video_duration <= clip_duration:
        raise ValueError("Video is too short for this operation.")

    # Randomly choose a start time for the segment to delete
    t_delete = random.uniform(0, video_duration - clip_duration)
    print(f"Replacing segment from {t_delete:.2f}s to {t_delete + clip_duration:.2f}s")

    # Temp output files
    tmp_before = "tmp_before.mp4"
    tmp_after = "tmp_after.mp4"
    tmp_concat = "concat.txt"

    # Extract part before the deleted segment
    run_ffmpeg(f"ffmpeg -y -ss 0 -t {t_delete} -i {input_path} -c copy {tmp_before}")

    # Extract part after the deleted segment
    run_ffmpeg(f"ffmpeg -y -ss {t_delete + clip_duration} -i {input_path} -c copy {tmp_after}")

    # Create concat list
    with open(tmp_concat, 'w') as f:
        f.write(f"file '{tmp_before}'\n")
        f.write(f"file '{replacement_path}'\n")
        f.write(f"file '{tmp_after}'\n")

    # Concatenate all parts
    run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {tmp_concat} -c copy {output_path}")

    # Cleanup
    for f in [tmp_before, tmp_after, tmp_concat]:
        if os.path.exists(f):
            os.remove(f)

# === Example Usage ===
input_video = "output_video_test.mp4"
replacement_clip = "2440175990.mp4"  # Must be exactly 8 seconds long
output_video = "output_modified.mp4"

process_video(input_video, replacement_clip, output_video)
