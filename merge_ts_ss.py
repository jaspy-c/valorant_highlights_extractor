import os
import sys
import subprocess
import shutil

def run_command(command):
    """Run a command using subprocess and exit if it fails."""
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Command failed:", " ".join(command))
        print(result.stderr)
        sys.exit(1)
    return result

def parse_map_transitions(file_path):
    """Parse map transition timestamps from file."""
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, "r") as f:
        transitions = [float(line.strip()) for line in f if line.strip().replace('.', '', 1).isdigit()]
    return transitions

def convert_image_to_ts(image_path, ts_output_path, duration=3):
    """
    Convert an image to a TS segment with a 1920x1080 black background.
    The image is not scaled and is centered. The output will last for the specified duration.
    """
    command = [
        "ffmpeg",
        "-y",                    # force overwrite
        "-f", "lavfi",          # use lavfi for silent audio
        "-i", f"anullsrc=r=48000:cl=stereo:d={duration}",  # generate silent audio
        "-loop", "1",            # loop the input
        "-framerate", "60",      # input framerate
        "-i", image_path,        # input image
        "-t", str(duration),     # duration
        "-vf", "pad=1920:1080:(1920-iw)/2:(1080-ih)/2:color=black",  # center on black background
        "-r", "60",              # output framerate
        "-pix_fmt", "yuv420p",   # pixel format
        "-c:v", "libx264",       # video codec
        "-c:a", "aac",          # audio codec
        "-profile:v", "high",    # high profile for better quality
        "-preset", "ultrafast",  # faster encoding
        "-b:v", "3000k",        # video bitrate
        "-shortest",            # end when shortest input ends
        "-f", "mpegts",          # output format
        ts_output_path
    ]
    run_command(command)

def main():
    clips_dir = "highlights"
    screenshots_dir = "screenshots"
    output_file = "final_highlights.mp4"
    temp_dir = "ts_files"

    # Create a temporary directory for TS files
    os.makedirs(temp_dir, exist_ok=True)

    transitions = parse_map_transitions("map_transitions.txt")
    if len(transitions) < 1:
        print("No transitions found in map_transitions.txt.")
        sys.exit(1)
    valid_transitions = transitions[1:]
    print("Using valid transitions (ignoring first):", valid_transitions)

    total_screenshots = len(valid_transitions) + 1

    mp4_files = sorted(
        [f for f in os.listdir(clips_dir) if f.lower().endswith(".mp4")],
        key=lambda x: float(os.path.splitext(x.split("_")[3])[0])
    )

    if not mp4_files:
        print("No highlight files found in directory:", clips_dir)
        sys.exit(1)

    ts_files = []
    screenshot_index = 0

    # Process each highlight in order
    for mp4 in mp4_files:
        mp4_path = os.path.join(clips_dir, mp4)
        parts = mp4.split("_")
        if len(parts) < 4:
            print(f"Filename {mp4} doesn't match expected pattern.")
            continue
        try:
            clip_timestamp = float(os.path.splitext(parts[3])[0])
        except ValueError:
            print(f"Cannot parse timestamp from {mp4}")
            continue

        # Insert screenshots before this highlight if needed
        while screenshot_index < len(valid_transitions) and clip_timestamp >= valid_transitions[screenshot_index]:
            screenshot_filename = f"map_{screenshot_index+1}_screenshot.png"
            screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
            if os.path.exists(screenshot_path):
                ts_screenshot = os.path.join(temp_dir, f"{screenshot_filename}.ts")
                print(f"Inserting screenshot {screenshot_filename} before highlight {mp4}")
                convert_image_to_ts(screenshot_path, ts_screenshot, duration=3)
                ts_files.append(ts_screenshot)
            else:
                print(f"Screenshot file {screenshot_path} not found.")
            screenshot_index += 1

        # Convert the current highlight to TS format
        ts_path = os.path.join(temp_dir, f"{os.path.splitext(mp4)[0]}.ts")
        command = [
            "ffmpeg",
            "-y",  # force overwrite
            "-i", mp4_path,
            "-c", "copy",
            "-bsf:v", "h264_mp4toannexb",
            "-f", "mpegts",
            ts_path
        ]
        run_command(command)
        ts_files.append(ts_path)

    # Add final screenshot if needed
    while screenshot_index < total_screenshots:
        screenshot_filename = f"map_{screenshot_index+1}_screenshot.png"
        screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
        if os.path.exists(screenshot_path):
            ts_screenshot = os.path.join(temp_dir, f"{screenshot_filename}.ts")
            print(f"Inserting final screenshot {screenshot_filename}")
            convert_image_to_ts(screenshot_path, ts_screenshot, duration=3)
            ts_files.append(ts_screenshot)
        else:
            print(f"Screenshot file {screenshot_path} not found.")
        screenshot_index += 1

    # Add the map_all screenshot at the very end
    map_all_path = os.path.join(screenshots_dir, "map_all_screenshot.png")
    if os.path.exists(map_all_path):
        ts_map_all = os.path.join(temp_dir, "map_all_screenshot.ts")
        print("Inserting final map_all screenshot")
        convert_image_to_ts(map_all_path, ts_map_all, duration=3)
        ts_files.append(ts_map_all)
    else:
        print("Warning: map_all_screenshot.png not found in screenshots directory")

    # Concatenate all TS files into the final video
    concat_input = "concat:" + "|".join(ts_files)
    command = [
        "ffmpeg",
        "-y",  # force overwrite
        "-i", concat_input,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        output_file
    ]
    run_command(command)
    print(f"Finished! Final video saved as {output_file}")

    # Cleanup temporary TS files
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()