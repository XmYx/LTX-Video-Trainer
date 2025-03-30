#!/usr/bin/env python3
"""
End-to-end video training pipeline.

This script performs the following steps:
1. Captions the videos using caption_videos.py.
2. Preprocesses the dataset using preprocess_dataset.py.
3. Updates the YAML training configuration with user-specified parameters such as
   the preprocessed data root, output directory (timestamped), video dimensions, and training token.
4. Runs the training using train.py with the updated config.
"""

import argparse
import subprocess
import os
import sys
import datetime
import yaml

def run_captioning(args, captions_output):
    """Run the captioning step."""
    caption_cmd = [
        sys.executable, os.path.join("scripts", "caption_videos.py"),
        args.dataset_dir,
        "--output", captions_output,
        "--captioner-type", args.captioner_type
    ]
    print("Running captioning:")
    print(" ".join(caption_cmd))
    subprocess.run(caption_cmd, check=True)

def run_preprocessing(args, captions_output):
    """Run the preprocessing step."""
    preprocess_cmd = [
        sys.executable, os.path.join("scripts", "preprocess_dataset.py"),
        captions_output,
        "--caption-column", args.caption_column,
        "--video-column", args.video_column,
        "--id-token", args.id_token,
        "--resolution-buckets", args.resolution_buckets
    ]
    print("Running preprocessing:")
    print(" ".join(preprocess_cmd))
    subprocess.run(preprocess_cmd, check=True)

def update_yaml_config(args, training_output_dir):
    """
    Open the original YAML config, update file paths and parameters, and save to a new file.
    Updates include:
      - data.preprocessed_data_root (either from an override or derived from dataset_dir)
      - validation.video_dims (if provided via --video_dims or using the resolution_buckets)
      - data.training_token (set to the provided id_token)
      - output_dir is updated to the unique training output folder.
    """
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the preprocessed data root.
    if args.preprocessed_data_root:
        config['data']['preprocessed_data_root'] = args.preprocessed_data_root
    else:
        # Default: use a subfolder named ".precomputed" inside the dataset directory.
        config['data']['preprocessed_data_root'] = os.path.join(args.dataset_dir, ".precomputed")

    # Update the output directory to our unique timestamped folder.
    config['output_dir'] = training_output_dir

    # Update video dimensions for validation.
    # If a specific video_dims is provided (format: WxHxF), use it.
    if args.video_dims:
        try:
            dims = [int(x) for x in args.video_dims.split('x')]
            if len(dims) != 3:
                raise ValueError("video_dims must be in WxHxF format.")
            config['validation']['video_dims'] = dims
        except Exception as e:
            print("Error parsing --video_dims:", e)
    else:
        # Otherwise, attempt to parse resolution_buckets (e.g. "768x768x25")
        try:
            dims = [int(x) for x in args.resolution_buckets.split('x')]
            if len(dims) == 3:
                config['validation']['video_dims'] = dims
        except Exception as e:
            print("Error parsing --resolution_buckets for video_dims:", e)

    # Optionally update training token in the config (store it in the data section).
    config['data']['training_token'] = args.id_token

    # Save the updated configuration to a new YAML file inside the training output folder.
    updated_config_filename = os.path.basename(args.config_path).replace('.yaml', '_updated.yaml')
    updated_config_path = os.path.join(training_output_dir, updated_config_filename)
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Updated YAML config saved to: {updated_config_path}")
    return updated_config_path

def run_training(updated_config_path):
    """Run the training step using the updated YAML configuration."""
    train_cmd = [
        sys.executable, os.path.join("scripts", "train.py"),
        updated_config_path
    ]
    print("Running training:")
    print(" ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end video training pipeline: caption, preprocess, update config, and train."
    )
    # Required dataset/video folder.
    parser.add_argument("dataset_dir", type=str, help="Path to folder containing the videos")

    # Output configuration for captions and training.
    parser.add_argument("--output_dir_base", type=str, default="outputs", help="Base directory for training outputs")
    parser.add_argument("--captions_output", type=str, default=None, help="File path to save captions JSON")

    # Captioning parameters.
    parser.add_argument("--captioner_type", type=str, default="llava_next_7b", help="Type of captioner to use")

    # Preprocessing parameters.
    parser.add_argument("--caption_column", type=str, default="caption", help="Name of the caption column")
    parser.add_argument("--video_column", type=str, default="media_path", help="Name of the video path column")
    parser.add_argument("--id_token", type=str, default="T1m3l4ps3", help="Token used for preprocessing")
    parser.add_argument("--resolution_buckets", type=str, default="768x768x25", help="Resolution buckets in WxHxF format")

    # YAML configuration parameters.
    parser.add_argument("--config_path", type=str, default="configs/ltxv_2b_lora.yaml", help="Path to the YAML training config")
    parser.add_argument("--preprocessed_data_root", type=str, default=None, help="Override for data.preprocessed_data_root in YAML")
    parser.add_argument("--video_dims", type=str, default=None, help="Override for validation.video_dims (format: WxHxF, e.g. 768x768x89)")

    args = parser.parse_args()

    # Set default captions_output if not provided.
    if args.captions_output is None:
        args.captions_output = os.path.join(args.dataset_dir, "captions.json")

    # Create a unique timestamped folder for training output.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    training_output_dir = os.path.join(args.output_dir_base, f"train_{timestamp}")
    os.makedirs(training_output_dir, exist_ok=True)
    print(f"Training output directory: {training_output_dir}")

    # Step 1: Run captioning.
    run_captioning(args, args.captions_output)

    # Step 2: Run preprocessing.
    run_preprocessing(args, args.captions_output)

    # Step 3: Update the YAML config.
    updated_config_path = update_yaml_config(args, training_output_dir)

    # Step 4: Run training.
    run_training(updated_config_path)

if __name__ == '__main__':
    main()
