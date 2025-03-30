#!/usr/bin/env python3
"""
End-to-End Video Training Pipeline

This script:
  1. Captions videos using caption_videos.py.
  2. Preprocesses the dataset using preprocess_dataset.py.
  3. Loads an existing YAML training configuration, updates key parameters 
     (such as file paths, output directory, resolution buckets, training token, 
     and video dimensions), and writes the updated config to a timestamped folder.
  4. Runs training using train.py with the updated configuration.

If no caption output file is specified, it will be automatically saved as 
"captions.json" in the dataset folder and then used for preprocessing.
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
    Update the training YAML configuration.
    
    Updates include:
      - data.preprocessed_data_root (override if provided, otherwise defaults to a subfolder in dataset_dir)
      - validation.video_dims (from --video_dims or derived from resolution_buckets)
      - data.training_token (set to id_token)
      - output_dir (set to a unique timestamped folder)
    """
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set preprocessed data root (override or default to dataset_dir/.precomputed)
    if args.preprocessed_data_root:
        config['data']['preprocessed_data_root'] = args.preprocessed_data_root
    else:
        config['data']['preprocessed_data_root'] = os.path.join(args.dataset_dir, ".precomputed")

    # Set a unique output directory for training
    config['output_dir'] = training_output_dir

    # Update video dimensions if provided, or derive from resolution_buckets if possible
    if args.video_dims:
        try:
            dims = [int(x) for x in args.video_dims.split('x')]
            if len(dims) != 3:
                raise ValueError("video_dims must be in WxHxF format.")
            config['validation']['video_dims'] = dims
        except Exception as e:
            print("Error parsing --video_dims:", e)
    else:
        # Fallback to using resolution_buckets (if in WxHxF format)
        try:
            dims = [int(x) for x in args.resolution_buckets.split('x')]
            if len(dims) == 3:
                config['validation']['video_dims'] = dims
        except Exception as e:
            print("Error parsing resolution_buckets for video_dims:", e)

    # Update training token in data section
    config['data']['training_token'] = args.id_token

    # Save the updated configuration to a new YAML file in the training output folder.
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
        description="End-to-end training: caption videos, preprocess dataset, update YAML config, and run training."
    )
    # Required: dataset folder path
    parser.add_argument("dataset_dir", type=str, help="Path to the folder containing videos")
    
    # Optional output file for captions (if not given, will use dataset_dir/captions.json)
    parser.add_argument("--captions_output", type=str, default=None,
                        help="File path for saving captions JSON. Default: dataset_dir/captions.json")

    # Optional base directory for training outputs
    parser.add_argument("--output_dir_base", type=str, default="outputs", help="Base directory for training outputs")

    # Captioning parameters
    parser.add_argument("--captioner_type", type=str, default="llava_next_7b", help="Type of captioner to use")

    # Preprocessing parameters
    parser.add_argument("--caption_column", type=str, default="caption", help="Caption column name")
    parser.add_argument("--video_column", type=str, default="media_path", help="Video path column name")
    parser.add_argument("--id_token", type=str, default="T1m3l4ps3", help="Training token for preprocessing")
    parser.add_argument("--resolution_buckets", type=str, default="768x768x25", help="Resolution buckets in WxHxF format")

    # YAML config parameters
    parser.add_argument("--config_path", type=str, default="configs/ltxv_2b_lora.yaml", help="Path to the YAML training config")
    parser.add_argument("--preprocessed_data_root", type=str, default=None, help="Override for data.preprocessed_data_root in YAML")
    parser.add_argument("--video_dims", type=str, default="768x768x89", help="Override for validation.video_dims in WxHxF format (e.g., 768x768x89)")

    args = parser.parse_args()

    # Set default captions_output if not provided: save in dataset folder as captions.json
    if args.captions_output is None:
        args.captions_output = os.path.join(args.dataset_dir, "captions.json")

    # Create a unique timestamped folder for training outputs.
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
