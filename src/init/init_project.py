import subprocess
import os
from ..config import Config

def run_command(command, description):
    print(f"\n🔧 {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed during: {description}")
        exit(3)
    print(f"✅ Done: {description}")

if __name__ == "__main__":
    steps = [
        ("python -m src.init.dump_XML_to_dir", "Extract meta-information from XML"),
        ("python -m src.init.split", "Split dataset into train/val sets"),
        ("python -m src.init.split_images", "Split images into word-level crops"),
        ("python -m src.init.generate_labels", "Generate labels from XML"),
    ]
    if not os.path.exists(Config.Paths.images_meta_info_path):
        print(f"❌ You should have the meta data of the dataset in the {Config.Paths.images_meta_info_path} directory")
        exit(1)

    if not os.path.exists(Config.Paths.original_images_path):
        print(f"❌ You should have the dataset in the {Config.Paths.original_images_path} directory")
        exit(2)

    for cmd, desc in steps:
        run_command(cmd, desc)

    print("\n🎉 Project initialization complete!")