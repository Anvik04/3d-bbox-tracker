import os
import sys

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synth_fixtures import generate_all_fixtures


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(repo_root, "data", "fixtures")
    print(f"Generating fixtures in {output_dir}...")
    generate_all_fixtures(output_dir, num_frames=10)
    print("Done!")


if __name__ == "__main__":
    main()
