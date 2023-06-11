import zipfile
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_dir", help="Path to solution directory.", type=str, default="./solution/")
    parser.add_argument("--zip_file", help="Path to result zip file.", type=str, default="./solution.zip")
    return parser


parser = get_parser()
args = parser.parse_args()

solution_dir = args.solution_dir
zip_file = args.zip_file


def zip_directory(directory_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))


# Usage example
zip_directory(solution_dir, zip_file)
