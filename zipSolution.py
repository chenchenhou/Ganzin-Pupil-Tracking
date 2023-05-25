import zipfile
import os


def zip_directory(directory_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))


# Usage example
directory_to_zip = "/home/yuchien/Ganzin-Pupil-Tracking/solution"
zip_file_path = "./solution.zip"
zip_directory(directory_to_zip, zip_file_path)
