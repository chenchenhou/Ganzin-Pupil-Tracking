# Computer Vision Final Project: Pupil Tracking

Computer Vision (NTU EEE 5053, Spring 2023)

Instructor: Prof. Shao-Yi Chien

## Prepare dependencies and dataset

1. Install dependencies.

```bash
pip install -r requirement.txt
```

2. Download the data arrange it according to the following structure.

```bash
dataset
├── S1
│   ├── subfolders (01, 02, ...)
├── S2
├── S3
├── S4
├── S5
├── S6
├── S7
├── S8
```

## Inference

1. Download our pre-trained weights of DeepLabV3 [here](https://drive.google.com/drive/folders/1hMi-NeT1JfuIAxB2KHboP0lQusQS8gCa?usp=share_link).
2. Create ./checkpoints/ directory and put the pre-trained weight in it.
3. Run the inference script by entering the following command:

```bash
bash inference.sh
```

3. Wait for the script to finish running. It will create two directories: ./solution_original/ and ./solution_gamma/.

   - The ./solution_original/ directory will contain the predicted segmentation masks without gamma correction.
   - The ./solution_gamma/ directory will contain the predicted segmentation masks with gamma correction.

4. Ensemble the predicted masks from ./solution_original/ and ./solution_gamma/.

```bash
python merge.py
```

5. Incorporate the pupil masks into fully formed elliptical shapes.

```bash
python find_ellipse.py
```

6. Apply refinement on confidence scores.

```bash
python conf_refinement.py
```

7. Now the final segmentation masks and confidence scores should be stored in ./soultion/.
8. Zip the ./solution/ directory into a zip file.

```bash
python zipSolution.py
```
