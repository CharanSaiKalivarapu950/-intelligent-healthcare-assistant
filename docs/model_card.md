# Model Card: Chest X-ray Classification (Experimental)

## Intended use
Educational experimentation with a public, labeled chest X-ray dataset.

## Not intended for
Diagnosis, treatment decisions, triage, or use on real patients.

## Model
ResNet18 transfer learning; two dataset classes (NORMAL, PNEUMONIA).

## Data
Document dataset source, license, class distribution, split method, and known limitations.

## Evaluation
Report class-wise precision, recall, F1, confusion matrix, ROC-AUC, and test setup.

## Limitations
- Dataset may contain source-specific artifacts and biases.
- Dataset labels are not equivalent to clinical ground truth.
- Generalization to other populations, devices, hospitals, and image protocols is unknown.
- Confidence values are not calibrated unless calibration is explicitly tested.
- Grad-CAM highlights model-associated regions but does not establish causal reasoning.
