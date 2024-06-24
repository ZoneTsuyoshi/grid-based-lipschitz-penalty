# GLiP (Grid-Based Lipschitz Penalty)
This repository is dedicated to loss function GLiP proposed in the paper.

## Data
For ethical reasons, the data are not opened; however, you can try this method with the following data.

- DICOM data
    - DICOM data correspond to `dcd` extension and can be loaded by `pydicom` package.
    - Each DICOM file contains the CT image as `pixel_array`, the voxel spacing as `PixelSpacing` and `SliceThickness`, the origin of measurement as `ImagePositionPatient`.
- Landmark data
    - Landmark data correspond to `csv` extension.
    - The data includes path to DICOM file as `DICOMPath` column, landmarks as `[R,L,N]CC_[x,y,z]` columns, the grade of CT image as `CTQuality` column.
    - In our experiments, we prepare `spacing_[x,y,z]` and `origin_[x,y,z]` columns from each DICOM file.

## Environment
In our experiments, we use the following computational environment.

- Machine environment
    - OS: Red Hat 8.6
    - CUDA: v12.1
    - GPU: NVIDIA Tesla A40
- Python environment
    - Python 3.11.3
    - numpy 1.24.3
    - pandas 2.0.1
    - matplotlib 3.7.1
    - seaborn 0.11.0
    - scikit-learn 1.2.2
    - scipy 1.10.1
    - torch 2.0.1
    - pydicom 2.3.1 (for loading DICOM file)

## Citation
If you would like to cite this repository, please cite the following paper.

```bibtex
@article{glip,
    author = {},
    journal = {},
    pages = {},
    year = {},
}
```