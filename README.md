# U-Net microscopy image segmentation

An end-to-end PyTorch project for binary segmentation of cell regions in grayscale microscopy images. The workflow uses the public **BBBC019v2** wound-healing dataset and an assignment-style U-Net with unpadded convolutions, 64 starting channels, transpose-convolution upsampling, and cropped skip connections.

The official BBBC019v2 page catalogs **171 manually segmented DIC microscopy images** divided among eight source datasets: Init, SN15, Melanoma, TScratch, Scatter, Microfluidics, HEK293, and MDCK. The completed run in this repository used **165 verified image-mask pairs**: the downloaded Init data contributed 22 pairs rather than the 28 images listed on the catalog page. The saved run metadata records the exact counts used.

The workflow uses a reproducible **source-disjoint** split. The model is evaluated on microscopy sources that never appear during training, preventing related images from the same source experiment from leaking across splits:

| Split         | Sources                        | Images |
| ------------- | ------------------------------ | -----: |
| Training      | Init, Melanoma, SN15, TScratch |    120 |
| Validation    | Microfluidics, Scatter         |     19 |
| Held-out test | HEK293, MDCK                   |     26 |

The official dataset is downloaded directly by the notebook and is excluded from Git. BBBC019v2 is provided by the Broad Bioimage Benchmark Collection under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/). Use the citation requested on the [official BBBC019 page](https://bbbc.broadinstitute.org/BBBC019).

## Run in Google Colab

1. Open [`BBBC019_UNet_segmentation.ipynb`](BBBC019_UNet_segmentation.ipynb) in Google Colab.
2. Choose **Runtime > Change runtime type > GPU**.
3. Run every cell in order. The notebook downloads and validates all eight official archives.
4. Training runs for up to 40 epochs with learning-rate reduction and early stopping. Validation Dice selects the checkpoint.
5. The final cells evaluate the held-out test set once and download `bbbc019_portfolio_results.zip`.
6. Extract the zip in the repository root. It creates a `bbbc019/` results directory that can be committed to GitHub.

Colab normally includes all required packages. If its runtime reports a missing import, run:

```python
%pip install torch numpy Pillow matplotlib
```

The exported portfolio package contains:

- Aggregate test loss, accuracy, Dice, IoU, precision, and recall
- Mean per-image Dice and a CSV with every image's metrics
- Separate metrics for the held-out HEK293 and MDCK sources
- Training and validation history plus a loss chart
- A prediction preview for every held-out image
- The exact split and training configuration
- A generated Markdown experiment report with dataset attribution

## Run with Python scripts

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python bbbc019.py --data-dir data/BBBC019 --output splits/bbbc019.json
python train.py --data-dir data/BBBC019 --split splits/bbbc019.json --output-dir runs/bbbc019 --epochs 40 --batch-size 3
python test.py --data-dir data/BBBC019 --split splits/bbbc019.json --checkpoint runs/bbbc019/best_model.pt --output-dir results/bbbc019 --batch-size 3
```

The preparation command downloads the official archives, safely extracts them, matches each image with its `_manual.png` mask, verifies dimensions and source counts, and writes the source-disjoint split. The completed run used 120/19/26 images. Pass `--split-strategy stratified` only when you specifically want an easier in-domain experiment; do not compare its score directly with the default cross-source result.

## Model and preprocessing

- Grayscale images normalized to `[0, 1]`
- Aspect-ratio-preserving square padding followed by 572 x 572 resizing
- Nearest-neighbor interpolation for masks
- Horizontal and vertical flips, 90-degree rotations, zoom, and gamma augmentation on training samples only
- Two valid 3 x 3 convolutions per block
- 64-channel first block and 1,024-channel bottleneck
- Four downsampling and four upsampling stages
- Cropped skip connections and two-class output logits
- 388 x 388 prediction from a 572 x 572 input
- Adam optimizer, cross-entropy loss, validation-based checkpointing, learning-rate scheduling, and early stopping

## Repository structure

| Path                                                                  | Purpose                                                                |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [`BBBC019_UNet_segmentation.ipynb`](BBBC019_UNet_segmentation.ipynb) | Standalone Colab portfolio workflow                                    |
| [`bbbc019.py`](bbbc019.py)                                           | Official download, validation, and source-disjoint splitting           |
| [`preprocessing.py`](preprocessing.py)                               | Generic paired-image dataset, padding, normalization, and augmentation |
| [`model.py`](model.py)                                               | Assignment-style U-Net                                                 |
| [`train.py`](train.py)                                               | Training, scheduling, early stopping, and checkpoint creation          |
| [`test.py`](test.py)                                                 | Held-out evaluation and prediction export                              |
| [`metrics.py`](metrics.py)                                           | Aggregate, per-image, and per-source metrics                           |
| [`reporting.py`](reporting.py)                                       | Charts, CSV files, and generated experiment report                     |
| [`tests/`](tests/)                                                   | Dataset, preprocessing, reporting, and model checks                    |

## BBBC019 results

The completed Colab run stopped after 16 of the requested 40 epochs. Epoch 8 was selected by validation Dice. Training took 453.6 seconds on a CUDA GPU, or about 7 minutes 34 seconds.

| Held-out metric     |           Result |
| ------------------- | ---------------: |
| Cross-entropy loss  |           0.3803 |
| Pixel accuracy      |           0.8579 |
| Aggregate Dice      | **0.8884** |
| Mean per-image Dice |           0.8897 |
| IoU                 |           0.7993 |
| Precision           |           0.8588 |
| Recall              |           0.9202 |

| Held-out source | Images |   Dice |    IoU | Precision | Recall |
| --------------- | -----: | -----: | -----: | --------: | -----: |
| HEK293          |     12 | 0.8835 | 0.7914 |    0.7945 | 0.9951 |
| MDCK            |     14 | 0.8922 | 0.8053 |    0.9149 | 0.8706 |

![Training and validation loss with held-out test loss](bbbc019/loss_curves.png)

Representative predictions, shown as input, ground truth, and prediction:

| Strong example: Dice 0.9776                                                   | Weak MDCK example: Dice 0.6577                                 |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------- |
| ![Strong HEK293 prediction](bbbc019/HEK293_Yaniv__dic_L23_Sum001_preview.png) | ![Weak MDCK prediction](bbbc019/MDCK_DKWH7_L2_003_preview.png) |

The per-image Dice scores range from 0.6577 to 0.9776. HEK293 recall is very high while precision is lower, indicating over-segmentation on some images. The weakest MDCK predictions have high precision and low recall, indicating under-segmentation. The validation loss is unstable after the selected epoch, so the early-stopped checkpoint should be used instead of the final epoch.

The full generated report, all 26 prediction previews, raw metrics, per-image CSV, training history, and configuration are in [`bbbc019/`](bbbc019/README.md). This is a credible cross-source portfolio baseline, but the test set contains only two held-out sources and 26 images; it should not be presented as a state-of-the-art benchmark result.

For a credible portfolio result, report the aggregate Dice together with mean per-image Dice and separate HEK293/MDCK performance. BBBC019 contains materially different acquisition conditions, so a single aggregate score can hide weak performance on a particular source.

## Previous 38-image BMMC baseline

The earlier run is retained as a baseline. It used a fixed 30/4/4 split, trained for 15 epochs, and selected epoch 15 by validation Dice.

| Held-out metric     | Result |
| ------------------- | -----: |
| Cross-entropy loss  | 0.4171 |
| Pixel accuracy      | 0.8189 |
| Aggregate Dice      | 0.8290 |
| Mean per-image Dice | 0.7888 |
| IoU                 | 0.7080 |

The test set contained only four images and per-image Dice ranged from 0.4290 to 0.9746, so this baseline should not be presented as evidence of broad generalization. Its complete artifacts remain under [`results/results/`](results/results/README.md).

## Verification

Run the local checks with:

```bash
python -m unittest discover -s tests -v
```

Dataset preparation and report-generation tests run without PyTorch. The U-Net forward-shape test runs when PyTorch is installed.
