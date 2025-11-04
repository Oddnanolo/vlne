# Training vlne Models on MCC9 Data

This tutorial describes how to train neural network energy estimators on the
MCC9 datasets. Instructions on how to prepare the dataset can be found by this
[link](https://github.com/usert5432/vlne_data_extractors_uboone).

## Prerequisites

Before starting the training, ensure you have:

1. **vlne package installed**: Follow the installation instructions in the
   [main vlne README](https://github.com/usert5432/vlne#installation)

2. **Environment variables configured**: Set up the required directory
   structure and environment variables as described in the vlne documentation.
   Specifically, you need:
   - `$VLNE_DATADIR` - Directory containing your training datasets
   - `$VLNE_OUTDIR` - Directory where trained models will be saved

3. **Training dataset prepared**: Once the dataset is prepared following
   [instructions](https://github.com/usert5432/vlne_data_extractors_uboone)
   it should be copied to:
```
   $VLNE_DATADIR/uboone/mcc9_v0/dataset_train_fc+pc_numu+nue_shuffled.hdf
```

## Training Scripts Overview

This directory contains several training scripts with different configurations:

- `final_v0.py` - Final production training configuration, use this if goal
    is to reproduce the energy estimator of the
    [paper](https://arxiv.org/abs/2406.10123)
- `test_train.py` - Quick test configuration for debugging
- `tune-*.py` - Various hyperparameter tuning configurations

Each script is self-contained and includes an inline configuration dictionary
that defines:
- Dataset paths and preprocessing
- Model architecture (LSTM-based)
- Training hyperparameters (batch size, learning rate, epochs, etc.)
- Output directory for saved models

To understand what a particular script does, simply open it and examine the
`config` dictionary. The configuration structure is intuitive and
well-organized.

## Running Training

To start training with a specific configuration, simply run:
```bash
python final_v0.py
```

The training process will:
1. Load and preprocess the dataset
2. Split data into training and validation sets (80/20 split by default)
3. Train the neural network
4. Save checkpoints and monitor validation performance
5. Apply early stopping when validation loss stops improving

Training progress will be displayed in the console, showing epoch number,
         training loss, and validation loss.

## Training Outputs

Once training completes, the trained model and associated files will be saved
under:
```
$VLNE_OUTDIR/uboone/mcc9/test/model_HASH/
```

where `HASH` is an MD5 hash of the training configuration, ensuring that each
unique configuration gets its own directory.

The output directory contains:

- `model.h5` - Trained neural network weights
- `config.json` - Complete training configuration
- `log.csv` - Training history (loss values per epoch)
- Additional diagnostic files

## Modifying Training Configuration

To customize the training, you can either:

1. **Edit an existing script**: Modify the `config` dictionary in any training
   script. Key parameters you might want to adjust:
   - `batch_size` - Batch size for training (default: 1024)
   - `epochs` - Maximum number of training epochs (default: 200)
   - `optimizer.lr` - Learning rate (default: 0.001)
   - `data.val_size` - Validation set fraction (default: 0.2)
   - `outdir` - Output directory for the trained model

2. **Create a new script**: Copy an existing script and modify the
   configuration as needed.

## Evaluating the Trained Model

After training completes, you can evaluate your model's performance using
vlne's evaluation tools and the evaluation datasets prepared in the
[data processing step](https://github.com/usert5432/vlne_data_extractors_uboone).

### Basic Energy Resolution Plot

To generate energy resolution histograms for your trained model:
```bash
python scripts/eval/eval_model.py \
    --preset uboone_numu --split test --weight none \
    --data $VLNE_DATADIR/uboone/mcc9_v0/dataset_eval_numu_fc.hdf \
    -e pdf \
    -- $VLNE_OUTDIR/uboone/mcc9/test/model_HASH/
```

where `model_HASH` is the directory created during training.

**Available presets:**
- `uboone_numu`    - For NuMu datasets
- `uboone_nue`     - For NuE datasets
- `uboone_generic` - For mixed datasets

**Evaluation datasets:** You can evaluate on any of the datasets created during
data processing:
- `dataset_eval_numu_fc.hdf`        - NuMu fully-contained events
- `dataset_eval_numu+nue_fc+pc.hdf` - Combined evaluation dataset
- Or any other merged evaluation dataset you created

The evaluation script will generate resolution plots showing how well the model
predicts lepton energy, hadronic energy, and total neutrino energy. Plots will
be saved under:
```
$VLNE_OUTDIR/uboone/mcc9/test/model_HASH/evals/
```

### Other Evaluation Scripts

vlne provides several other evaluation tools in the `scripts/eval/` directory:

- `make_binstat_plots.py` - Mean/RMS of energy resolution vs true energy
- Additional diagnostic scripts for detailed performance analysis

Run any evaluation script with `--help` to see available options:
```bash
python scripts/eval/eval_model.py --help
```

## Exporting Models for Production

Once you're satisfied with your model's performance, you can export it for use
in C++ analysis frameworks (e.g., for integration with NOvA's CAFAna or
MicroBooNE's analysis tools).

### Export to TensorFlow Graph

To export your trained model to a frozen TensorFlow graph:
```bash
python scripts/tf/export_model.py $VLNE_OUTDIR/uboone/mcc9/test/model_HASH/
```

This will create a frozen graph file that can be loaded and used in C++
applications. The exported model retains the same inference behavior as the
trained Keras model but in a format optimized for production deployment.

Refer to the export script's help for additional options:
```bash
python scripts/tf/export_model.py --help
```


