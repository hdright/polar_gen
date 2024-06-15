
# Polar Generator
This repository is used to initially generate datasets of ResNet for blind recognition of polar codes.

## Features

- **Mode Selection**: Allows for the selection between training and testing modes.
- **SNR Range Configuration**: Enables the specification of the Signal-to-Noise Ratio (SNR) range for dataset generation.

## Configuration

The script includes several key configuration options:

- `reco_mode`: Determines the recognition mode.      Options include 'len', and 'prate'.
- `trainortest`: Specifies whether the dataset is for training ('train') or testing ('test').
- `no_samples_total`: The total number of samples to generate.      This value adjusts based on the mode and conditions.
- `snr_range`: The range of SNR values for which the dataset is generated.      Configurable to suit various testing and training needs.
- `N`: The base value for N, configurable based on the reconstruction mode.

## Usage

To use this script, simply configure the parameters at the top of the script to suit your dataset generation needs.      The script will then generate a dataset based on these parameters, suitable for use in simulations.
