# CheminTF: Transformer-based Foundation Model for Trajectory Prediction

## Repository for the Sigspatial 2025 tutorial: Building a Foundation Model for Trajectory from Scratch

CheminTF is an educational model repository for the SIGSPATIAL 2025 tutorial on building a foundation model for
trajectory prediction from scratch. It serves as a mockup model, designed to be a foundational basis for developing more
complex and specialized trajectory prediction models.

## Repository structure

The repository has two main directories:

- `src`: Contains the source code for the model, including data processing, model architecture, training scripts, and
  evaluation scripts. It is meant to be a reusable codebase for building trajectory prediction models.
- `notebooks`: Contains Jupyter notebooks that contains the same code as the `src` directory, but with additional
  explanations, visualizations, and is self-contained for educational purposes. It is meant to be a learning resource
  for understanding the model and its components.
- `weights`: Stores pre-trained model weights, allowing for quick deployment and evaluation without retraining.
- `Slides.pdf`: The slides used during the [SIGSPATIAL 2025 tutorial presentation](https://drive.google.com/file/d/1f71YpgT_Qs6g9lv0RgWphcp7FYL-jyYK/view). 
  You can also access them 
  [in Google slides](https://docs.google.com/presentation/d/1V9tzojY14bL9RbNIlVyHrkj47ZCgB1APsd9ZTROfZmY/edit?usp=sharing)
  to view the animations and videos.

## Opening the notebooks in Google Colab

In the notebooks directory, there is chemintf which is meant to be followed during the tutorial presentation, other
notebooks are meant to be used as a learning resource.

To open the main notebook in Google Colab, click on the following link
[Open CheminTF in Google Colab](https://colab.research.google.com/github/GaspardMerten/cheminTF/blob/main/notebooks/chemintf.ipynb).

## Source Code Modules (`src/`)

The `src` directory contains the Python modules that make up the CheminTF model.

* **`dataset.py`**: This module is responsible for generating and preparing the synthetic trajectory data used for
  training and evaluation. The `SyntheticTrajectoryDataset` class creates trajectories with random starting points,
  movements, and noise.

* **`evaluate.py`**: This module provides the `evaluate_and_plot` function to assess the performance of a trained model.
  It loads a model checkpoint, generates predictions on a validation set, and plots the results.

* **`synthetic_trajectory.py`**: This module contains the `SyntheticTrajectoryGenerator` class, which is the core of the
  data generation process. It creates a sequence of (latitude, longitude, timestamp) points that simulate a trajectory.

* **`train.py`**: This is the main training script. It orchestrates the training process, including:
    * Splitting the data into training and validation sets.
    * Setting up the model, loss function, and optimizer.
    * Running the training and validation loops.
    * Visualizing predictions at the end of each epoch.
    * Saving the trained model.

* **`modules/`**: This sub-directory contains the building blocks of the CheminTF model architecture.
    * **`constants.py`**: Defines global constants used across the project, such as embedding dimensions and feature
      sizes.
    * **`embedding.py`**: Implements the `SpatioTemporalEmbeddings`, which takes encoded spatial and temporal features and
      projects them into a high-dimensional embedding space.
    * **`features.py`**: Contains functions for feature engineering. `spatial_encoding` and
      `temporal_encoding` convert raw coordinates and timestamps into meaningful features for the model.
    * **`model.py`**: Defines the main `CheminTF` class, which assembles the complete Transformer-based architecture.
    * **`output.py`**: Implements the `OutputModule`, a linear layer that maps the final transformer hidden states to
      the output prediction (delta latitude, delta longitude, delta time).
    * **`position.py`**: Implements `PositionalEmbedding` to inject positional information into the input
      sequence, allowing the model to understand the order of the trajectory points.

## Notebooks (`notebooks/`)

The `notebooks` directory provides a hands-on, educational path to understanding the CheminTF model.

* **`chemintf.ipynb`**: This is the main, self-contained notebook for the tutorial. It includes all the code from the
  `src` directory, allowing you to run the entire project in a single environment like Google Colab.

* **Step-by-Step Tutorial Notebooks**: These notebooks break down the model into its core components, providing detailed
  explanations and visualizations for each part. They are designed to be followed in order:
    1. **`step_1_building_an_encoder.ipynb`**: Focuses on the feature extraction and encoding process.
    2. **`step_2_adding_gpt2_like_positional_encoding.ipynb`**: Explains and implements positional encoding.
    3. **`step_3_transformer_block.ipynb`**: Details the Transformer encoder architecture.
    4. **`step_4_working_model.ipynb`**: Assembles all the components into a complete, working model.

## Authors

This repository is developed and maintained by:

* [Gaspard Merten](https://github.com/GaspardMerten/)
* [Mahmoud Sakr](https://github.com/mahmsakr)
* [Gilles Dejaegere](https://github.com/gdejaege)

All authors are affiliated with the [Université libre de Bruxelles (ULB)](https://www.ulb.be).
