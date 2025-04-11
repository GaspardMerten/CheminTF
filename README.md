# CheminTF: Transformer-based Foundation Model for AIS Trajectory Prediction

CheminTF is a transformer-based foundation model designed for AIS (Automatic Identification System) trajectory prediction and maritime mobility analysis. It uses a GPT-style architecture to learn patterns in vessel movement data and generate future trajectories.

## Features

- **Transformer Architecture**: Implements a GPT-style model with multi-head attention and positional embeddings
- **AIS Trajectory Processing**: Handles AIS trajectory data in patches, supporting various vessel attributes per point
- **Efficient Training**: Implements gradient accumulation and mixed precision training
- **Monitoring**: Integrated with Weights & Biases for experiment tracking
- **Memory Efficient**: Uses iterable datasets for handling large-scale AIS trajectory data

## Project Structure

```
src/
├── model.py          # Core transformer model implementation
├── dataset.py        # Data loading and preprocessing
├── train.py          # Training loop and utilities
├── generate.py       # Trajectory generation utilities
├── config.py         # Configuration parameters
└── utils/            # Utility functions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CheminTF.git
cd CheminTF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Prepare your AIS trajectory data in Parquet format with the following structure:
- Each file should contain a column named "trajectory"
- Each trajectory should be a flattened list of coordinates (latitude, longitude), timestamps, and optional vessel attributes
- The data should be placed in a `prepared_data` directory
- Example trajectory format: `[lat1, lon1, timestamp1, speed1, heading1, lat2, lon2, timestamp2, speed2, heading2, ...]`

### Training

```python
from src.model import CheminGPT
from src.dataset import create_dataloader
from src.train import train_model
import torch

# Initialize model
model = CheminGPT(
    patch_size=10,
    embedding_size=256,
    num_layers=6,
    n_heads=8,
    max_length=256
)

# Create dataloaders
train_dataloader = create_dataloader(
    batch_size=4,
    max_length=256,
    stride=128,
    patch_size=10
)

eval_dataloader = create_dataloader(
    batch_size=4,
    max_length=256,
    stride=128,
    patch_size=10
)

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_model(
    device=device,
    learning_rate=1e-4,
    patch_size=10,
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_steps=10000
)
```

### Generation

```python
from src.generate import generate_and_decode_next_patch

# Generate next trajectory patch
next_patch = generate_and_decode_next_patch(
    model=model,
    input_sequence=input_sequence,
    device=device,
    patch_size=10
)
```

## Model Architecture

The model consists of three main components:

1. **TrajectoryEmbedding**: Converts trajectory patches into embeddings
2. **GPTBlock**: Implements the transformer block with multi-head attention
3. **CheminGPT**: Main model combining embeddings and transformer blocks

## Configuration

Key configuration parameters can be adjusted in `config.py`:
- `VALUES_PER_POINT`: Number of values per trajectory point
- `NEUTRAL_VECTOR`: Default values for padding

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{CheminTF,
  author = {Gaspard Merten},
  title = {CheminTF: Transformer-based Foundation Model for Mobility},
  year = {2024},
  url = {https://github.com/GaspardMerten/CheminTF}
}
``` 