# %%
import os

import matplotlib.pyplot as plt
import wandb
from src.generate import generate_and_decode_next_patch
batch_size = 10  # The number of sequences to process at once, it has no impact on the model, just on training speed
max_length = 100
patch_size = 4  # The number of spatiotemporal points per embedding so actual amount of float in embedding is patch_size * 4 for lat,lon,daily_timestamp,weekday
embedding_size = 16
layers = 4
heads = 2
accumulation_steps = 50


wandb.login(key=os.environ["WANDB_API_KEY"])
# %%
from src.utils.prepare_dataset import prepare_dataset, transform_float_list_back_to_trajectory

prepare_dataset("data", "prepared_data", scale=True, training_ratio=0.8)

# %%
from src.dataset import create_dataloader

train_dataloader = create_dataloader(
    prepared_data_folder="prepared_data/train",
    batch_size=batch_size,
    max_length=max_length,
    stride=max_length,
    # The stride is the step size for generating sequences, it should be equal to max_length for non-overlapping sequences
    drop_last=True,
    patch_size=patch_size,
    num_workers=4,
)

eval_dataloader = create_dataloader(
    prepared_data_folder="prepared_data/test",
    batch_size=1,
    max_length=max_length,
    stride=max_length,
    drop_last=True,
    patch_size=patch_size,
    num_workers=4,
)
# %%
import torch
from src.model import CheminGPT

"""train(
    device: torch.device,
    learning_rate: float,
    patch_size: int,
    model: CheminGPT,
    train_dataloader,
    eval_dataloader,
    num_epochs=10,
    save_path=None,
)"""

model = CheminGPT(
    patch_size=patch_size,
    embedding_size=embedding_size,
    num_layers=layers,
    n_heads=heads,
    max_length=max_length,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)


# %%
from src.train import train_model

model = train_model(
    device=torch.device("cuda"),
    learning_rate=1e-4,
    patch_size=patch_size,
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_steps=1000,
    eval_every=10,
    accumulation_steps=accumulation_steps,
    save_path="model.pt",
)

for data_in in eval_dataloader:
    data_in = data_in
    for i in range(len(data_in)):
        data_in[i] = data_in[i].to(device)

        total_traj =[]

        for j in range(len(data_in[i][0])):
            total_traj += (transform_float_list_back_to_trajectory(data_in[i][0][j]))

        print(transform_float_list_back_to_trajectory(data_in[i][0][-1]))
        out = generate_and_decode_next_patch(
            model=model,
            input_sequence=data_in[i],
            device=device,
            patch_size=patch_size,
        )
        print(out.shape)
        print(total_traj)
        print(transform_float_list_back_to_trajectory(out[0]))

        plt.plot(
            [x[0] for x in total_traj],
            [x[1] for x in total_traj],
            label="Input Trajectory",
        )
        plt.plot(
            [x[0] for x in transform_float_list_back_to_trajectory(out[0])],
            [x[1] for x in transform_float_list_back_to_trajectory(out[0])],
            label="Predicted Trajectory",
        )
        plt.legend()
        plt.show()
        input()
        continue
