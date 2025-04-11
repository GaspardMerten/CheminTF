import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.generate import generate_and_decode_next_patch


def train_model(
    device: torch.device,
    learning_rate: float,
    patch_size: int,
    model,
    train_dataloader,
    eval_dataloader,
    max_steps: int = 10_000,
    accumulation_steps: int = 4,
    log_every: int = 10,
    eval_every: int = 50,
    save_path: str = None,
    project_name: str = "trajectory-llm",
    run_name: str = None,
):
    os.makedirs("checkpoints", exist_ok=True)

    if save_path and os.path.exists(save_path):
        print("🔁 Loading existing model weights.")
        model.load_state_dict(torch.load(save_path, map_location=device))
        return model

    # ✅ Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "learning_rate": learning_rate,
            "patch_size": patch_size,
            "accumulation_steps": accumulation_steps,
            "max_steps": max_steps,
        },
    )

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    model.train()
    global_step = 0
    epoch = 0
    total_loss = 0.0

    print(f"🚀 Starting step-based training for {max_steps} steps...")
    torch.cuda.reset_peak_memory_stats(device)

    train_iter = iter(train_dataloader)
    pbar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        optimizer.zero_grad()

        for _ in range(accumulation_steps):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                epoch += 1
                inputs, targets = next(train_iter)
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        pbar.update(1)

        if global_step % log_every == 0:
            avg_loss = total_loss / log_every
            total_loss = 0.0
            current_mem = torch.cuda.memory_allocated(device) / 1024**2
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2

            tqdm.write(
                f"🧠 Step {global_step}/{max_steps} | Loss: {avg_loss:.4f} | "
                f"Mem: {current_mem:.2f} MB | Peak: {peak_mem:.2f} MB"
            )

            # ✅ Log to wandb
            wandb.log({
                "train/loss": avg_loss,
                "gpu/mem": current_mem,
                "gpu/peak": peak_mem,
            }, step=global_step)

        if global_step % 100 == 0:
            save_path = f"checkpoints/model_step_{global_step}.pt"
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': total_loss,
            }, save_path)
            tqdm.write(f"💾 Saved model checkpoint at step {global_step} to {save_path}")

        if global_step % eval_every == 0:
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            with torch.no_grad():
                eval_iter = iter(eval_dataloader)
                try:
                    for _ in range(10):  # Evaluate on 10 batches
                        eval_inputs, eval_targets = next(eval_iter)
                        eval_inputs, eval_targets = eval_inputs.to(device), eval_targets.to(device)
                        with autocast():
                            eval_outputs = model(eval_inputs)
                            batch_loss = criterion(eval_outputs, eval_targets)
                        eval_loss += batch_loss.item()
                        eval_batches += 1
                except StopIteration:
                    tqdm.write("⚠️  Evaluation data exhausted before 10 batches.")

                if eval_batches > 0:
                    avg_eval_loss = eval_loss / eval_batches
                    tqdm.write(f"📉 Eval Loss at step {global_step}: {avg_eval_loss:.4f}")
                    wandb.log({"eval/loss": avg_eval_loss}, step=global_step)
                else:
                    tqdm.write("⚠️  No evaluation batches available.")

                # Optional: one sample prediction
                try:
                    eval_sample = next(iter(eval_dataloader))
                    eval_inputs, _ = eval_sample
                    eval_inputs = eval_inputs.to(device)
                    predicted_patch = generate_and_decode_next_patch(
                        model, eval_inputs, device, patch_size
                    )
                    tqdm.write("📊 Eval sample prediction (lat, lon, t):")
                    tqdm.write(str(predicted_patch))
                    wandb.log({"eval/sample_patch": str(predicted_patch)})
                except StopIteration:
                    tqdm.write("⚠️  No data in evaluation set.")

            model.train()

    pbar.close()
    print(f"\n✅ Training complete at step {global_step}")

    if save_path:
        print(f"💾 Saving model weights to {save_path}")
        torch.save(model.state_dict(), save_path)
        wandb.save(save_path)  # ✅ Log artifact to wandb

    wandb.finish()
    return model
