import os
import torch

from tqdm import tqdm

from dataset import DEFAULT_METADATA, preprocess


def oneStepMSE(simulator, dataloader, metadata=DEFAULT_METADATA):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        scale = 1  # torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise ** 2) #.cuda()
        for data in dataloader:
            data = data.to(metadata["device"], non_blocking=True)
            pred = simulator(data)
            mse = ((pred - data.y) * scale) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((pred - data.y) ** 2).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count, total_mse / batch_count


def rollout(model, data, metadata=DEFAULT_METADATA):
    model.eval()
    window_size = model.window_size + 1
    total_time = data["position"].size(1)

    traj = data["position"][:, :window_size, :]
    cell_type = data["cell_type"][:, :1]

    for time in range(total_time - window_size):
        with torch.no_grad():
            boundary = torch.tensor(metadata["bounds"])  # .to(metadata["device"], non_blocking=True)
            graph = preprocess(
                cell_type, traj[:, -window_size:], boundary,
                noise_std=0.0
            )
            graph = graph.to(metadata["device"], non_blocking=True)
            acceleration = model(graph)
            # acceleration = acceleration * torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2) + torch.tensor(metadata["acc_mean"])
            recent_position = traj[:, -1]
            recent_velocity = recent_position - traj[:, -2]
            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity
            traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)  # .to(metadata["device"], non_blocking=True)

    return traj


def rolloutMSE(simulator, dataset, metadata=DEFAULT_METADATA):
    total_loss = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        for rollout_data in dataset:
            rollout_data = rollout_data.to(metadata["device"], non_blocking=True)
            rollout_out = rollout(simulator, rollout_data)
            loss = (rollout_out - rollout_data["position"]) ** 2
            loss = loss.sum(dim=-1).mean()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count


def train(params, simulator, train_loader, valid_loader=None, rollout_loader=None, model_path="./", metadata=DEFAULT_METADATA):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))

    # recording loss curve
    train_loss_list = []
    eval_loss_list = []
    onestep_mse_list = []
    rollout_mse_list = []
    total_step = 0

    for i in range(params["epoch"]):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            data = data.to(metadata["device"], non_blocking=True)
            pred = simulator(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / batch_count, "lr": optimizer.param_groups[0]["lr"]})
            total_step += 1
            train_loss_list.append((total_step, loss.item()))

            # evaluation
            if valid_loader:
                if total_step % params["eval_interval"] == 0:
                    simulator.eval()
                    eval_loss, onestep_mse = oneStepMSE(simulator, valid_loader, metadata)
                    eval_loss_list.append((total_step, eval_loss))
                    onestep_mse_list.append((total_step, onestep_mse))
                    tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                    simulator.train()

            # do rollout on valid set
            if rollout_loader:
                if total_step % params["rollout_interval"] == 0:
                    simulator.eval()
                    rollout_mse = rolloutMSE(simulator, rollout_loader, metadata)
                    rollout_mse_list.append((total_step, rollout_mse))
                    tqdm.write(f"\nEval: Rollout MSE: {rollout_mse}")
                    simulator.train()

            # save model
            if total_step % params["save_interval"] == 0:
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(model_path, f"checkpoint_{total_step}.pt")
                )
    return train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list