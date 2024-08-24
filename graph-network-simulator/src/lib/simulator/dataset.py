import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import torch
import torch_geometric as pyg

FEATURES = ["cell_type"] # , "cell_density_micron3"]
DIMS = ["position_x", "position_y"] #, 'position_z'


DEFAULT_METADATA = dict(
    window_length=5,
    timestep=2.0,
    radius=50,
    noise_std=1e-8,
    bounds=[[-1000, 1000], [-1000, 1000]],
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)


def generate_noise(position_seq, noise_std):
    """Generate noise for a trajectory"""
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    time_steps = velocity_seq.size(1)
    velocity_noise = torch.randn_like(velocity_seq) * (noise_std / (time_steps ** 0.5 + 1e-12))
    velocity_noise = velocity_noise.cumsum(dim=1)
    position_noise = velocity_noise.cumsum(dim=1)
    position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
    return position_noise


def preprocess(cell_features, input_seq, boundary, target_seq=None, radius=1, noise_std=0.0):
    """Preprocess a trajectory and construct the graph"""

    # apply noise to the trajectory
    position_noise = generate_noise(input_seq, noise_std)
    input_seq += position_noise

    # calculate the velocities of cells
    recent_position = input_seq[:, -1]
    velocity_seq = input_seq[:, 1:] - input_seq[:, :-1]

    # construct the graph based on the distances between cells
    n_cell = recent_position.size(0)
    edge_index = pyg.nn.radius_graph(
        recent_position, radius,
        loop=True, max_num_neighbors=n_cell
    )

    # node-level features: velocity, distance to the boundary
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = (distance_to_lower_boundary, distance_to_upper_boundary)
    distance_to_boundary = torch.cat(distance_to_boundary, dim=-1)
    distance_to_boundary = torch.clip(distance_to_boundary / radius, -1.0, 1.0)
    # velocity_seq = (velocity_seq - torch.tensor(metadata["vel_mean"])) / torch.sqrt(torch.tensor(metadata["vel_std"]) ** 2 + noise_std ** 2)

    # edge-level features: displacement, distance
    dim = recent_position.size(-1)
    edge_displacement = torch.gather(
        recent_position, dim=0,
        index=edge_index[0].unsqueeze(-1).expand(-1, dim)
    )
    edge_displacement -= torch.gather(
        recent_position, dim=0,
        index=edge_index[1].unsqueeze(-1).expand(-1, dim)
    )
    edge_displacement /= radius
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # ground truth for training
    labels = None
    if target_seq is not None:
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_seq - recent_position
        next_velocity += position_noise[:, -1]
        acceleration = next_velocity - last_velocity
        # acceleration = (acceleration - torch.tensor(metadata["acc_mean"])) / torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2)
        labels = acceleration  # torch.cat((acceleration, ), dim=-1)

    return pyg.data.Data(
        x=cell_features,
        y=labels,
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        pos=torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1),
    )


class OneStepDataset(pyg.data.Dataset):
    def __init__(self, data, features=FEATURES, dims=DIMS, metadata=DEFAULT_METADATA, return_pos=False):
        self.data = data
        self.features = features
        self.dims = dims

        self.window_length = metadata["window_length"]
        self.radius = metadata["radius"]
        self.bounds = metadata["bounds"]
        self.timestep = metadata["timestep"]
        self.noise_std = metadata["noise_std"]
        # self.device = metadata["device"]

        self.return_pos = return_pos

        self.unique_cells = self.data.cell_id.unique()
        self.windows = self._get_windows()
        super().__init__()

    def len(self):
        return len(self.windows)

    def _get_windows(self):
        trajectories = self.data.traj.unique()
        windows = []
        for traj_id in trajectories:
            traj = self.data[self.data.traj == traj_id]
            traj_steps = traj.time.max()
            for idx in range(int(traj_steps/self.timestep - self.window_length + 1)):
                start = idx * self.timestep
                end = min((idx + self.window_length) * self.timestep, traj_steps)
                window = traj[(traj.time >= start) & (traj.time < end)]
                windows.append(window)
        return windows

    def _prepare_data(self, window):
        timesteps = window.time.unique()
        cell_features = window[self.features].values
        cell_features.resize(self.unique_cells.size, timesteps.size)

        position_seq = window[self.dims].astype("float32").values
        position_seq.resize(timesteps.size, self.unique_cells.size, len(self.dims))
        position_seq = position_seq.transpose(1, 0, 2)

        return cell_features, position_seq

    def get(self, idx):
        window = self.windows[idx]
        cell_features, position_seq = self._prepare_data(window)

        cell_features = torch.from_numpy(cell_features)  # .to(self.device, non_blocking=True)
        position_seq = torch.from_numpy(position_seq)  # .to(self.device, non_blocking=True)

        input_seq = position_seq[:, :-1]
        target_seq = position_seq[:, -1]

        with torch.no_grad():
            boundary = torch.tensor(self.bounds)  # .to(self.device, non_blocking=True)
            graph = preprocess(
                cell_features, input_seq, boundary, target_seq,
                radius=self.radius,
                noise_std=self.noise_std
            )

        if self.return_pos:
            return graph, target_seq

        return graph


class RolloutDataset(OneStepDataset):
    def __init__(self, data, features=FEATURES, dims=DIMS, metadata=DEFAULT_METADATA):
        super().__init__(
            data, features=features, dims=dims,
            return_pos=False,
            metadata=metadata,
      )

    def _get_windows(self):
        trajectories = []
        for traj_id in self.data.traj.unique():
            traj = self.data[self.data.traj == traj_id]
            trajectories.append(traj)
        return trajectories

    def get(self, idx):
        window = self.windows[idx]
        cell_features, position_seq = self._prepare_data(window)

        cell_features = torch.from_numpy(cell_features)  # .to(self.device, non_blocking=True)
        position_seq = torch.from_numpy(position_seq)  # .to(self.device, non_blocking=True)

        return {"cell_type": cell_features, "position": position_seq}