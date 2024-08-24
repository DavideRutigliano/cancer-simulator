import torch
import torch_scatter
import torch_geometric as pyg

from dataset import DEFAULT_METADATA

class InteractionNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = pyg.nn.models.MLP(
            in_channels=hidden_size * 3,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=layers,
            norm=None,
        )
        self.lin_node = pyg.nn.models.MLP(
            in_channels=hidden_size * 2,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=layers,
            norm=None,
        )

    def forward(self, x, edge_index, edge_feature):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)


class LearnedSimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size=128,
        n_mp_layers=10,
        num_cell_types=3,
        dim=2,
        window_size=DEFAULT_METADATA["window_length"],
        # cell_embedding_dim=16, # embedding dimension of cell types
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        # self.embed_type = torch.nn.Embedding(num_cell_types, cell_embedding_dim)
        self.node_in = pyg.nn.models.MLP(
            in_channels=dim * (window_size + 2) + 1,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=3,
            norm=None,
        )
        self.edge_in = pyg.nn.models.MLP(
            in_channels=dim + 1,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=3,
            norm=None,
        )
        self.node_out = pyg.nn.models.MLP(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            out_channels=dim,
            num_layers=3,
            norm="batch_norm",
        )
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([
            InteractionNetwork(hidden_size, 3)
            for _ in range(n_mp_layers)
        ])
        # self.reset_parameters()

    # def reset_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data):
        # pre-processing
        # node feature: combine categorial feature data.x and contiguous feature data.pos
        node_feature = torch.cat((data.x, data.pos), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)
        # post-processing
        out = self.node_out(node_feature)
        return out