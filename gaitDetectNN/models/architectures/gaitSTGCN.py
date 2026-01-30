import torch
import torch.nn as nn
from gaitDetectNN.utils.registry import MODELS
from skeletons import get_skeleton_by_name


class GraphConvolution(nn.Module):
    """ Basic GCN layer: Z = A * X * W """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1))

    def forward(self, x, A):
        # x: (B, C, T, V)
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # Graph convolution: 'nkctv,kvw->nctw'
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return x.contiguous()


class STGCNBlock(nn.Module):
    """ Spatial-Temporal Block: GCN + TCN """

    def __init__(self, in_channels, out_channels, kernel_size, A_shape, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, A_shape[0])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, (kernel_size[1], 1), (stride, 1),
                      padding=((kernel_size[1] - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return x + res


@MODELS.register
class GaitSTGCN(nn.Module):
    def __init__(self, input_size, features_config, skeleton_name, num_classes=4,
                 hidden_channels=64, num_layers=3, dropout=0.2, tcn_kernel_size=9, strategy="uniform", **kwargs):
        super().__init__()

        raw_keypoints = features_config.get('keypoints', [])
        raw_kinematics = features_config.get('kinematics', [])

        if not raw_keypoints:
            raise ValueError("Config musí obsahovat 'keypoints'.")

        # Sort Keypoints
        self.node_names = sorted(raw_keypoints)
        self.V = len(self.node_names)

        if input_size % self.V != 0:
            raise ValueError(f"Input size {input_size} does not match the number of nodes {self.V}.")

        self.C = input_size // self.V

        # Reconstruct feature order
        # We are basically simulating the ordering process of preprocessing.py
        simulated_feature_names = []

        # Keypoints (_x, _y)
        for kp in raw_keypoints:
            simulated_feature_names.extend([f"{kp}_x", f"{kp}_y"])

        # Kinematics (_vx, _vy, _ax, _ay)
        if raw_kinematics:
            for kp in raw_kinematics:
                if kp in raw_keypoints:
                    simulated_feature_names.extend([f"{kp}_vx", f"{kp}_vy", f"{kp}_ax", f"{kp}_ay"])

        # Final sort
        input_feature_order = sorted(simulated_feature_names)

        # print(f"\n[DEBUG MODEL] Total: {len(input_feature_order)}")
        # print(f"[DEBUG MODEL] First 5: {input_feature_order[:5]}")
        # print(f"[DEBUG MODEL] Last 5:  {input_feature_order[-5:]}")

        if len(input_feature_order) != input_size:
            raise ValueError(f"Number of features ({len(input_feature_order)}) does not match input_size ({input_size}).")

        # Create permutation index
        permutation_indices = []
        for node_name in self.node_names:
            node_features = [f for f in input_feature_order if f.startswith(f"{node_name}_")]
            node_features = sorted(node_features)  # (ax, ay, vx, vy, x, y)

            if len(node_features) != self.C:
                raise ValueError(f"Attribute error for node {node_name}.")

            indices = [input_feature_order.index(f) for f in node_features]
            permutation_indices.extend(indices)

        self.register_buffer('perm_idx', torch.tensor(permutation_indices, dtype=torch.long))



        # Build graph
        skeleton = get_skeleton_by_name(skeleton_name)
        adj_list = skeleton.get_adjacency_list(self.node_names)

        # Build Spatial Graph (K=3)

        if strategy == "uniform":
            A = self._build_adjacency_matrix(adj_list, self.V)

        elif strategy == "spatial":
            A = self._build_spatial_adjacency_matrix(adj_list, self.V, self.node_names)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.register_buffer('A', A)

        # Architecture
        self.data_bn = nn.BatchNorm1d(input_size)
        self.layers = nn.ModuleList()
        self.layers.append(STGCNBlock(self.C, hidden_channels, (1, tcn_kernel_size), A.size(), stride=1, dropout=dropout))

        for _ in range(1, num_layers):
            self.layers.append(
                STGCNBlock(hidden_channels, hidden_channels, (1, tcn_kernel_size), A.size(), stride=1, dropout=dropout))

        self.fcn = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def _build_adjacency_matrix(self, adj_list, num_nodes):
        A = torch.zeros((num_nodes, num_nodes))
        for i, j in adj_list:
            A[i, j] = 1
            A[j, i] = 1

        # Normalize: D^-1 * (A + I)
        A_with_loop = A + torch.eye(num_nodes)
        D = torch.sum(A_with_loop, dim=1)
        D_inv = torch.pow(D, -1)
        D_inv[torch.isinf(D_inv)] = 0.
        A_norm = torch.einsum('i,ij->ij', D_inv, A_with_loop)
        return A_norm.unsqueeze(0)  # (1, V, V)

    def _get_hop_distance(self, num_node, adj_list, center_idx):
        """ Calculate distance to center """
        dist = [-1] * num_node
        dist[center_idx] = 0
        queue = [center_idx]

        neighbors = {i: [] for i in range(num_node)}
        for i, j in adj_list:
            neighbors[i].append(j)
            neighbors[j].append(i)

        while queue:
            u = queue.pop(0)
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def _build_spatial_adjacency_matrix(self, adj_list, num_nodes, node_names):
        """ Creates Spatial Graph (K=3): Root, Centripetal, Centrifugal """

        # Find center point
        center_keywords = ['SpineBase', 'Hip', 'Pelvis', 'Torso', 'MidHip']
        center_idx = 0  # Default fallback
        found = False
        for kw in center_keywords:
            for idx, name in enumerate(node_names):
                if kw in name:
                    center_idx = idx
                    found = True
                    break
            if found: break

        if not found:
            print(f"[ST-GCN Warning] Could not detect center node from names. Using first node: {node_names[0]}")

        # Distance from center node
        dist = self._get_hop_distance(num_nodes, adj_list, center_idx)

        # Split to 3 partitions
        # K=3: 0=self, 1=closer (centripetal), 2=further (centrifugal)
        A = torch.zeros((3, num_nodes, num_nodes))

        # Partition 0: Self-loops
        indices = torch.arange(num_nodes)
        A[0, indices, indices] = 1

        # Partition 1 & 2: Neighbor connections
        for i, j in adj_list:
            if dist[j] < dist[i]:
                A[1, i, j] = 1  # Neighbor is closer -> Centripetal
            elif dist[j] > dist[i]:
                A[2, i, j] = 1  # Neighbor is further -> Centrifugal
            # dist equal -> cycle -> ignore

            if dist[i] < dist[j]:
                A[1, j, i] = 1
            elif dist[i] > dist[j]:
                A[2, j, i] = 1

        # Row Normalization
        for k in range(3):
            A_k = A[k]
            row_sum = A_k.sum(dim=1)
            row_sum[row_sum == 0] = 1e-6  # Avoid div by zero
            # D^-1 * A
            A[k] = A_k / row_sum.unsqueeze(1)

        return A  # (3, V, V)

    def forward(self, x, lengths):
        # x z collate_pad: (Batch, Time, Features)

        # Permute to (Batch, Features, Time) for BatchNorm and the rest of the network
        x = x.permute(0, 2, 1)

        B, F, T = x.shape
        x = self.data_bn(x)

        # Order nodes based on the calculated index
        x = x[:, self.perm_idx, :]

        # Reshape to ST-GCN format:
        # (B, V*C, T) -> (B, V, C, T) -> (B, C, T, V)
        x = x.view(B, self.V, self.C, T)
        x = x.permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            x = layer(x, self.A)

        # Global Pool & Classify
        x = torch.mean(x, dim=3, keepdim=True)  # Pool nodes -> (B, Hidden, T, 1)
        x = self.fcn(x)  # Conv1x1 -> (B, Classes, T, 1)

        # Back to (B, T, Classes) to match the expected output shape
        return x.squeeze(-1).permute(0, 2, 1)
