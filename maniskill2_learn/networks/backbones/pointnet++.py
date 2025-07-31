import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNetPlusPlus, self).__init__()

        # Set abstraction layers
        self.sa1 = SetAbstraction(0.2, 0.2, 32, 6+3, [32, 32, 64], False)
        self.sa2 = SetAbstraction(0.4, 0.4, 64, 64+3, [64, 64, 128], True)
        self.sa3 = SetAbstraction(0.8, 0.8, 128, 128+3, [128, 128, 256], True)
        self.sa4 = SetAbstraction(1.6, 1.6, 256, 256+3, [256, 256, 512], True)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Set abstraction layers
        x, xyz1, points1 = self.sa1(x)
        x, xyz2, points2 = self.sa2(x)
        x, xyz3, points3 = self.sa3(x)
        x, xyz4, points4 = self.sa4(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Return logits
        return x

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint  # Number of centroids (points)
        self.radius = radius  # Radius of the local region to be considered
        self.nsample = nsample  # Number of nearest neighbors to query
        self.group_all = group_all  # If True, aggregates all features into one group
        if not self.group_all:
            self.query_and_group = QueryAndGroup(npoint, radius, nsample, use_xyz=True)

        # MLPs
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3)
        points: (B, C, N)
        """

        if self.group_all:
            # Use the whole point cloud as a single group
            new_xyz = None
            idx = knn(xyz, xyz, 1)  # (B, N, 1)
            idx = idx.view(-1, 1)  # (B*N, 1)
            points = points.permute(0, 2, 1).contiguous()  # (B, N, C) -> (B, C, N)
            grouped_points = torch.gather(points, dim=2, index=idx.repeat(1, 1, points.size(2)))  # (B*N, C, 1)
            grouped_points = grouped_points.view(points.size(0), points.size(2), points.size(1)).contiguous()  # (B, N, C)
            grouped_points = grouped_points.permute(0, 2, 1).contiguous()  # (B, C, N)
            new_points = grouped_points.unsqueeze(-1)
            new_xyz = xyz.unsqueeze(2).repeat(1, 1, self.npoint, 1)
        else:
            new_xyz, idx = self.query_and_group(xyz, points)

        new_points = torch.cat([new_xyz, new_points], dim=1)  # (B, C+3, npoint, nsample)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points, _ = torch.max(new_points, dim=-1)  # (B, mlp[-1], npoint)

        return new_xyz, new_points
