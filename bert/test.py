# import torch
# from sklearn.cluster import KMeans

# # Mock setup: Create random tensor data to simulate rep_des (embeddings of relations)
# # Assuming rep_des is a 2D tensor with shape (number_of_relations, embedding_dimension)
# rep_des = torch.rand(10, 128)  # 10 relations with 128-dimensional embeddings

# # Number of clusters to generate (adjust based on your actual scenario)
# num_clusters = 3

# # Initialize KMeans clustering model
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# # Fit KMeans to the representation data and get cluster assignments
# # Converting tensor to numpy array for KMeans fitting
# cluster_assignments = kmeans.fit_predict(rep_des.detach().cpu().numpy())
# print(cluster_assignments)
# # Mock data to map indices to relation names
# seen_relid = list(range(10))  # Assuming relation IDs are [0, 1, ..., 9]
# id2rel = {i: f'Relation_{chr(65 + i)}' for i in seen_relid}  # e.g., {0: 'Relation_A', 1: 'Relation_B', ...}

# # Initialize dictionary to store relations grouped by cluster
# relation_cluster_dict = {}

# # Iterate over each cluster assignment and group relations accordingly
# for idx, cluster_label in enumerate(cluster_assignments):
#     print(idx)
#     # print('1111')
#     # for abc in cluster_label:
#     #     print(abc)
#     relation_name = id2rel[seen_relid[idx]]
#     # Initialize cluster list if not already in the dictionary
#     if cluster_label not in relation_cluster_dict:
#         relation_cluster_dict[cluster_label] = []
#     # Append the relation name to the appropriate cluster
#     relation_cluster_dict[cluster_label].append(relation_name)

# # Create the final mapping of each relation to the list of relations in its cluster
# final_relation_dict = {}
# for cluster_relations in relation_cluster_dict.values():
#     for rel in cluster_relations:
#         final_relation_dict[rel] = cluster_relations

# # Print the resulting dictionary
# print("Relation Clusters:", final_relation_dict)


import torch
import torch.nn as nn
import torch.nn.functional as F

# Dummy distance metric function for demonstration purposes
class TripletDistanceMetric:
    @staticmethod
    def COSINE(x, y):
        # Calculate cosine distance between x and y
        return 1 - F.cosine_similarity(x, y)

# Define the TripletLoss class as provided
class TripletLoss(nn.Module):
    def __init__(self, distance_metric=TripletDistanceMetric.COSINE, triplet_margin: float = 1.0) -> None:
        super().__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, rep_anchor, rep_pos, rep_neg) -> torch.Tensor:
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        # Calculate the triplet loss
        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        # Set losses that equal the margin to zero
        losses[losses == self.triplet_margin] = 0.0
        print(type(losses))
        print(losses)
        return losses.mean()

# Test the TripletLoss
def test_triplet_loss():
    # Initialize the loss function
    triplet_loss = TripletLoss(triplet_margin=1.0)

    # Create dummy data for anchor, positive, and negative embeddings
    rep_anchor = torch.randn(5, 128)  # 5 samples, 128 dimensions
    rep_pos = torch.randn(5, 128)
    rep_neg = torch.randn(5, 128)

    # Compute the loss
    loss_value = triplet_loss(rep_anchor, rep_pos, rep_neg)

    # Print the loss
    print(f"Computed Triplet Loss: {loss_value.item()}")

# Run the test
test_triplet_loss()
