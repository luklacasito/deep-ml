import torch

def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)
    # Your implementation here
    N, D = points_t.shape
    for i in range(max_iterations):

        dist = torch.cdist(points_t,centroids)
        assignments = torch.argmin(dist,dim=1)


        new_centroids=centroids.clone()

        for j in range(k):
            mask = (assignments == j)
            new_centroids[j] = points_t[mask].mean(dim=0)
        if torch.allclose(new_centroids, centroids, atol= 1e-4):
            centroids= new_centroids
            break
        centroids=new_centroids
    return centroids.tolist()
