import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import time


seq_data = np.load("sequences.npy")
attribute_data = np.load("attributes.npy")


attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[","")
    aa_1 = aa.replace("]","")
    aa_2 = aa_1.replace("\'","")
    attr_new.append(aa_2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)
    
    


X = seq_data[:]

print("X shape: ",X.shape)


# # tSNE functions

# In[10]:


def gaussian_kernel(X, sigma=1.0):
    pairwise_sq_dists = pairwise_distances(X, metric='sqeuclidean')
    return np.exp(-pairwise_sq_dists / (2 * sigma**2))

def isolation_kernel(X, perplexity=30, epsilon=1e-5):
    n = X.shape[0]
    pairwise_sq_dists = pairwise_distances(X, metric='sqeuclidean')
    P = np.zeros((n, n))

    for i in range(n):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        for _ in range(100):
            D_i = pairwise_sq_dists[i, np.concatenate((np.arange(i), np.arange(i+1, n)))]
#             P_i = np.exp(-D_i * beta)
            max_D_i = np.max(D_i)
            distance_scaling = 1.0
            beta = perplexity / (distance_scaling * (max_D_i + epsilon))  # Compute beta
            P_i = np.exp(-(D_i - max_D_i) * beta)

            sum_P_i = np.sum(P_i)

            if sum_P_i > 0:
                H = np.log(sum_P_i) + beta * np.sum(D_i * P_i) / sum_P_i
                H_diff = H - np.log(perplexity)

                if np.abs(H_diff) <= epsilon:
                    break

                if H_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2
                    else:
                        beta = (beta + beta_min) / 2
            else:
                H = np.log(sum_P_i)
                H_diff = H - np.log(perplexity)

                if np.abs(H_diff) <= epsilon:
                    break

                if H_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2
                    else:
                        beta = (beta + beta_min) / 2

        if sum_P_i > 0:
            P[i, np.concatenate((np.arange(i), np.arange(i+1, n)))] = P_i / sum_P_i

    return P



def modified_isolation_kernel(X, perplexity=30, epsilon=1e-5, distance_scale=1.0, weights=None):
    n = X.shape[0]
    pairwise_dists = pairwise_distances(X)
    pairwise_dists_scaled = pairwise_dists * distance_scale
    P = np.zeros((n, n))

    for i in range(n):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        for _ in range(100):
            D_i = pairwise_dists_scaled[i, np.concatenate((np.arange(i), np.arange(i+1, n)))]
#             P_i = np.exp(-D_i * beta)
            max_D_i = np.max(D_i)
            distance_scaling = 1.0
            beta = perplexity / (distance_scaling * (max_D_i + epsilon))  # Compute beta
            P_i = np.exp(-(D_i - max_D_i) * beta)
            
            sum_P_i = np.sum(P_i)

            if sum_P_i > 0:
                H = np.log(sum_P_i) + beta * np.sum(D_i * P_i) / sum_P_i
                H_diff = H - np.log(perplexity)

                if np.abs(H_diff) <= epsilon:
                    break

                if H_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2
                    else:
                        beta = (beta + beta_min) / 2
            else:
                H = np.log(sum_P_i)
                H_diff = H - np.log(perplexity)

                if np.abs(H_diff) <= epsilon:
                    break

                if H_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2
                    else:
                        beta = (beta + beta_min) / 2

        if weights is not None:
            P[i, np.concatenate((np.arange(i), np.arange(i+1, n)))] = (P_i / sum_P_i) * weights[i]
        else:
            if sum_P_i > 0:
                P[i, np.concatenate((np.arange(i), np.arange(i+1, n)))] = P_i / sum_P_i

    return P

def tsne_p(P, X_org, no_dims=2):
#     if labels is None:
#         labels = []
    
    if isinstance(no_dims, int):
        initial_solution = False
    else:
        initial_solution = True
        ydata = no_dims
        no_dims = ydata.shape[1]
    
    n = P.shape[0]
    momentum = 0.5
    final_momentum = 0.8
    mom_switch_iter = 250
    stop_lying_iter = 100
    max_iter = 1000
    epsilon = 500
    min_gain = 0.01
    
    P[np.arange(n), np.arange(n)] = 0
    P = 0.5 * (P + P.T)
    P = np.maximum(P / np.sum(P), np.finfo(P.dtype).eps)
    const = np.sum(P * np.log(P))
    
    if not initial_solution:
        P = P * 4
    
    if not initial_solution:
        ydata = 0.0001 * np.random.randn(n, no_dims) # we can also use random_walk_init function

#         PCA Initialization
        pca = PCA(n_components=no_dims)
        ydata = pca.fit_transform(X_org)
        
# #         Random Walk Initialization
#         ydata = random_walk_init(len(X_org),2)
        
        
    
    y_incs = np.zeros_like(ydata)
    gains = np.ones_like(ydata)
    
    for iter in range(max_iter):
        if iter%100==0:
            print("Iterations: ",iter,"/",max_iter)
        sum_ydata = np.sum(ydata ** 2, axis=1)
        num = 1.0 / (1 + np.add.outer(sum_ydata, sum_ydata) - 2 * np.dot(ydata, ydata.T))
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / np.sum(num), np.finfo(num.dtype).eps)
        
        L = (P - Q) * num
        y_grads = 4 * (np.diag(np.sum(L, axis=1)) - L) @ ydata
        
        gains = (gains + 0.2) * (np.sign(y_grads) != np.sign(y_incs)) + (gains * 0.8) * (np.sign(y_grads) == np.sign(y_incs))
        gains[gains < min_gain] = min_gain
        y_incs = momentum * y_incs - epsilon * (gains * y_grads)
        ydata += y_incs
        ydata -= np.mean(ydata, axis=0)
        
        if iter == mom_switch_iter:
            momentum = final_momentum
        if iter == stop_lying_iter and not initial_solution:
            P /= 4
    
    return ydata

def random_walk_init(n, no_dims):
    ydata = np.random.randn(n, no_dims)  # Initialize ydata with random values
    
    # Perform random walk initialization
    for i in range(n):
        for j in range(1000):  # Number of random walk steps
            # Randomly select a neighbor
            neighbor = np.random.randint(n)
            
            # Update the position of ydata[i] towards the neighbor
            ydata[i] += (ydata[neighbor] - ydata[i]) / np.sqrt(j + 1)
    
    return ydata


# # tSNE functions calling

# In[11]:


# Compute affinity matrices
sigma = 1.0
psi = 1.0

##########################################################
# Compute the pairwise distances between data points
D = pairwise_distances(X)
average_distance = np.mean(D)  # D is the pairwise distance matrix
distance_scaling = average_distance
# distance_scaling = 1.0
##########################################################

##########################################################
# weights = np.random.rand(n)
# Estimate density using DBSCAN
epsilon = 0.5  # Adjust the value of epsilon as per your dataset
# min_samples = 5  # Adjust the value of min_samples as per your dataset
min_samples = len(X[0]) + 1 # should be greater than or equal to the dimensionality of the dataset plus one

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
dbscan.fit(X)
weights = dbscan.labels_
##########################################################

start_time = time.time()
affinity_gaussian = gaussian_kernel(X, sigma)
end_time = time.time()
gaussian_time = end_time - start_time
print(f"Time for gaussian_kernel: {gaussian_time:.5f} seconds")

start_time = time.time()
affinity_isolation = isolation_kernel(X, psi)
end_time = time.time()
isolation_time = end_time - start_time
print(f"Time for isolation_kernel: {isolation_time:.5f} seconds")

start_time = time.time()
affinity_modified_isolation = modified_isolation_kernel(X, psi, distance_scaling, weights)
end_time = time.time()
modified_isolation_time = end_time - start_time
print(f"Time for modified_isolation_kernel: {modified_isolation_time:.5f} seconds")





# In[ ]:


# Apply t-SNE
no_dims = 2
ydata_gaussian = tsne_p(affinity_gaussian, X , no_dims)
print("Gaussian Done")

ydata_isolation = tsne_p(affinity_isolation, X, no_dims)
print("Isolation Done")

ydata_modified_isolation = tsne_p(affinity_modified_isolation, X, no_dims)
print("Modified Isolation Done")


# # Evaluating tSNE

# In[90]:


from sklearn.manifold import trustworthiness

original_data = X[:]
# embeddings = ydata_gaussian[:]



# # Compute the pairwise distances between data points
# D = pairwise_distances(original_data)

Y_gaussian = ydata_gaussian[:]
Y_isolation = ydata_isolation[:]
Y_modified_isolation = ydata_modified_isolation[:]

# Compute the neighbor embedding metrics
# Here, we compute the average percentage of neighbors that are preserved at
# various values of k (the number of neighbors to consider)
gaussian_values = []
isolation_values = []
modified_isolation_values = []

n_neighbors = 100
for k in range(1, n_neighbors + 1):
    idx_orig = np.argsort(D, axis=1)[:, 1:k+1]
    
    idx_gaussian = np.argsort(pairwise_distances(Y_gaussian), axis=1)[:, 1:k+1]
    idx_isolation = np.argsort(pairwise_distances(Y_isolation), axis=1)[:, 1:k+1]
    idx_modified_isolation = np.argsort(pairwise_distances(Y_modified_isolation), axis=1)[:, 1:k+1]
    
    count_gaussian = np.sum(np.isin(idx_gaussian, idx_orig))
    count_isolation = np.sum(np.isin(idx_isolation, idx_orig))
    count_modified_isolation = np.sum(np.isin(idx_modified_isolation, idx_orig))
    
    val_val = count_gaussian / (100 * k)
    gaussian_values.append(val_val)
    
    isolation_val = count_isolation / (100 * k)
    isolation_values.append(isolation_val)
    
    modified_isolation_val = count_modified_isolation / (100 * k)
    modified_isolation_values.append(modified_isolation_val)
    
    
    print(f"k={k}:")
    
    

# Compute the trustworthiness and continuity of the embeddings
# Here, we compute the trustworthiness and continuity for various values of k
trust_gaussian = []
trust_isolation = []
trust_modified_isolation = []

for k in range(1, n_neighbors + 1):
    trust_gaussian.append(trustworthiness(D, pairwise_distances(Y_gaussian), n_neighbors=k))
    trust_isolation.append(trustworthiness(D, pairwise_distances(Y_isolation), n_neighbors=k))
    trust_modified_isolation.append(trustworthiness(D, pairwise_distances(Y_modified_isolation), n_neighbors=k))

print("Trustworthiness:")
print(f"  Gaussuan: {trust_gaussian}")
print(f"  Isolation: {trust_isolation}")
print(f"  Modified Isolation: {trust_modified_isolation}")




# In[91]:


import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(gaussian_values, label='Gaussian')
ax.plot(isolation_values, label='Isolation')
ax.plot(modified_isolation_values, label='Modified Isolation')
ax.set_xlabel('k')
ax.set_ylabel('Neighborhood Agreement')
ax.legend()
plt.show()


# In[92]:


import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(trust_gaussian, label='Gaussian')
ax.plot(trust_isolation, label='Isolation')
ax.plot(trust_modified_isolation, label='Modified Isolation')
ax.set_xlabel('k')
ax.set_ylabel('Trustworthiness')
ax.legend()
plt.show()


# In[93]:


# Plot the t-SNE results
plt.scatter(ydata_gaussian[:, 0], ydata_gaussian[:, 1], s=10)
plt.title("t-SNE on Gaussian Kernel")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()


# In[94]:


# Plot the t-SNE results

plt.scatter(ydata_isolation[:, 0], ydata_isolation[:, 1], s=10)
plt.title("t-SNE on Isolation Kernel")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()


# In[95]:


# Plot the t-SNE results

plt.scatter(ydata_modified_isolation[:, 0], ydata_modified_isolation[:, 1], s=10)
plt.title("t-SNE on Modified Isolation Kernel")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()


# In[96]:


storage_path = ""

np.save(storage_path + dataset_name + "_Neighborhood_Agreement_Gaussian.npy",gaussian_values)
np.save(storage_path + dataset_name + "_Neighborhood_Agreement_Isolation.npy",isolation_values)
np.save(storage_path + dataset_name + "_Neighborhood_Agreement_Modified_Isolation.npy",modified_isolation_values)


np.save(storage_path + dataset_name + "_Trustworthiness_Gaussian.npy",trust_gaussian)
np.save(storage_path + dataset_name + "_Trustworthiness_Isolation.npy",trust_isolation)
np.save(storage_path + dataset_name + "_Trustworthiness_Modified_Isolation.npy",trust_modified_isolation)

np.save(storage_path + dataset_name + "_tsne_2D_Gaussian.npy",ydata_gaussian)
np.save(storage_path + dataset_name + "_tsne_2D_Isolation.npy",ydata_isolation)
np.save(storage_path + dataset_name + "_tsne_2D_Modified_Isolation.npy",ydata_modified_isolation)
