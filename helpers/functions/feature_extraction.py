import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP


'''
Potential PCS Alternatives:
1. t-Distributed Stochastic Neighbor Embedding (t-SNE)**: t-SNE is a machine learning algorithm for visualization and dimensionality reduction. It is particularly well-suited to the visualization of high-dimensional datasets. It is capable of capturing non-linear structures in the data, which PCA might miss.

2. Independent Component Analysis (ICA)**: ICA is another linear technique like PCA, but instead of looking for uncorrelated components, it looks for independent components. This can be useful when the data sources are statistically independent, such as in the case of separating mixed signals.

3. Autoencoders**: Autoencoders are a type of artificial neural network used for learning efficient codings of input data. They can capture complex, non-linear relationships in the data and can be very effective for dimensionality reduction.

4. Uniform Manifold Approximation and Projection (UMAP)**: UMAP is a dimension reduction technique that can be used for visualization similarly to t-SNE, but it has some advantages over t-SNE such as preserving more of the global structure of data.

5. Non-negative Matrix Factorization (NMF)**: NMF is a dimensionality reduction method that works by factorizing the original data matrix into non-negative matrices. It can be useful when the data matrix is non-negative, and the components are expected to be non-negative.

6. Random Forests/Gradient Boosting Machines (Feature Importance)**: Tree-based machine learning models like Random Forests and GBMs can be used to rank features based on their importance or contribution to the predictive model.

The choice of method depends on the specific characteristics of your data and the problem you're trying to solve. It's often a good idea to try several methods and see which one works best for your specific use case.
'''


def pca_feature_extraction(waveforms: np.ndarray, n_components: int=3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Principal Component Analysis (PCA) on waveforms data.

    This function applies PCA to each channel of the input waveforms and returns the transformed features and the
    explained variance ratio of the PCA components.

    Parameters
    -----------
    waveforms: np.ndarray
        The input waveforms, a 3D array of size (length(window) x #spikes x #channels).
    n_components: int, optional
        The number of principal components to consider. Default is 3.

    Returns:
    -----------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the following elements:
        - b (np.ndarray): The transformed features, a 2D array of size (#spikes x #features).
        - explained_variance_ratio (np.ndarray): The explained variance ratio of the PCA components.
    """
    
    # transpose waveforms to get it in the shape of (#spikes x length(window) x #channels)
    waveforms = waveforms.transpose(1, 0, 2)
    
    # initialize PCA
    pca = PCA(n_components=n_components)
    
    num_spikes, _, num_channels = waveforms.shape
    b = np.empty((num_spikes, num_channels * n_components))
    
    # apply PCA to each channel
    for i in range(num_channels):
        waveforms_channel = waveforms[:, :, i]
        waveforms_channel = waveforms_channel.reshape((num_spikes, -1))  # Reshape to 2D array
        b[:, i*n_components:(i+1)*n_components] = pca.fit_transform(waveforms_channel)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    return b, explained_variance_ratio


def plot_pca_features(b, n_components=3):
    """
    Plot PCA features for each pair of channels.

    This function creates scatter plots for each unique pair of channels. The scatter plots are generated for the 
    same principal component within each figure. The x and y labels are added for all subplots for clarity.

    Parameters
    -----------
    b: np.ndarray
        The transformed features, a 2D array of size (#spikes x #features).
    n_components: int, optional
        The number of principal components to consider. Default is 3.

    Returns
    -----------
    None
    """
    
    num_channels = b.shape[1] // n_components
    labels = [f'Ch{i+1}' for i in range(num_channels)]
    
    for n in range(n_components):
        plt.figure(figsize=(5, 5))
        plt.suptitle(f'Scatter plots (component {n+1})', fontsize=20)
        subplot_idx = 1
        for i in range(num_channels):
            for j in range(i+1, num_channels):
                ax = plt.subplot(num_channels-1, num_channels-1, subplot_idx)
                ax.scatter(b[:, i*n_components+n], b[:, j*n_components+n], s=.3, c="black")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(labels[i])
                ax.set_ylabel(labels[j])
                subplot_idx += 1
        plt.tight_layout()
        plt.show()


def tsne_feature_extraction(waveforms: np.ndarray, n_components: int=2) -> np.ndarray:
    """
    Perform t-SNE on waveforms data.

    This function applies t-SNE to each channel of the input waveforms and returns the transformed features.

    Parameters
    -----------
    waveforms, np.ndarray: 
        The input waveforms, a 3D array of size (length(window) x #spikes x #channels).
    n_components: int, optional 
        The number of components to consider. Default is 2.

    Returns
    -----------
    np.ndarray: 
        The transformed features, a 2D array of size (#spikes x #features).
    """
    
    # transpose waveforms to get it in the shape of (#spikes x length(window) x #channels)
    waveforms = waveforms.transpose(1, 0, 2)
    
    # initialize t-SNE
    tsne = TSNE(n_components=n_components)
    
    num_spikes, _, num_channels = waveforms.shape
    b = np.empty((num_spikes, num_channels * n_components))
    
    # apply t-SNE to each channel
    for i in range(num_channels):
        waveforms_channel = waveforms[:, :, i]
        # reshape to 2D array
        waveforms_channel = waveforms_channel.reshape((num_spikes, -1))  
        b[:, i*n_components:(i+1)*n_components] = tsne.fit_transform(waveforms_channel)
    
    return b


def plot_tsne_features(b, n_components=2):
    """
    Plot t-SNE (or UMAP) features for each pair of channels.

    This function creates scatter plots for each unique pair of channels. The scatter plots are generated for the 
    same principal component within each figure. The x and y labels are added for all subplots for clarity.

    Parameters
    -----------
    b: np.ndarray
        The transformed features, a 2D array of size (#spikes x #features).
    n_components: int, optional
        The number of principal components to consider. Default is 2.

    Returns
    -----------
    None
    """

    num_channels = b.shape[1] // n_components
    labels = [f'Ch{i+1}' for i in range(num_channels)]
    
    for n in range(n_components):
        plt.figure(figsize=(5, 5))
        plt.suptitle(f'Scatter plots (component {n+1})', fontsize=20)
        subplot_idx = 1
        for i in range(num_channels):
            for j in range(i+1, num_channels):
                ax = plt.subplot(num_channels-1, num_channels-1, subplot_idx)
                ax.scatter(b[:, i*n_components+n], b[:, j*n_components+n], s=.4, c="black")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(labels[i])
                ax.set_ylabel(labels[j])
                subplot_idx += 1
        plt.tight_layout()
        plt.show()


def umap_feature_extraction(waveforms: np.ndarray, n_components: int=2) -> np.ndarray:
    """
    Perform UMAP on waveforms data.

    This function applies UMAP to each channel of the input waveforms and returns the transformed features.

    Parameters
    -----------
    waveforms: np.ndarray
        The input waveforms, a 3D array of size (length(window) x #spikes x #channels).
    n_components: int, optional
        The number of components to consider. Default is 2.

    Returns
    -----------
    np.ndarray
        The transformed features, a 2D array of size (#spikes x #features).
    """
    
    # transpose waveforms to get it in the shape of (#spikes x length(window) x #channels)
    waveforms = waveforms.transpose(1, 0, 2)
    
    # initialize UMAP
    umap = UMAP(n_components=n_components)
    
    num_spikes, _, num_channels = waveforms.shape
    b = np.empty((num_spikes, num_channels * n_components))
    
    # apply UMAP to each channel
    for i in range(num_channels):
        waveforms_channel = waveforms[:, :, i]
        # reshape to 2D array
        waveforms_channel = waveforms_channel.reshape((num_spikes, -1))  
        b[:, i*n_components:(i+1)*n_components] = umap.fit_transform(waveforms_channel)
    
    return b