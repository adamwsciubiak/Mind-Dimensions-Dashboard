# notebook_functions.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import plotly.subplots as sp
from math import ceil
# Matplotlib is no longer strictly needed if we convert all plots to Plotly
# import matplotlib.pyplot as plt 
# import matplotlib.colors as mcolors

# --- Data Loading and Preprocessing ---
# ... (load_selected_local_datasets and preprocess_data_from_notebook - no changes) ...
def load_selected_local_datasets(study_numbers_to_load: list):
    # ... (no change)
    datasets = []
    if not study_numbers_to_load: return []
    if 1 in study_numbers_to_load:
        try: datasets.append(pd.read_csv('data/bodyheartmind_study1.csv'))
        except FileNotFoundError: print("Error: bodyheartmind_study1.csv not found.")
    if 2 in study_numbers_to_load:
        try: datasets.append(pd.read_csv('data/bodyheartmind_study2.csv'))
        except FileNotFoundError: print("Error: bodyheartmind_study2.csv not found.")
    if 4 in study_numbers_to_load:
        try: datasets.append(pd.read_csv('data/bodyheartmind_study4.csv'))
        except FileNotFoundError: print("Error: bodyheartmind_study4.csv not found.")
    if not datasets: print("Error: None of the selected dataset files were found.")
    return datasets

def preprocess_data_from_notebook(datasets_list: list,
                                  conditions_to_keep: list = None,
                                  capacities_to_remove: list = None, 
                                  apply_standard_scaler=False,
                                  normalizing=False):
    # ... (no change)
    if not datasets_list: return pd.DataFrame(), []
    concated_data = pd.concat(datasets_list, ignore_index=True)
    demographic_columns = [col for col in concated_data.columns if "race" in col or "religion" in col]
    other_columns = ["subid", "date", "start_time", "end_time", "finished", "mturkcode", "feedback", "display_order", "CATCH", "yob", "gender", "education"]
    non_relevant_columns = demographic_columns + other_columns
    existing_columns_to_drop = [col for col in non_relevant_columns if col in concated_data.columns]
    relevant_data = concated_data.drop(columns=existing_columns_to_drop)
    if conditions_to_keep and isinstance(conditions_to_keep, list) and len(conditions_to_keep) > 0:
        if 'condition' in relevant_data.columns:
            filtered_data_df = relevant_data[relevant_data["condition"].isin(conditions_to_keep)]
        else: filtered_data_df = relevant_data.copy()
    else: filtered_data_df = relevant_data.copy()
    if 'condition' in filtered_data_df.columns: numerical_data = filtered_data_df.drop(columns='condition')
    else: numerical_data = filtered_data_df
    if capacities_to_remove and isinstance(capacities_to_remove, list):
        cols_to_actually_remove = [col for col in capacities_to_remove if col in numerical_data.columns]
        if cols_to_actually_remove: numerical_data = numerical_data.drop(columns=cols_to_actually_remove)
    cols_to_process = []
    for col in numerical_data.columns:
        try:
            temp_col = pd.to_numeric(numerical_data[col], errors='coerce')
            if not temp_col.isnull().all():
                 cols_to_process.append(col); numerical_data[col] = temp_col
        except ValueError: pass
    numerical_data_cleaned = numerical_data[cols_to_process].copy()
    preprocessed_data = numerical_data_cleaned.dropna(how='any')
    mental_labels = list(preprocessed_data.columns)
    if not mental_labels or preprocessed_data.empty: return pd.DataFrame(), []
    if apply_standard_scaler:
        scaler = StandardScaler()
        preprocessed_data = pd.DataFrame(scaler.fit_transform(preprocessed_data), columns=mental_labels, index=preprocessed_data.index)
    if normalizing:
        normalizer = Normalizer()
        preprocessed_data = pd.DataFrame(normalizer.fit_transform(preprocessed_data), columns=mental_labels, index=preprocessed_data.index)
    return preprocessed_data, mental_labels

# --- Dimensionality Reduction ---
def perform_pca(data_transposed, n_components=3, random_state=42):
    # ... (no change)
    if data_transposed.empty: return None, None
    actual_n_components = min(n_components, data_transposed.shape[0], data_transposed.shape[1])
    if actual_n_components < 1: return None, None
    pca = PCA(n_components=actual_n_components, random_state=random_state)
    try:
        data_reduced = pca.fit_transform(data_transposed)
        explained_variance = pca.explained_variance_ratio_
        if actual_n_components < n_components and data_reduced.shape[1] < n_components:
            padding_cols = n_components - data_reduced.shape[1]
            padding = np.zeros((data_reduced.shape[0], padding_cols))
            data_reduced = np.hstack((data_reduced, padding))
        return data_reduced, explained_variance
    except Exception: return None, None

# --- Clustering Algorithms ---
# ... (calculate_dbscan, calculate_kmeans, calculate_gmm, calculate_hierarchical - no changes)
def calculate_dbscan(data_for_clustering: pd.DataFrame, eps_values: np.ndarray, min_samples: int = 2):
    results = {}
    if data_for_clustering.empty or data_for_clustering.shape[0] < min_samples : return results
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_for_clustering)
        results[eps] = {"labels": labels}
    return results

def calculate_kmeans(data_for_clustering: pd.DataFrame, k_values: range, random_state: int = 42):
    results = {}
    if data_for_clustering.empty or data_for_clustering.shape[0] < 2 : return results 
    pca_centers_model = None; plot_dim = 3
    if data_for_clustering.shape[1] > plot_dim and data_for_clustering.shape[0] > plot_dim :
        pca_n_components_centers = min(plot_dim, data_for_clustering.shape[0] -1 if data_for_clustering.shape[0] > plot_dim else data_for_clustering.shape[0], data_for_clustering.shape[1]) 
        if pca_n_components_centers >=1 and pca_n_components_centers <= data_for_clustering.shape[0]: 
            pca_centers_model = PCA(n_components=pca_n_components_centers, random_state=random_state)
            try: pca_centers_model.fit(data_for_clustering)
            except ValueError: pca_centers_model = None
    for k in k_values:
        if k <=0 or k > data_for_clustering.shape[0]: continue
        kmeans_model = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans_model.fit(data_for_clustering)
        labels = kmeans_model.labels_
        centers_original_dim = kmeans_model.cluster_centers_
        centers_plot_dim = centers_original_dim
        if data_for_clustering.shape[1] > plot_dim and pca_centers_model:
            try: centers_plot_dim = pca_centers_model.transform(centers_original_dim)
            except Exception: centers_plot_dim = centers_original_dim[:, :plot_dim] if centers_original_dim.shape[1] >= plot_dim else centers_original_dim
        if centers_plot_dim.shape[1] < plot_dim:
            padding_cols = plot_dim - centers_plot_dim.shape[1]
            if padding_cols > 0: centers_plot_dim = np.hstack((centers_plot_dim, np.zeros((centers_plot_dim.shape[0], padding_cols))))
        score = np.abs(kmeans_model.score(data_for_clustering))
        results[k] = {"labels": labels, "centers": centers_plot_dim, "score": score}
    return results

def calculate_gmm(data_for_clustering: pd.DataFrame, k_values: range, covariance_type: str = 'full', random_state: int = 42):
    results = {}
    if data_for_clustering.empty or data_for_clustering.shape[0] < 2 : return results
    pca_means_model = None; plot_dim = 3
    if data_for_clustering.shape[1] > plot_dim and data_for_clustering.shape[0] > plot_dim:
        pca_n_components_means = min(plot_dim, data_for_clustering.shape[0] -1 if data_for_clustering.shape[0] > plot_dim else data_for_clustering.shape[0], data_for_clustering.shape[1])
        if pca_n_components_means >=1 and pca_n_components_means <= data_for_clustering.shape[0]:
            pca_means_model = PCA(n_components=pca_n_components_means, random_state=random_state)
            try: pca_means_model.fit(data_for_clustering)
            except ValueError: pca_means_model = None
    for k in k_values:
        if k <=0 or k > data_for_clustering.shape[0]: continue
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=random_state)
        gmm.fit(data_for_clustering)
        labels = gmm.predict(data_for_clustering)
        means_original_dim = gmm.means_
        means_plot_dim = means_original_dim
        if data_for_clustering.shape[1] > plot_dim and pca_means_model:
            try: means_plot_dim = pca_means_model.transform(means_original_dim)
            except Exception: means_plot_dim = means_original_dim[:, :plot_dim] if means_original_dim.shape[1] >= plot_dim else means_original_dim
        if means_plot_dim.shape[1] < plot_dim:
            padding_cols = plot_dim - means_plot_dim.shape[1]
            if padding_cols > 0: means_plot_dim = np.hstack((means_plot_dim, np.zeros((means_plot_dim.shape[0], padding_cols))))
        results[k] = {'labels': labels, 'gmm_probabilities': gmm.predict_proba(data_for_clustering), 'gmm_means': means_plot_dim, 'gmm_covariances': gmm.covariances_, 'gmm_weights': gmm.weights_}
    return results

def calculate_hierarchical(data_for_clustering: pd.DataFrame, k_values: range, linkage_method: str = 'ward'):
    results = {}
    if data_for_clustering.empty or data_for_clustering.shape[0] < 2: return results
    for k in k_values:
        if k <= 0 or k > data_for_clustering.shape[0]: continue
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = model.fit_predict(data_for_clustering)
        results[k] = {'labels': labels}
    return results

def compute_linkage_matrix(data, linkage_method='ward'):
    if data.empty: return None
    return linkage(data, method=linkage_method, metric='euclidean')

# --- Plotting Functions ---
def create_scatter_trace(data_3d, cluster_labels=None, text_labels=None, colorscale='Bluered', trace_name='Mental Capacities'):
    # ... (no change)
    if data_3d is None or data_3d.shape[1] < 3: return None
    if cluster_labels is None: cluster_labels = np.zeros(data_3d.shape[0])
    if text_labels is None or len(text_labels) != data_3d.shape[0]:
        text_labels = [f"Point {i}" for i in range(data_3d.shape[0])]
    trace = go.Scatter3d(x=data_3d[:, 0], y=data_3d[:, 1], z=data_3d[:, 2], mode='markers+text', marker=dict(size=5, color=cluster_labels, colorscale=colorscale, opacity=0.8, showscale=False), text=text_labels, textposition='top center', hoverinfo='text', name=trace_name)
    return trace

# create_animation_frames and create_animation_controls are no longer needed if we remove the animation slider
# and use create_static_plot_for_param.

def create_static_plot_for_param(data_3d, clustering_results, selected_param_value, text_labels, algorithm_name, colorscale='Bluered', specific_sub_param_value=""):
    # ... (no change from previous correct version, ensure showlegend=False)
    if data_3d is None or data_3d.shape[1] < 3:
        fig = go.Figure(); fig.update_layout(title_text="Error: Cannot generate 3D plot.", height=500); return fig
    if not clustering_results or selected_param_value not in clustering_results:
        fig = go.Figure(); fig.update_layout(title_text=f"No results for {algorithm_name} at selected parameter.", height=500); return fig
    current_result = clustering_results[selected_param_value]
    current_labels = current_result["labels"]
    static_trace = create_scatter_trace(data_3d, current_labels, text_labels, colorscale)
    if static_trace is None:
        fig = go.Figure(); fig.update_layout(title_text="Error: Could not create scatter trace.", height=500); return fig
    figure_data_list = [static_trace]
    centers_or_means_data = None
    if "centers" in current_result and current_result["centers"] is not None and isinstance(current_result["centers"], np.ndarray) and current_result["centers"].size > 0:
        centers_or_means_data = current_result["centers"]
    elif "gmm_means" in current_result and current_result["gmm_means"] is not None and isinstance(current_result["gmm_means"], np.ndarray) and current_result["gmm_means"].size > 0:
        centers_or_means_data = current_result["gmm_means"]
    if (algorithm_name in ['KMeans', 'GMM']) and centers_or_means_data is not None and centers_or_means_data.shape[1] == 3:
        centers_trace = go.Scatter3d(x=centers_or_means_data[:, 0], y=centers_or_means_data[:, 1], z=centers_or_means_data[:, 2], mode='markers', marker=dict(size=10, color='red', symbol='x', opacity=1), name='Centers/Means')
        figure_data_list.append(centers_trace)
    fig = go.Figure(data=figure_data_list)
    param_display = f"{selected_param_value:.2f}" if isinstance(selected_param_value, float) else selected_param_value
    sub_param_title_part = f" ({specific_sub_param_value})" if specific_sub_param_value and specific_sub_param_value != 'default_run' else ""
    fig_title = f"{algorithm_name}{sub_param_title_part} (param = {param_display})"
    fig.update_layout(title=dict(text=fig_title, y=0.95, x=0.5, xanchor='center', yanchor='top'), width=None, height=600, scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3', xaxis_showticklabels=True, yaxis_showticklabels=True, zaxis_showticklabels=True), margin=dict(l=0, r=0, b=0, t=70), showlegend=False)
    return fig

def plot_gradient_branch_dendrogram(data_transposed: pd.DataFrame, linkage_method: str = 'ward', 
                                    orientation: str = 'left', # Default to 'left'
                                    title: str = "Dendrogram",
                                    labels_for_clusters: np.ndarray = None, # For coloring by final clusters
                                    num_final_clusters: int = 0): # Number of clusters for coloring
    if data_transposed.empty:
        fig = go.Figure(); fig.update_layout(title_text="Error: Empty data for dendrogram.", height=300); return fig
        
    leaf_labels_list = data_transposed.index.tolist()
    
    # Use a fixed colorscale or generate one if labels_for_clusters is provided
    if labels_for_clusters is not None and num_final_clusters > 0:
        # Create a discrete colorscale based on the number of clusters
        # This is a simplified approach for ff.create_dendrogram.
        # True per-cluster subtree coloring is complex.
        # We'll use color_threshold to differentiate main groups.
        Z = linkage(data_transposed.values, method=linkage_method, metric='euclidean')
        if Z is None or Z.shape[0] < num_final_clusters -1: # Check if linkage matrix is valid for this k
             color_threshold = 0.7 * np.max(Z[:, 2]) if Z is not None else 0
        else:
            # Cut the dendrogram at the (k-1)th highest merge to get k clusters
            # The (N-k)th value in Z[:, 2] (sorted distances) is a common threshold
            # For ff.create_dendrogram, it's often easier to use a fraction of max distance
             if Z.shape[0] >= num_final_clusters -1 and num_final_clusters > 1:
                 color_threshold = (Z[-(num_final_clusters-1), 2] + Z[-(num_final_clusters), 2])/2 if num_final_clusters <= Z.shape[0] else Z[-1,2]*0.7
             else:
                 color_threshold = Z[-1,2]*0.7 # Fallback if k is too small or large
        colorscale_param = None # Use default colors if color_threshold does the job
    else:
        colorscale_param = [[0.0, 'blue'], [0.5, 'purple'], [1.0, 'red']] # Gradient
        color_threshold = 0 # No specific thresholding if not coloring by clusters

    try:
        fig = ff.create_dendrogram(
            data_transposed.values, 
            orientation=orientation, 
            labels=leaf_labels_list,
            colorscale=colorscale_param, 
            color_threshold=color_threshold if labels_for_clusters is not None else None,
            linkagefun=lambda x: linkage(x, method=linkage_method, metric='euclidean')
        )
    except Exception as e: 
        fig = go.Figure(); fig.update_layout(title_text=f"Error creating dendrogram: {e}", height=300); return fig

    fig.update_layout(
        title_text=title,
        height=max(600, len(leaf_labels_list) * 18) if orientation in ['left', 'right'] else 600, 
        width=800, showlegend=False 
    )
    if orientation in ['bottom', 'top']:
        fig.update_layout(xaxis_title="Mental Capacities", yaxis_title="Distance")
    else: 
        fig.update_layout(xaxis_title="Distance", yaxis_title="Mental Capacities")
        fig.update_yaxes(tickangle=0)
    return fig

def generate_descriptive_plots(df_num_data, mental_labels_list):
    # ... (no change) ...
    if df_num_data.empty or not mental_labels_list: return []
    figs = []; num_columns = len(mental_labels_list); plots_per_row = 4; num_rows_total = ceil(num_columns / plots_per_row)
    for r_idx in range(num_rows_total):
        cols_this_row = mental_labels_list[r_idx * plots_per_row : (r_idx + 1) * plots_per_row]
        if not cols_this_row: continue
        fig = sp.make_subplots(rows=1, cols=len(cols_this_row), subplot_titles=cols_this_row, horizontal_spacing=0.08, vertical_spacing=0.1)
        for c_idx, column_name in enumerate(cols_this_row):
            if column_name not in df_num_data.columns: continue
            value_counts = df_num_data[column_name].value_counts().sort_index()
            bar = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': 'Rating', 'y': 'Count'})
            for trace in bar['data']: fig.add_trace(trace, row=1, col=c_idx + 1)
        fig.update_layout(height=350, showlegend=False, margin=dict(l=50, r=20, t=50, b=50))
        figs.append(fig)
    return figs

def plot_silhouette_scores_vs_param(data_for_metric, clustering_results_dict, param_name, mental_labels, title_suffix=""):
    # ... (no change) ...
    if data_for_metric.empty or not clustering_results_dict: return None
    param_values = sorted(list(clustering_results_dict.keys()))
    scores = []; valid_params_for_plot = []
    for p_val in param_values:
        labels = clustering_results_dict[p_val].get("labels")
        if labels is not None and data_for_metric.shape[0] > 1 and len(np.unique(labels)) > 1 and len(np.unique(labels)) < data_for_metric.shape[0]:
            try: score = silhouette_score(data_for_metric, labels); scores.append(score); valid_params_for_plot.append(p_val)
            except ValueError: pass
    if not scores: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid_params_for_plot, y=scores, mode='lines+markers', name='Silhouette Score'))
    fig.update_layout(title=f'Silhouette Score vs. {param_name.capitalize()}{title_suffix}', xaxis_title=f'{param_name.capitalize()}', yaxis_title='Silhouette Score', height=350, margin=dict(t=50, b=40, l=40, r=20))
    return fig

def plot_negative_silhouette_samples_plotly_v2(data_for_metric, labels_dict_all_methods, mental_capacity_names, param_value_str_dict, title_prefix=""):
    # ... (no change, ensure it uses Plotly Express or GO for stacked bar)
    if data_for_metric.empty or not labels_dict_all_methods: return None
    all_neg_sil_data = []
    capacities_with_any_neg = set()
    for method_full_name, labels in labels_dict_all_methods.items():
        if labels is None or len(np.unique(labels)) <= 1 or len(np.unique(labels)) >= data_for_metric.shape[0]: continue
        try:
            sample_silhouette_values = silhouette_samples(data_for_metric, labels)
            for i, score in enumerate(sample_silhouette_values):
                if score < 0:
                    all_neg_sil_data.append({'Capacity': mental_capacity_names[i], 'Method': method_full_name, 'Abs Negative Silhouette': abs(score)})
                    capacities_with_any_neg.add(mental_capacity_names[i])
        except ValueError: continue
    if not all_neg_sil_data:
        fig = go.Figure(); fig.update_layout(title=f"{title_prefix}No capacities with negative silhouette scores", height=100, xaxis_visible=False, yaxis_visible=False); return fig
    df_plot = pd.DataFrame(all_neg_sil_data)
    capacity_total_neg = df_plot.groupby('Capacity')['Abs Negative Silhouette'].sum().reset_index()
    capacity_total_neg = capacity_total_neg.sort_values(by='Abs Negative Silhouette', ascending=False)
    ordered_capacities = capacity_total_neg['Capacity'].tolist()
    fig = px.bar(df_plot, x='Capacity', y='Abs Negative Silhouette', color='Method', category_orders={'Capacity': ordered_capacities}, title=f"{title_prefix}Stacked Negative Silhouette Contributions", labels={'Abs Negative Silhouette': 'Sum of Abs(Negative Silhouette Score)'})
    fig.update_layout(height=max(400, len(ordered_capacities) * 20), xaxis_tickangle=-45, barmode='stack')
    return fig


def plot_gmm_cluster_probabilities_plotly(mental_labels, gmm_probabilities, chosen_k, title_suffix=""):
    """Creates a Plotly stacked bar chart of GMM probabilistic assignments."""
    if gmm_probabilities is None or len(mental_labels) != gmm_probabilities.shape[0]:
        return None
    
    n_samples, n_clusters = gmm_probabilities.shape

    # Sort by dominant cluster and max probability (similar to matplotlib reference)
    dominant_clusters = np.argmax(gmm_probabilities, axis=1)
    max_probs = np.max(gmm_probabilities, axis=1)
    combined = list(zip(range(n_samples), mental_labels, max_probs, dominant_clusters, gmm_probabilities))
    def custom_sort_key(x):
        idx, label, prob, cluster, prob_vec = x
        is_100 = 1 if np.isclose(prob, 1.0) else 0
        return (cluster, -is_100, -prob, idx)
    sorted_combined = sorted(combined, key=custom_sort_key)
    
    sorted_labels_display = [x[1] for x in sorted_combined]
    sorted_probabilities_array = np.array([x[4] for x in sorted_combined])

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly # Get a list of distinct colors

    for i in range(n_clusters):
        fig.add_trace(go.Bar(
            y=sorted_labels_display, # Capacities on Y for horizontal bar
            x=sorted_probabilities_array[:, i],
            name=f'Cluster {i}',
            orientation='h',
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        barmode='stack',
        title=f"GMM Probabilistic Assignments of Mental Capacities (k={chosen_k}){title_suffix}",
        xaxis_title="Probability",
        yaxis_title="Mental Capacities (Sorted)",
        height=max(400, len(sorted_labels_display) * 18), # Dynamic height
        legend_title_text="Cluster",
        yaxis={'categoryorder':'array', 'categoryarray':sorted_labels_display[::-1]} # Show sorted top to bottom
    )
    return fig

def plot_validation_metrics_comparison_plotly(data_for_metrics, all_clustering_results_map, selected_metrics, k_range_for_plot):
    # ... (Dual Y-axis implementation - no major change needed for 3rd axis on 2D plot) ...
    if data_for_metrics.empty or not all_clustering_results_map or not selected_metrics: return None
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]]) 
    min_k, max_k = k_range_for_plot; plot_k_values = list(range(min_k, max_k + 1))
    colors = px.colors.qualitative.Plotly; color_idx = 0
    ch_selected = "Calinski-Harabasz" in selected_metrics
    db_selected = "Davies-Bouldin" in selected_metrics
    sil_selected = "Silhouette" in selected_metrics
    primary_y_metrics = []; secondary_y_metrics = []
    if sil_selected: primary_y_metrics.append("Silhouette")
    if db_selected:
        if ch_selected: primary_y_metrics.append("Davies-Bouldin") # If CH takes secondary, DB goes to primary
        else: secondary_y_metrics.append("Davies-Bouldin") # Else DB can take secondary
    if ch_selected: secondary_y_metrics.append("Calinski-Harabasz")
    if len(primary_y_metrics) == 0 and len(secondary_y_metrics) == 1:
        primary_y_metrics = secondary_y_metrics; secondary_y_metrics = []

    for config_name, (results_dict, param_name, _) in all_clustering_results_map.items():
        if param_name not in ["k", "n_clusters"] or not results_dict: continue
        algo_label_part = config_name 
        metric_values = {"Silhouette": [], "Davies-Bouldin": [], "Calinski-Harabasz": []}
        actual_k_plotted_for_config = []
        for k in plot_k_values:
            if k in results_dict:
                labels = results_dict[k].get("labels")
                if labels is not None and data_for_metrics.shape[0] > 1 and len(np.unique(labels)) > 1 and len(np.unique(labels)) < data_for_metrics.shape[0]:
                    actual_k_plotted_for_config.append(k)
                    try: metric_values["Silhouette"].append(silhouette_score(data_for_metrics, labels) if "Silhouette" in selected_metrics else np.nan)
                    except: metric_values["Silhouette"].append(np.nan)
                    try: metric_values["Davies-Bouldin"].append(davies_bouldin_score(data_for_metrics, labels) if "Davies-Bouldin" in selected_metrics else np.nan)
                    except: metric_values["Davies-Bouldin"].append(np.nan)
                    try: metric_values["Calinski-Harabasz"].append(calinski_harabasz_score(data_for_metrics, labels) if "Calinski-Harabasz" in selected_metrics else np.nan)
                    except: metric_values["Calinski-Harabasz"].append(np.nan)
                else:
                    for m_key in metric_values: metric_values[m_key].append(np.nan)
            else:
                 for m_key in metric_values: metric_values[m_key].append(np.nan)
        current_color = colors[color_idx % len(colors)]; line_styles = {'Silhouette': 'solid', 'Davies-Bouldin': 'dash', 'Calinski-Harabasz': 'dot'}
        for metric_name in selected_metrics:
            if any(pd.notna(metric_values[metric_name])):
                use_secondary = metric_name in secondary_y_metrics
                fig.add_trace(go.Scatter(x=actual_k_plotted_for_config, y=metric_values[metric_name], mode='lines+markers', name=f'{algo_label_part} ({metric_name[:3]})', line=dict(color=current_color, dash=line_styles.get(metric_name, 'solid'))), secondary_y=use_secondary)
        color_idx += 1
    fig.update_layout(title='Comparison of Validation Metrics vs. Number of Clusters (k)', xaxis_title='Number of Clusters (k)', height=500, legend_title_text='Algorithm (Metric)')
    primary_y_title = " / ".join(primary_y_metrics) if primary_y_metrics else "Metric Value"
    fig.update_yaxes(title_text=primary_y_title, secondary_y=False)
    if secondary_y_metrics:
        secondary_y_title = " / ".join(secondary_y_metrics)
        fig.update_yaxes(title_text=secondary_y_title, secondary_y=True, showgrid=False if primary_y_metrics else True)
    return fig

def plot_cophenetic_correlation_plotly(ccc_dict):
    if not ccc_dict: return None
    methods = list(ccc_dict.keys()); ccc_values = list(ccc_dict.values())
    fig = px.bar(x=methods, y=ccc_values, labels={'x': "Linkage Method", 'y': "CCC Value"}, title="Cophenetic Correlation Coefficient (CCC) by Linkage Method")
    fig.update_layout(yaxis_range=[0,1], height=400)
    return fig