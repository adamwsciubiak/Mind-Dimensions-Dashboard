# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, cophenet # For CCC
from scipy.spatial.distance import pdist # For CCC
import functions as nf

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Interactive Clustering Explorer")

# --- Initialize Session State ---
# ... (Session state initialization - ensure 'linkage_methods' and 'covariance_types' are lists) ...
if 'run_config' not in st.session_state:
    st.session_state.run_config = {
        "selected_studies_names": ("Study 1", "Study 2", "Study 4"),
        "selected_conditions": ["ALL"], 
        "capacities_to_remove": [], 
        "apply_standard_scaler": True,
        "normalize_data": False,
        "selected_algorithms": ["KMeans"], 
        "algo_specific_params": { 
            "KMeans": {'k_min': 2, 'k_max': 5},
            "DBSCAN": {'eps_min': 14.0, 'eps_max': 20.0, 'eps_step': 0.2, 'min_samples': 2},
            "Agglomerative Clustering": {'k_min': 2, 'k_max': 5, 'linkage_methods': ['ward']},
            "GMM": {'k_min': 2, 'k_max': 5, 'covariance_types': ['full']}
        }
    }
if 'pending_sidebar_params' not in st.session_state: 
    st.session_state.pending_sidebar_params = {
        k: v.copy() if isinstance(v, dict) else (list(v) if isinstance(v, (tuple, list)) else v)
        for k, v in st.session_state.run_config.items()
    }
    st.session_state.pending_sidebar_params["algo_specific_params"] = {
        k: v.copy() for k, v in st.session_state.run_config["algo_specific_params"].items()
    }
if 'analysis_has_been_run_at_least_once' not in st.session_state:
    st.session_state.analysis_has_been_run_at_least_once = False 
if 'condition_options_for_sidebar' not in st.session_state:
    st.session_state.condition_options_for_sidebar = ["ALL", "robot", "beetle"] 
if 'mental_capacity_options_for_sidebar' not in st.session_state: 
    st.session_state.mental_capacity_options_for_sidebar = []
if 'show_data_preview' not in st.session_state:
    st.session_state.show_data_preview = True
if 'show_descriptive_stats' not in st.session_state:
    st.session_state.show_descriptive_stats = False
if 'show_validation_plots' not in st.session_state: # Global flag for validation plots
    st.session_state.show_validation_plots = False


# --- Caching Functions (Assumed correct) ---
# ... (load_data_cached, get_initial_mental_labels, get_preprocessed_data_cached, run_pca_for_viz_cached, get_clustering_results_cached) ...
@st.cache_data
def load_data_cached(selected_study_numbers_tuple):
    raw_df = None; conditions = ["ALL"] 
    selected_study_numbers = list(selected_study_numbers_tuple)
    if not selected_study_numbers: return None, conditions
    datasets_list = nf.load_selected_local_datasets(selected_study_numbers)
    if not datasets_list: return None, conditions
    raw_df = pd.concat(datasets_list, ignore_index=True)
    if raw_df is not None and not raw_df.empty and 'condition' in raw_df.columns:
        unique_conds = sorted(list(set(str(c) for c in raw_df['condition'].unique() if pd.notna(c))))
        conditions.extend(unique_conds)
    return raw_df, sorted(list(set(conditions)))

@st.cache_data
def get_initial_mental_labels(_raw_df_tuple, conditions_to_keep_list):
    if _raw_df_tuple is None or not _raw_df_tuple[0]: return []
    raw_df = pd.DataFrame(_raw_df_tuple[0], columns=_raw_df_tuple[1])
    demographic_columns = [col for col in raw_df.columns if "race" in col or "religion" in col]
    other_columns = ["subid", "date", "start_time", "end_time", "finished", "mturkcode", "feedback", "display_order", "CATCH", "yob", "gender", "education"]
    non_relevant_columns = demographic_columns + other_columns
    existing_columns_to_drop = [col for col in non_relevant_columns if col in raw_df.columns]
    relevant_data = raw_df.drop(columns=existing_columns_to_drop)
    actual_conditions_to_filter = None
    if conditions_to_keep_list and "ALL" not in conditions_to_keep_list and conditions_to_keep_list != []:
        actual_conditions_to_filter = conditions_to_keep_list
    if actual_conditions_to_filter and 'condition' in relevant_data.columns:
        filtered_data_df = relevant_data[relevant_data["condition"].isin(actual_conditions_to_filter)]
    else: filtered_data_df = relevant_data.copy()
    if 'condition' in filtered_data_df.columns: numerical_data_guess = filtered_data_df.drop(columns='condition')
    else: numerical_data_guess = filtered_data_df
    potential_mental_labels = []
    for col in numerical_data_guess.columns:
        try: pd.to_numeric(numerical_data_guess[col], errors='raise'); potential_mental_labels.append(col)
        except: pass 
    return sorted(list(set(potential_mental_labels)))

@st.cache_data
def get_preprocessed_data_cached(_raw_df_tuple, conditions_to_keep_list, capacities_to_remove_list, apply_scaler_bool, normalizing_bool):
    if _raw_df_tuple is None or not _raw_df_tuple[0]: return None, []
    raw_df = pd.DataFrame(_raw_df_tuple[0], columns=_raw_df_tuple[1])
    actual_conditions_to_filter = None
    if conditions_to_keep_list and "ALL" not in conditions_to_keep_list and conditions_to_keep_list != []:
        actual_conditions_to_filter = conditions_to_keep_list
    processed_df, mental_labels = nf.preprocess_data_from_notebook(
        datasets_list=[raw_df], conditions_to_keep=actual_conditions_to_filter, 
        capacities_to_remove=capacities_to_remove_list, 
        apply_standard_scaler=apply_scaler_bool, normalizing=normalizing_bool
    )
    if processed_df is None or processed_df.empty: return None, []
    return processed_df, mental_labels

@st.cache_data
def run_pca_for_viz_cached(_data_transposed_tuple):
    if _data_transposed_tuple is None or not _data_transposed_tuple[0]: return None, None
    data_transposed = pd.DataFrame(_data_transposed_tuple[0], columns=_data_transposed_tuple[1], index=_data_transposed_tuple[2])
    if data_transposed.empty: return None, None
    data_3d, explained_variance = nf.perform_pca(data_transposed)
    return data_3d, explained_variance

@st.cache_data
def get_clustering_results_cached(_data_to_cluster_tuple, algorithm_name, params_dict_algo_single):
    if _data_to_cluster_tuple is None or not _data_to_cluster_tuple[0]: return None, "", []
    data_to_cluster = pd.DataFrame(_data_to_cluster_tuple[0], columns=_data_to_cluster_tuple[1], index=_data_to_cluster_tuple[2])
    if data_to_cluster.empty: return None, "", []
    clustering_results = None; param_name_for_animation = ""; param_values_for_animation = []
    if algorithm_name == "KMeans":
        k_min = params_dict_algo_single.get('k_min', 2)
        k_max = min(params_dict_algo_single.get('k_max', 5), data_to_cluster.shape[0] -1 if data_to_cluster.shape[0]>1 else 1) 
        if k_min > k_max : k_min = k_max 
        if k_max < 2: k_values = [] 
        else: k_values = range(k_min, k_max + 1)
        if k_values: clustering_results = nf.calculate_kmeans(data_to_cluster, k_values)
        param_name_for_animation = "k"; param_values_for_animation = list(k_values)
    elif algorithm_name == "DBSCAN":
        eps_values_np = np.arange(params_dict_algo_single['eps_min'], params_dict_algo_single['eps_max'] + params_dict_algo_single['eps_step'], params_dict_algo_single['eps_step'])
        eps_values = [round(e, 2) for e in eps_values_np if e <= params_dict_algo_single['eps_max']]
        if not eps_values and params_dict_algo_single['eps_min'] <= params_dict_algo_single['eps_max']: eps_values = [params_dict_algo_single['eps_min']]
        if not eps_values: return None, "", []
        clustering_results = nf.calculate_dbscan(data_to_cluster, np.array(eps_values), params_dict_algo_single['min_samples'])
        param_name_for_animation = "eps"; param_values_for_animation = eps_values
    elif algorithm_name == "Agglomerative Clustering":
        k_min = params_dict_algo_single.get('k_min', 2)
        k_max = min(params_dict_algo_single.get('k_max', 5), data_to_cluster.shape[0] -1 if data_to_cluster.shape[0]>1 else 1)
        if k_min > k_max : k_min = k_max
        if k_max < 2: k_values = []
        else: k_values = range(k_min, k_max + 1)
        if k_values: clustering_results = nf.calculate_hierarchical(data_to_cluster, k_values, params_dict_algo_single['linkage_method'])
        param_name_for_animation = "n_clusters"; param_values_for_animation = list(k_values)
    elif algorithm_name == "GMM":
        k_min = params_dict_algo_single.get('k_min', 2)
        k_max = min(params_dict_algo_single.get('k_max', 5), data_to_cluster.shape[0] -1 if data_to_cluster.shape[0]>1 else 1)
        if k_min > k_max : k_min = k_max
        if k_max < 2: k_values = []
        else: k_values = range(k_min, k_max + 1)
        if k_values: clustering_results = nf.calculate_gmm(data_to_cluster, k_values, params_dict_algo_single['covariance_type'])
        param_name_for_animation = "k"; param_values_for_animation = list(k_values)
    return clustering_results, param_name_for_animation, param_values_for_animation


# --- Streamlit App Layout ---
st.title("📊 Interactive Clustering Explorer (Mind Perception Dimensions)")
st.markdown("This dashboard allows you to explore clustering algorithms on mental capacity data...")

# --- Sidebar ---
with st.sidebar:
    # ... (Sidebar setup code from previous correct version - ENSURE KEYS ARE UNIQUE FOR WIDGETS) ...
    st.header("Analysis Controls")
    st.markdown("### 1. Data Source")
    study_options_map = {"Study 1": 1, "Study 2": 2, "Study 4": 4}
    sb_selected_study_names = st.multiselect(
        "Select studies to combine", options=list(study_options_map.keys()),
        default=list(st.session_state.pending_sidebar_params.get("selected_studies_names", ("Study 1", "Study 2", "Study 4")))
    )
    st.session_state.pending_sidebar_params["selected_studies_names"] = tuple(sb_selected_study_names)
    sb_selected_study_numbers_tuple = tuple(sorted([study_options_map[name] for name in sb_selected_study_names]))
    _, current_condition_options = load_data_cached(sb_selected_study_numbers_tuple)
    st.session_state.condition_options_for_sidebar = current_condition_options if current_condition_options else ["ALL", "robot", "beetle"]

    st.markdown("### 2. Data Filtering")
    default_sb_conds = st.session_state.pending_sidebar_params.get("selected_conditions", ["ALL"])
    valid_default_sb_conds = [c for c in default_sb_conds if c in st.session_state.condition_options_for_sidebar]
    if not valid_default_sb_conds:
        if "ALL" in st.session_state.condition_options_for_sidebar: valid_default_sb_conds = ["ALL"]
        elif "robot" in st.session_state.condition_options_for_sidebar and "beetle" in st.session_state.condition_options_for_sidebar: valid_default_sb_conds = ["robot", "beetle"]
        elif st.session_state.condition_options_for_sidebar: valid_default_sb_conds = [st.session_state.condition_options_for_sidebar[0]]
        else: valid_default_sb_conds = []
    sb_selected_conditions = st.multiselect(
        "Select conditions ('ALL' or empty for all)", options=st.session_state.condition_options_for_sidebar,
        default=valid_default_sb_conds
    )
    st.session_state.pending_sidebar_params["selected_conditions"] = sb_selected_conditions if sb_selected_conditions else ["ALL"]

    temp_raw_df_for_labels, _ = load_data_cached(sb_selected_study_numbers_tuple)
    if temp_raw_df_for_labels is not None:
        temp_raw_df_tuple_for_labels = (temp_raw_df_for_labels.values.tolist(), temp_raw_df_for_labels.columns.tolist())
        st.session_state.mental_capacity_options_for_sidebar = get_initial_mental_labels(temp_raw_df_tuple_for_labels, st.session_state.pending_sidebar_params["selected_conditions"])
    else:
        st.session_state.mental_capacity_options_for_sidebar = []
    default_sb_caps_removed = st.session_state.pending_sidebar_params.get("capacities_to_remove", [])
    valid_default_caps_removed = [c for c in default_sb_caps_removed if c in st.session_state.mental_capacity_options_for_sidebar]
    sb_capacities_to_remove = st.multiselect(
        "Select mental capacities to REMOVE (optional)",
        options=st.session_state.mental_capacity_options_for_sidebar,
        default=valid_default_caps_removed
    )
    st.session_state.pending_sidebar_params["capacities_to_remove"] = sb_capacities_to_remove

    st.markdown("### 3. Preprocessing")
    st.session_state.pending_sidebar_params["apply_standard_scaler"] = st.checkbox("Apply StandardScaler", value=st.session_state.pending_sidebar_params.get("apply_standard_scaler", True))
    st.session_state.pending_sidebar_params["normalize_data"] = st.checkbox("Normalize data (L2, after scaling)", value=st.session_state.pending_sidebar_params.get("normalize_data", False))

    st.markdown("### 4. Clustering Algorithm & Parameters")
    algorithm_options_all = ["KMeans", "DBSCAN", "Agglomerative Clustering", "GMM"]
    default_selected_algos = st.session_state.pending_sidebar_params.get("selected_algorithms", ["KMeans"])
    if not isinstance(default_selected_algos, list): default_selected_algos = [default_selected_algos]
    valid_default_selected_algos = [alg for alg in default_selected_algos if alg in algorithm_options_all]
    if not valid_default_selected_algos: valid_default_selected_algos = ["KMeans"]
    st.session_state.pending_sidebar_params["selected_algorithms"] = st.multiselect(
        "Choose clustering algorithm(s)", algorithm_options_all, default=valid_default_selected_algos
    )
    if not st.session_state.pending_sidebar_params["selected_algorithms"]:
        st.warning("Please select at least one clustering algorithm.")
        st.session_state.pending_sidebar_params["selected_algorithms"] = ["KMeans"]
    num_potential_capacities = len(st.session_state.mental_capacity_options_for_sidebar) - len(st.session_state.pending_sidebar_params.get("capacities_to_remove", []))
    max_k_ui = max(2, num_potential_capacities if num_potential_capacities > 1 else 2)

    for algo_name_pending in st.session_state.pending_sidebar_params["selected_algorithms"]:
        with st.expander(f"Parameters for {algo_name_pending}", expanded=True): # Expanded by default
            if algo_name_pending not in st.session_state.pending_sidebar_params["algo_specific_params"]:
                st.session_state.pending_sidebar_params["algo_specific_params"][algo_name_pending] = \
                    st.session_state.run_config["algo_specific_params"].get(algo_name_pending, {}).copy()
            pending_algo_specific_params_ui = st.session_state.pending_sidebar_params["algo_specific_params"][algo_name_pending]

            if algo_name_pending == "KMeans":
                pending_algo_specific_params_ui['k_min'] = st.slider("Min k", 2, max(2, max_k_ui -1) , pending_algo_specific_params_ui.get('k_min', 2), key=f"km_kmin_sb_{algo_name_pending}")
                pending_algo_specific_params_ui['k_max'] = st.slider("Max k", pending_algo_specific_params_ui['k_min'], max_k_ui, max(pending_algo_specific_params_ui['k_min'], pending_algo_specific_params_ui.get('k_max', min(5,max_k_ui))), key=f"km_kmax_sb_{algo_name_pending}")
            elif algo_name_pending == "DBSCAN":
                pending_algo_specific_params_ui['eps_min'] = st.slider("Min Epsilon", 0.1, 50.0, pending_algo_specific_params_ui.get('eps_min', 14.0), 0.1, format="%.1f", key=f"db_epsmin_sb_{algo_name_pending}")
                pending_algo_specific_params_ui['eps_max'] = st.slider("Max Epsilon", pending_algo_specific_params_ui['eps_min'], 50.0, max(pending_algo_specific_params_ui['eps_min'], pending_algo_specific_params_ui.get('eps_max', 20.0)), 0.1, format="%.1f", key=f"db_epsmax_sb_{algo_name_pending}")
                pending_algo_specific_params_ui['eps_step'] = st.slider("Epsilon Step", 0.1, 5.0, pending_algo_specific_params_ui.get('eps_step', 0.2), 0.1, format="%.1f", key=f"db_epsstep_sb_{algo_name_pending}")
                pending_algo_specific_params_ui['min_samples'] = st.slider("Min samples", 1, 20, pending_algo_specific_params_ui.get('min_samples', 2), key=f"db_minsamp_sb_{algo_name_pending}")
            elif algo_name_pending == "Agglomerative Clustering":
                pending_algo_specific_params_ui['k_min'] = st.slider("Min n_clusters", 2, max(2,max_k_ui-1), pending_algo_specific_params_ui.get('k_min', 2), key=f"ac_kmin_sb_{algo_name_pending}")
                pending_algo_specific_params_ui['k_max'] = st.slider("Max n_clusters", pending_algo_specific_params_ui['k_min'], max_k_ui, max(pending_algo_specific_params_ui['k_min'], pending_algo_specific_params_ui.get('k_max', min(5,max_k_ui))), key=f"ac_kmax_sb_{algo_name_pending}")
                linkage_options = ['ward', 'complete', 'average', 'single']
                pending_algo_specific_params_ui['linkage_methods'] = st.multiselect("Linkage method(s)", linkage_options, default=pending_algo_specific_params_ui.get('linkage_methods', ['ward']), key=f"ac_link_sb_{algo_name_pending}")
                if not pending_algo_specific_params_ui['linkage_methods']: pending_algo_specific_params_ui['linkage_methods'] = ['ward']
            elif algo_name_pending == "GMM":
                pending_algo_specific_params_ui['k_min'] = st.slider("Min n_components (k)", 2, max(2,max_k_ui-1), pending_algo_specific_params_ui.get('k_min', 2), key=f"gmm_kmin_sb_{algo_name_pending}")
                pending_algo_specific_params_ui['k_max'] = st.slider("Max n_components (k)", pending_algo_specific_params_ui['k_min'], max_k_ui, max(pending_algo_specific_params_ui['k_min'], pending_algo_specific_params_ui.get('k_max', min(5,max_k_ui))), key=f"gmm_kmax_sb_{algo_name_pending}")
                cov_types = ['full', 'tied', 'diag', 'spherical']
                pending_algo_specific_params_ui['covariance_types'] = st.multiselect("Covariance type(s)", cov_types, default=pending_algo_specific_params_ui.get('covariance_types', ['full']), key=f"gmm_cov_sb_{algo_name_pending}")
                if not pending_algo_specific_params_ui['covariance_types']: pending_algo_specific_params_ui['covariance_types'] = ['full']
            st.session_state.pending_sidebar_params["algo_specific_params"][algo_name_pending] = pending_algo_specific_params_ui

    st.markdown("---")
    st.session_state.show_validation_plots = st.toggle( # Global validation plot toggle
        "Show Validation Plots Section (Overall & Per Config)",
        value=st.session_state.show_validation_plots,
        key="global_validation_toggle_sidebar"
    )
    st.markdown("---")

    if st.button("🚀 Run Analysis", type="primary"):
        st.session_state.run_config = {
            k: v.copy() if isinstance(v, dict) else (list(v) if isinstance(v, (tuple, list)) else v)
            for k, v in st.session_state.pending_sidebar_params.items()
        }
        st.session_state.run_config["algo_specific_params"] = {
            k: v.copy() for k, v in st.session_state.pending_sidebar_params["algo_specific_params"].items()
        }
        st.session_state.analysis_has_been_run_at_least_once = True
        st.rerun()

# --- Main Area ---
# ... (Data Preview section - no change) ...
main_display_study_numbers_tuple = tuple(sorted([study_options_map[name] for name in st.session_state.run_config["selected_studies_names"]]))
main_display_raw_df, _ = load_data_cached(main_display_study_numbers_tuple)

show_preview_main_toggle = st.toggle("Show Data Preview", value=st.session_state.show_data_preview, key="data_preview_main_toggle")
if show_preview_main_toggle != st.session_state.show_data_preview:
    st.session_state.show_data_preview = show_preview_main_toggle
    st.rerun()

if st.session_state.show_data_preview:
    if main_display_raw_df is None or main_display_raw_df.empty:
        st.info("Data not loaded. Select studies from the sidebar and click 'Run Analysis'.")
    else:
        st.header("📋 Data Preview (based on last 'Run Analysis' settings)")
        col1_main_preview, col2_main_preview = st.columns(2)
        with col1_main_preview:
            st.subheader(f"Raw Data (Top 5 rows)")
            st.dataframe(main_display_raw_df.head())
            if 'condition' in main_display_raw_df.columns:
                st.subheader("Items per Condition (from selected studies & conditions of last run)")
                conditions_filter_for_preview = st.session_state.run_config["selected_conditions"]
                df_to_count_preview = main_display_raw_df
                if conditions_filter_for_preview and "ALL" not in conditions_filter_for_preview:
                    df_to_count_preview = main_display_raw_df[main_display_raw_df['condition'].isin(conditions_filter_for_preview)]
                condition_counts_preview = df_to_count_preview['condition'].value_counts().reset_index()
                condition_counts_preview.columns = ['Condition', 'Count']
                st.dataframe(condition_counts_preview)

        main_raw_df_tuple_for_preview = (main_display_raw_df.values.tolist(), main_display_raw_df.columns.tolist())
        actual_conditions_for_preview = None
        if st.session_state.run_config["selected_conditions"] and "ALL" not in st.session_state.run_config["selected_conditions"]:
             actual_conditions_for_preview = st.session_state.run_config["selected_conditions"]
        
        preview_processed_df, preview_mental_labels = get_preprocessed_data_cached(
            main_raw_df_tuple_for_preview, actual_conditions_for_preview, 
            st.session_state.run_config.get("capacities_to_remove", []),
            st.session_state.run_config["apply_standard_scaler"], 
            st.session_state.run_config["normalize_data"]
        )
        
        if preview_processed_df is not None and not preview_processed_df.empty:
            with col2_main_preview:
                st.subheader("Processed & Transposed Data (Top 5 capacities)")
                st.dataframe(preview_processed_df.T.head())
        else:
            with col2_main_preview: st.warning("Could not process data for preview with current run settings.")
        
        show_desc_main_toggle = st.toggle("Show Descriptive Statistics", value=st.session_state.show_descriptive_stats, key="desc_stats_main_toggle")
        if show_desc_main_toggle != st.session_state.show_descriptive_stats:
            st.session_state.show_descriptive_stats = show_desc_main_toggle
            st.rerun()
            
        if st.session_state.show_descriptive_stats:
            if preview_processed_df is not None and not preview_processed_df.empty and preview_mental_labels:
                st.subheader("Descriptive Statistics (Processed Participant Data)")
                st.dataframe(preview_processed_df.describe())
                st.subheader("Distribution of Ratings (Processed Participant Data)")
                descriptive_figs = nf.generate_descriptive_plots(preview_processed_df, preview_mental_labels)
                for fig_idx, fig_desc in enumerate(descriptive_figs): 
                    st.plotly_chart(fig_desc, use_container_width=True, key=f"desc_plot_{fig_idx}")
            else: st.info("No processed data for descriptives (Run Analysis with valid settings).")

st.markdown("---")
st.header("📈 Clustering Results on Mental Capacities")
results_container = st.container()

if st.session_state.analysis_has_been_run_at_least_once:
    analysis_raw_df, _ = load_data_cached(tuple(sorted([study_options_map[name] for name in st.session_state.run_config["selected_studies_names"]])))
    if analysis_raw_df is None or analysis_raw_df.empty:
        results_container.error("Failed to load data for analysis based on current run configuration.")
    else:
        analysis_raw_df_tuple = (analysis_raw_df.values.tolist(), analysis_raw_df.columns.tolist())
        analysis_conditions_filter = None
        if st.session_state.run_config["selected_conditions"] and "ALL" not in st.session_state.run_config["selected_conditions"]:
             analysis_conditions_filter = st.session_state.run_config["selected_conditions"]
        
        run_processed_df, run_mental_labels = get_preprocessed_data_cached(
            analysis_raw_df_tuple, analysis_conditions_filter,
            st.session_state.run_config.get("capacities_to_remove", []),
            st.session_state.run_config["apply_standard_scaler"], 
            st.session_state.run_config["normalize_data"]
        )
        run_data_for_clustering_transposed = pd.DataFrame()
        if run_processed_df is not None and not run_processed_df.empty:
            run_data_for_clustering_transposed = run_processed_df.T

        if run_data_for_clustering_transposed.empty or not run_mental_labels:
            results_container.error("⚠️ Cannot run analysis: Data for clustering is empty or no features.")
        else:
            all_configs_to_run_display = []
            for algo_name_run_config in st.session_state.run_config["selected_algorithms"]:
                algo_params_run_config = st.session_state.run_config["algo_specific_params"].get(algo_name_run_config, {})
                if algo_name_run_config == "Agglomerative Clustering":
                    for linkage_method in algo_params_run_config.get('linkage_methods', ['ward']):
                        cfg = algo_params_run_config.copy(); cfg['linkage_method'] = linkage_method
                        all_configs_to_run_display.append({'algo_name': algo_name_run_config, 'sub_type_name': 'Linkage', 'sub_type_value': linkage_method, 'params_for_calc': cfg})
                elif algo_name_run_config == "GMM":
                    for cov_type in algo_params_run_config.get('covariance_types', ['full']):
                        cfg = algo_params_run_config.copy(); cfg['covariance_type'] = cov_type
                        all_configs_to_run_display.append({'algo_name': algo_name_run_config, 'sub_type_name': 'Covariance', 'sub_type_value': cov_type, 'params_for_calc': cfg})
                else:
                    all_configs_to_run_display.append({'algo_name': algo_name_run_config, 'sub_type_name': '', 'sub_type_value': algo_name_run_config + '_default', 'params_for_calc': algo_params_run_config})
            
            if not all_configs_to_run_display:
                results_container.info("No algorithms selected or configured for the run.")
            else:
                num_display_cols = len(all_configs_to_run_display)
                cols_results = results_container.columns(num_display_cols if num_display_cols > 0 else 1)
                all_iter_clustering_results_map = {} 

                for i, run_info_config in enumerate(all_configs_to_run_display):
                    current_column = cols_results[i % num_display_cols] if num_display_cols > 0 else results_container 
                    with current_column:
                        current_run_algo_name = run_info_config['algo_name']
                        current_run_sub_type_name = run_info_config['sub_type_name']
                        current_run_sub_type_value = run_info_config['sub_type_value'] 
                        current_run_calc_params_dict = run_info_config['params_for_calc']
                        
                        subheader_text = f"Algorithm: {current_run_algo_name}"
                        if current_run_sub_type_name: 
                            subheader_text += f" ({current_run_sub_type_name}: {current_run_sub_type_value})"
                        st.subheader(subheader_text)

                        with st.spinner(f"Running {current_run_algo_name} ({current_run_sub_type_value})..."):
                            run_data_to_cluster_tuple_iter = (
                                run_data_for_clustering_transposed.values.tolist(), 
                                run_data_for_clustering_transposed.columns.tolist(), run_mental_labels
                            )
                            run_data_3d_capacities_iter, _ = run_pca_for_viz_cached(run_data_to_cluster_tuple_iter)
                            
                            iter_clustering_results, iter_p_name, iter_p_values = get_clustering_results_cached(
                                run_data_to_cluster_tuple_iter, current_run_algo_name, current_run_calc_params_dict
                            )
                            config_validation_key = f"{current_run_algo_name}_{current_run_sub_type_value}" # Unique key for this config
                            all_iter_clustering_results_map[config_validation_key] = (iter_clustering_results, iter_p_name, iter_p_values)

                            selected_param_for_details = None # Will be set by the slider
                            
                            # Slider for selecting parameter for static plot, metrics, assignments
                            if iter_clustering_results and iter_p_values:
                                slider_key_details = f"details_slider_{current_run_algo_name}_{iter_p_name}_{current_run_sub_type_value}_{i}"
                                valid_param_options_details = [p for p in iter_p_values if p in iter_clustering_results]
                                if valid_param_options_details:
                                    default_val_details = valid_param_options_details[0]
                                    if isinstance(default_val_details, float):
                                        sel_param_disp_details = st.select_slider(f"Select '{iter_p_name}' for plot & details", options=[round(p,2) for p in valid_param_options_details], value=round(default_val_details,2), format_func=lambda x:f"{x:.2f}", key=slider_key_details)
                                        selected_param_for_details = next((p for p in valid_param_options_details if round(p,2) == sel_param_disp_details), default_val_details)
                                    else:
                                        selected_param_for_details = st.select_slider(f"Select '{iter_p_name}' for plot & details", options=valid_param_options_details, value=default_val_details, key=slider_key_details)

                            # Static Plot based on slider
                            plot_data_static = run_data_3d_capacities_iter
                            if run_data_3d_capacities_iter is None and not run_data_for_clustering_transposed.empty:
                                 if run_data_for_clustering_transposed.shape[1] == 2: plot_data_static = np.hstack([run_data_for_clustering_transposed.values, np.zeros((run_data_for_clustering_transposed.shape[0], 1))])
                                 elif run_data_for_clustering_transposed.shape[1] == 1: plot_data_static = np.hstack([run_data_for_clustering_transposed.values, np.zeros((run_data_for_clustering_transposed.shape[0], 2))])
                            
                            if iter_clustering_results and selected_param_for_details is not None and \
                               plot_data_static is not None and plot_data_static.shape[1] == 3:
                                static_fig = nf.create_static_plot_for_param(
                                    plot_data_static, iter_clustering_results, selected_param_for_details,
                                    text_labels=run_mental_labels, algorithm_name=current_run_algo_name,
                                    colorscale='Bluered', specific_sub_param_value=current_run_sub_type_value if current_run_sub_type_name else ''
                                )
                                st.plotly_chart(static_fig, use_container_width=True, key=f"static_plot_{current_run_algo_name}_{current_run_sub_type_value}_{i}")
                            elif iter_clustering_results: st.warning("3D static plot could not be generated.")
                            
                            # Detailed Results, Metrics, Assignments, and Specific Validation Plots
                            if iter_clustering_results and selected_param_for_details is not None and selected_param_for_details in iter_clustering_results:
                                param_disp_val_iter = f"{selected_param_for_details:.2f}" if isinstance(selected_param_for_details, float) else selected_param_for_details
                                specific_labels_iter = iter_clustering_results[selected_param_for_details]["labels"]
                                
                                if current_run_algo_name == "Agglomerative Clustering":
                                    st.markdown("###### Algorithm-Specific Validation")
                                    Z_matrix = nf.compute_linkage_matrix(run_data_for_clustering_transposed, current_run_sub_type_value)
                                    if Z_matrix is not None:
                                        ccc_val, _ = cophenet(Z_matrix, pdist(run_data_for_clustering_transposed.values))
                                        st.metric(label=f"Cophenetic Corr. Coeff. (CCC)", value=f"{ccc_val:.4f}")

                                st.markdown("###### General Validation Scores")
                                metrics_iter = {}
                                if run_data_for_clustering_transposed.shape[0] >=2 and len(np.unique(specific_labels_iter)) > 1 and len(np.unique(specific_labels_iter)) < run_data_for_clustering_transposed.shape[0]:
                                    try: metrics_iter["Silhouette"] = silhouette_score(run_data_for_clustering_transposed, specific_labels_iter)
                                    except: pass
                                    try: metrics_iter["Davies-Bouldin"] = davies_bouldin_score(run_data_for_clustering_transposed, specific_labels_iter)
                                    except: pass
                                    try: metrics_iter["Calinski-Harabasz"] = calinski_harabasz_score(run_data_for_clustering_transposed, specific_labels_iter)
                                    except: pass
                                else: metrics_iter = {"Silhouette": "N/A", "Davies-Bouldin": "N/A", "Calinski-Harabasz": "N/A"}; st.caption("(Metrics req. >1 & <N clusters)")
                                
                                n_clusters_iter = len(np.unique(specific_labels_iter))
                                if -1 in np.unique(specific_labels_iter) and current_run_algo_name == "DBSCAN": st.write(f"**For {iter_p_name}={param_disp_val_iter}:** {n_clusters_iter-1} clusters (excl. noise), {np.sum(specific_labels_iter == -1)} noise points")
                                else: st.write(f"**For {iter_p_name}={param_disp_val_iter}:** {n_clusters_iter} clusters found")
                                
                                metric_display_cols_iter = st.columns(len(metrics_iter) if metrics_iter else 1)
                                for m_idx, (m_name, m_val) in enumerate(metrics_iter.items()):
                                    with metric_display_cols_iter[m_idx % len(metric_display_cols_iter)]:
                                        if isinstance(m_val, float): st.metric(label=m_name, value=f"{m_val:.3f}")
                                        else: st.metric(label=m_name, value=str(m_val))
                                
                                st.markdown(f"###### Mental Capacity Assignments for {iter_p_name}={param_disp_val_iter}")
                                assigned_df_iter = pd.DataFrame({'Mental Capacity': run_mental_labels, 'Cluster': specific_labels_iter})
                                st.dataframe(assigned_df_iter.sort_values(by=['Cluster', 'Mental Capacity']).reset_index(drop=True))

                                if current_run_algo_name == "Agglomerative Clustering":
                                    dendro_title = f"Dendrogram ({current_run_sub_type_value} linkage)"
                                    dendro_fig_iter = nf.plot_gradient_branch_dendrogram(
                                        run_data_for_clustering_transposed, linkage_method=current_run_sub_type_value, 
                                        title=dendro_title, orientation='left', # Orientation left for dendrogram
                                        labels_for_clusters=specific_labels_iter, # Pass labels for coloring attempt
                                        num_final_clusters=len(np.unique(specific_labels_iter))
                                    )
                                    st.plotly_chart(dendro_fig_iter, use_container_width=True, key=f"dendro_plot_{current_run_algo_name}_{current_run_sub_type_value}_{i}")
                                
                                if current_run_algo_name == "GMM" and iter_clustering_results[selected_param_for_details].get('gmm_probabilities') is not None:
                                    st.markdown(f"###### GMM Probabilistic Assignments for k={param_disp_val_iter}")
                                    gmm_probs_fig = nf.plot_gmm_cluster_probabilities_plotly(
                                        run_mental_labels, 
                                        iter_clustering_results[selected_param_for_details]['gmm_probabilities'],
                                        int(selected_param_for_details), # Ensure k is int
                                        title_suffix=f" ({current_run_sub_type_value})" if current_run_sub_type_name else ""
                                    )
                                    if gmm_probs_fig: st.plotly_chart(gmm_probs_fig, use_container_width=True, key=f"gmm_probs_{current_run_sub_type_value}_{i}")


                                # Individual Validation Plots (Silhouette vs Param, Negative Silhouette Bars)
                                # Placed after assignments and specific plots like dendrogram/GMM probs
                                if st.session_state.show_validation_plots: # Check global toggle
                                    st.markdown("###### Further Validation Plots for this Configuration")
                                    # 1. Silhouette vs Param
                                    sil_plot_fig = nf.plot_silhouette_scores_vs_param(
                                        run_data_for_clustering_transposed, iter_clustering_results, iter_p_name, run_mental_labels,
                                        title_suffix=f" ({current_run_algo_name} - {current_run_sub_type_value if current_run_sub_type_name else ''})"
                                    )
                                    if sil_plot_fig: st.plotly_chart(sil_plot_fig, use_container_width=True, key=f"sil_vs_param_{current_run_algo_name}_{current_run_sub_type_value}_{i}")
                                    
                                    # 2. Negative Silhouette Samples Bar Plot
                                    # Prepare labels_dict_all_methods for this specific iteration
                                    current_method_label_for_neg_sil = f"{current_run_algo_name} ({current_run_sub_type_value if current_run_sub_type_name else iter_p_name+'='+param_disp_val_iter})"
                                    labels_for_this_neg_sil_plot = {current_method_label_for_neg_sil: specific_labels_iter}
                                    param_str_for_this_neg_sil_plot = {current_method_label_for_neg_sil: f"{iter_p_name}={param_disp_val_iter}"}

                                    neg_sil_fig_plotly = nf.plot_negative_silhouette_samples_plotly_v2(
                                        run_data_for_clustering_transposed, 
                                        labels_for_this_neg_sil_plot,
                                        run_mental_labels,
                                        param_str_for_this_neg_sil_plot,
                                        title_prefix="" # Already included in method name
                                    )
                                    if neg_sil_fig_plotly: st.altair_chart(neg_sil_fig_plotly, use_container_width=True)


                        if not iter_clustering_results:
                             st.error(f"Clustering failed or no results for {current_run_algo_name} ({current_run_sub_type_value}).")
                
                # Overall Validation Metrics Comparison Plot (displayed once after all algorithm columns if toggle is on)
                if st.session_state.show_validation_plots and all_iter_clustering_results_map:
                    st.markdown("---"); st.subheader("Overall Validation Metrics Comparison (k-based algorithms)")
                    val_metrics_options = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
                    selected_val_metrics = st.multiselect("Select metrics to compare:", val_metrics_options, 
                                                          default=["Silhouette", "Calinski-Harabasz"], 
                                                          key=f"val_metrics_select_overall_{len(all_iter_clustering_results_map)}")
                    min_k_overall_run = 100; max_k_overall_run = 0; k_based_algos_were_run = False
                    for algo_key_cfg, params_val_run_cfg in st.session_state.run_config["algo_specific_params"].items():
                        if algo_key_cfg in st.session_state.run_config["selected_algorithms"] and algo_key_cfg in ["KMeans", "Agglomerative Clustering", "GMM"]:
                            k_based_algos_were_run = True
                            min_k_overall_run = min(min_k_overall_run, params_val_run_cfg.get('k_min',2))
                            max_k_overall_run = max(max_k_overall_run, params_val_run_cfg.get('k_max',5))
                    if not k_based_algos_were_run or min_k_overall_run > max_k_overall_run:
                        min_k_overall_run = 2; max_k_overall_run = 5 
                    
                    k_col1_comp, k_col2_comp = st.columns(2)
                    selected_min_k_comp_plot = k_col1_comp.slider("Min k for comparison plot", 2, max(2, max_k_overall_run-1 if max_k_overall_run > 2 else 2) , min_k_overall_run, key="comp_k_min_overall")
                    selected_max_k_comp_plot = k_col2_comp.slider("Max k for comparison plot", selected_min_k_comp_plot, max_k_ui, max(selected_min_k_comp_plot, max_k_overall_run), key="comp_k_max_overall")

                    if selected_val_metrics:
                        comparison_fig = nf.plot_validation_metrics_comparison_plotly(
                            run_data_for_clustering_transposed, all_iter_clustering_results_map,
                            selected_val_metrics, (selected_min_k_comp_plot, selected_max_k_comp_plot)
                        )
                        if comparison_fig: st.plotly_chart(comparison_fig, use_container_width=True)
                        else: st.info("Could not generate comparison validation plot.")
                    else: st.info("Select at least one metric to compare for the overall plot.")
else:
    results_container.info("Adjust parameters in the sidebar and click 'Run Analysis' to see results.")