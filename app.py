import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from typing import Dict, List, Callable, Any
import time

# ======================
# Streamlit Config
# ======================
st.set_page_config(layout="wide")
alt.data_transformers.disable_max_rows()

# ======================
# Original M functions
# ======================
def M_instantaneous(t: float) -> float:
    if t < 26:
        return 1008000
    elif t < 78:
        return 504000
    elif t < 182:
        return 252000
    elif t < 390:
        return 126000
    elif t < 806:
        return 63000
    else:
        return 0

def M(t: float) -> float:
    if t < 0:
        return 0
    if t >= 806:
        return 131040000
    
    if t < 26:
        return 1008000 * t
    if t < 78:
        return 26208000 + 504000 * (t - 26)
    if t < 182:
        return 52416000 + 252000 * (t - 78)
    if t < 390:
        return 78624000 + 126000 * (t - 182)
    
    return 104832000 + 63000 * (t - 390)

# =====================================================
# 1. Homogeneous Poisson Process (existing function)
# =====================================================
def generate_poisson_process(
    total_weeks: int,
    mean_interarrival_time: float,
    random_seed: int = 1234
) -> List[int]:
    """
    Generate a homogeneous Poisson process (fixed rate = 1/mean_interarrival_time).
    Return G(t) = number of arrivals by time t, for t = 1,...,total_weeks.
    """
    np.random.seed(random_seed)
    
    # scale = mean_interarrival_time => mean of Exponential
    interarrival_times = np.random.exponential(
        scale=mean_interarrival_time,
        size=total_weeks * 2  # overshoot to be safe
    )
    arrival_times = np.cumsum(interarrival_times)
    
    G = []
    for t in range(1, total_weeks + 1):
        count = np.sum(arrival_times <= t)
        G.append(int(count))
    
    return G

# =====================================================
# 2. Non-homogeneous Poisson Process (NEW)
# =====================================================
def generate_poisson_process_nonhomogeneous(
    total_weeks: int,
    rate_func: Callable[[float], float],
    random_seed: int = 1234,
    num_samples_for_max_rate: int = 1000
) -> List[int]:
    """
    Generate a non-homogeneous Poisson process for t in [0, total_weeks]
    using the 'thinning' method. The rate function is given by `rate_func(t)`.

    Returns G(t) = number of arrivals by time t, for t = 1,...,total_weeks.
    """
    np.random.seed(random_seed)
    
    # 1) Estimate lambda_max by sampling the rate function over [0, total_weeks]
    sample_points = np.linspace(0, total_weeks, num_samples_for_max_rate + 1)
    lambda_vals = [rate_func(x) for x in sample_points]
    lambda_max = max(lambda_vals)
    
    # 2) Generate candidate events from a Homogeneous Poisson Process with rate = lambda_max
    arrival_times_candidate = []
    current_time = 0.0
    while True:
        # Exponential interarrival with mean = 1 / lambda_max
        interarrival = np.random.exponential(1.0 / lambda_max)
        current_time += interarrival
        if current_time > total_weeks:
            break
        arrival_times_candidate.append(current_time)
    
    # 3) Thinning step: accept arrival at time s with prob = rate_func(s)/lambda_max
    arrival_times = []
    for s in arrival_times_candidate:
        if np.random.rand() < (rate_func(s) / lambda_max):
            arrival_times.append(s)
    
    arrival_times = np.array(arrival_times)
    
    # 4) Build G(t) for t = 1,...,total_weeks
    G = []
    for t in range(1, total_weeks + 1):
        count = np.sum(arrival_times <= t)
        G.append(int(count))
    
    return G

# ======================
# Gamma functions
# ======================
def gamma_function_power(n: int, alpha: float) -> float:
    return (n ** alpha) / (1 + n ** alpha)

def gamma_logistic(n: int, b: float, c: float) -> float:
    return 1 / (1 + np.exp(-b * (n - c)))

def gamma_exponential(n: int, b: float) -> float:
    return 1-np.exp(-b * n)

def gamma_normalized_log(n: int, a: float, N_ref: float) -> float:
    return np.log(1+a*n) / np.log(1+a*N_ref)

# ======================================================
# run_simulation
# ======================================================
def run_simulation(
    total_weeks: int,
    mean_interarrival_time: float,
    gamma_func,
    gamma_func_params: tuple,
    random_seed: int = 2345,
    use_nonhomogeneous: bool = False,
    custom_rate_func: Callable[[float], float] = None
) -> Dict[str, Any]:
    """
    Runs one simulation path. 
    - If `use_nonhomogeneous` is False, generates G via homogeneous Poisson process 
      with mean_interarrival_time.
    - If True, uses the provided custom_rate_func(t) for a non-homogeneous Poisson.
    """
    results = {
        "time": np.arange(1, total_weeks + 1),
        "M": np.zeros(total_weeks),
        "G": np.zeros(total_weeks),
        "gamma": np.zeros(total_weeks),
        "vault": np.zeros(total_weeks),
        "minted": np.zeros(total_weeks),
        "redirected": np.zeros(total_weeks),
        "vault_drained": np.zeros(total_weeks),
        "cumulative_vault": np.zeros(total_weeks),
        "cumulative_minted": np.zeros(total_weeks),
        "cumulative_drained": np.zeros(total_weeks)
    }
    
    # --------------------------------------------------
    # Generate Poisson process G(t)
    # --------------------------------------------------
    if not use_nonhomogeneous:
        # Homogeneous
        G_vals = generate_poisson_process(total_weeks, mean_interarrival_time, random_seed)
    else:
        # Non-homogeneous
        G_vals = generate_poisson_process_nonhomogeneous(total_weeks, custom_rate_func, random_seed)
    
    results["G"] = np.array(G_vals)
    
    # --------------------------------------------------
    # Apply your existing "vault / minted" logic
    # --------------------------------------------------
    vault_total = 0
    minted_total = 0
    drained_total = 0
    start_draining_vault = False
    mint_amt = 0
    
    for t in range(1, total_weeks + 1):
        idx = t - 1
        
        # M(t)
        results["M"][idx] = M(t)
        delta_M = results["M"][idx] - (M(t-1) if t > 1 else 0)
        
        # gamma
        g_val = results["G"][idx]
        results["gamma"][idx] = gamma_func(g_val, *gamma_func_params)
        
        # Start draining if M(t) stops increasing
        if not start_draining_vault and delta_M == 0:
            start_draining_vault = True
            
        if start_draining_vault:
            # Drain from vault
            drain_amount = min(mint_amt, vault_total)
            results["vault_drained"][idx] = drain_amount
            vault_total -= drain_amount
            drained_total += drain_amount
            
            # The drained amount is effectively minted
            results["minted"][idx] += drain_amount
            minted_total += drain_amount
        else:
            # If still minting
            redirected = (1-results["gamma"][idx]) * delta_M if delta_M > 0 else 0
            results["redirected"][idx] = redirected
            
            results["vault"][idx] = redirected
            results["minted"][idx] = delta_M - redirected
            
            vault_total += redirected
            mint_amt = delta_M - redirected
            minted_total += mint_amt
        
        # Update cumulative
        results["cumulative_vault"][idx] = vault_total
        results["cumulative_minted"][idx] = minted_total
        results["cumulative_drained"][idx] = drained_total
    
    return results


# ======================================================
# run_monte_carlo_simulations
# ======================================================
def run_monte_carlo_simulations(
    n_simulations: int,
    total_weeks: int,
    mean_interarrival_time: float,
    gamma_func: Callable,
    gamma_func_params: tuple,
    progress_callback: Callable[[float], None] = None,
    random_seed: int = 2345,
    use_nonhomogeneous: bool = False,
    custom_rate_func: Callable[[float], float] = None
) -> Dict[str, Any]:
    """
    Repeatedly calls run_simulation() to produce multiple paths and 
    aggregate median and confidence intervals.
    """
    all_simulations = []
    
    for i in range(n_simulations):
        seed = random_seed + i
        result = run_simulation(
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_func,
            gamma_func_params=gamma_func_params,
            random_seed=seed,
            use_nonhomogeneous=use_nonhomogeneous,
            custom_rate_func=custom_rate_func
        )
        all_simulations.append(result)
        
        if progress_callback:
            progress_callback((i + 1) / n_simulations)
    
    aggregated_results = {
        "time": np.arange(1, total_weeks + 1),
        "M": np.zeros(total_weeks),
        "median": {},
        "lower_ci": {},
        "upper_ci": {}
    }
    
    # We can use M from the first simulation (same for all if M(t) is deterministic).
    aggregated_results["M"] = all_simulations[0]["M"]
    
    keys_for_stats = ["G", "gamma", "vault", "minted", "redirected", "vault_drained",
                      "cumulative_vault", "cumulative_minted", "cumulative_drained"]
    
    for key in keys_for_stats:
        all_values = np.array([sim[key] for sim in all_simulations])
        
        aggregated_results["median"][key] = np.median(all_values, axis=0)
        aggregated_results["lower_ci"][key] = np.percentile(all_values, 5, axis=0)
        aggregated_results["upper_ci"][key] = np.percentile(all_values, 95, axis=0)
    
    return aggregated_results

# ======================================================
# Plotting function
# ======================================================
def plot_monte_carlo_results(power_results: Dict[str, Any], logistic_results: Dict[str, Any], exponential_results: Dict[str, Any], normalized_log_results: Dict[str, Any]) -> Dict[str, alt.Chart]:
    """
    Creates Altair charts for the results. 
    For brevity, this is unchanged except for referencing the existing results.
    """
    power_color = "green"
    logistic_color = "purple"
    exponential_color = "red"
    normalized_log_color = "orange"
    
    # ========== Chart 1: Emission Schedule ==========
    emission_data = pd.DataFrame({
        'Week': power_results["time"],
        'M(t)': power_results["M"]/1e6,
        'Power_M*(t)': power_results["median"]["cumulative_minted"]/1e6,
        'Power_Lower_CI': power_results["lower_ci"]["cumulative_minted"]/1e6,
        'Power_Upper_CI': power_results["upper_ci"]["cumulative_minted"]/1e6,
        'Logistic_M*(t)': logistic_results["median"]["cumulative_minted"]/1e6,
        'Logistic_Lower_CI': logistic_results["lower_ci"]["cumulative_minted"]/1e6,
        'Logistic_Upper_CI': logistic_results["upper_ci"]["cumulative_minted"]/1e6,
        'Exponential_M*(t)': exponential_results["median"]["cumulative_minted"]/1e6,
        'Exponential_Lower_CI': exponential_results["lower_ci"]["cumulative_minted"]/1e6,
        'Exponential_Upper_CI': exponential_results["upper_ci"]["cumulative_minted"]/1e6,
        'Normalized_Log_M*(t)': normalized_log_results["median"]["cumulative_minted"]/1e6,
        'Normalized_Log_Lower_CI': normalized_log_results["lower_ci"]["cumulative_minted"]/1e6,
        'Normalized_Log_Upper_CI': normalized_log_results["upper_ci"]["cumulative_minted"]/1e6
    })
    
    emission_base = alt.Chart(emission_data).encode(
        x=alt.X('Week:Q', title='Weeks')
    )
    
    emission_line_base = emission_base.mark_line(color='blue').encode(
        y=alt.Y('M(t):Q', title='Cumulative M-TIG Minted'),
        tooltip=['Week:Q', 'M(t):Q']
    )
    
    power_emission_line = emission_base.mark_line(color=power_color).encode(
        y='Power_M*(t):Q',
        tooltip=['Week:Q', 'Power_M*(t):Q']
    )
    power_emission_area = emission_base.mark_area(opacity=0.3, color=power_color).encode(
        y='Power_Lower_CI:Q',
        y2='Power_Upper_CI:Q'
    )
    
    logistic_emission_line = emission_base.mark_line(color=logistic_color).encode(
        y='Logistic_M*(t):Q',
        tooltip=['Week:Q', 'Logistic_M*(t):Q']
    )
    logistic_emission_area = emission_base.mark_area(opacity=0.3, color=logistic_color).encode(
        y='Logistic_Lower_CI:Q',
        y2='Logistic_Upper_CI:Q'
    )

    exponential_emission_line = emission_base.mark_line(color=exponential_color).encode(
        y='Exponential_M*(t):Q',
        tooltip=['Week:Q', 'Exponential_M*(t):Q']
    )
    exponential_emission_area = emission_base.mark_area(opacity=0.3, color=exponential_color).encode(
        y='Exponential_Lower_CI:Q',
        y2='Exponential_Upper_CI:Q'
    )

    normalized_log_emission_line = emission_base.mark_line(color=normalized_log_color).encode(
        y='Normalized_Log_M*(t):Q',
        tooltip=['Week:Q', 'Normalized_Log_M*(t):Q']
    )
    normalized_log_emission_area = emission_base.mark_area(opacity=0.3, color=normalized_log_color).encode(
        y='Normalized_Log_Lower_CI:Q',
        y2='Normalized_Log_Upper_CI:Q'
    )

    emission_melt = pd.melt(
        emission_data, 
        id_vars=['Week'], 
        value_vars=['M(t)', 'Power_M*(t)', 'Logistic_M*(t)', 'Exponential_M*(t)'],
        var_name='Series', 
        value_name='Value'
    )
    
    emission_legend = alt.Chart(emission_melt).mark_line().encode(
        x='Week:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['M(t)', 'Power_M*(t)', 'Logistic_M*(t)', 'Exponential_M*(t)', 'Normalized_Log_M*(t)'], 
                                       range=['blue', power_color, logistic_color, exponential_color, normalized_log_color]),
                       legend=alt.Legend(title='', orient='top')),
    )
    
    emission_chart = alt.layer(
        power_emission_area, logistic_emission_area, exponential_emission_area, normalized_log_emission_area,
        emission_line_base, power_emission_line, logistic_emission_line, exponential_emission_line, normalized_log_emission_line,
        emission_legend
    ).properties(
        width=400, 
        height=300,
        title='Token Emission Schedule'
    )
    
    # ========== Chart 2: Event count (G) ==========
    event_data = pd.DataFrame({
        'Week': power_results["time"],
        'Median_G': power_results["median"]["G"],
        'Lower_G': power_results["lower_ci"]["G"],
        'Upper_G': power_results["upper_ci"]["G"]
    })
    
    event_base = alt.Chart(event_data).encode(
        x=alt.X('Week:Q', title='Weeks'),
        y=alt.Y('Median_G:Q', title='Total Challenges')
    )
    event_area = event_base.mark_area(opacity=0.3, color='red').encode(
        y=alt.Y('Lower_G:Q'),
        y2=alt.Y2('Upper_G:Q')
    )
    event_line = event_base.mark_line(color='red').encode(
        tooltip=['Week:Q', 'Median_G:Q']
    )
    
    event_chart = alt.layer(
        event_area, event_line
    ).properties(
        width=400, 
        height=300,
        title='Total Challenges'
    )
    
    # ========== Chart 3: Gamma chart ==========
    gamma_data = pd.DataFrame({
        'Week': power_results["time"],
        'PowerFn': power_results["median"]["gamma"],
        'PowerFn_Lower_CI': power_results["lower_ci"]["gamma"],
        'PowerFn_Upper_CI': power_results["upper_ci"]["gamma"],
        'LogisticFn': logistic_results["median"]["gamma"],
        'LogisticFn_Lower_CI': logistic_results["lower_ci"]["gamma"],
        'LogisticFn_Upper_CI': logistic_results["upper_ci"]["gamma"],
        'ExponentialFn': exponential_results["median"]["gamma"],
        'ExponentialFn_Lower_CI': exponential_results["lower_ci"]["gamma"],
        'ExponentialFn_Upper_CI': exponential_results["upper_ci"]["gamma"],
        'Normalized_LogFn': normalized_log_results["median"]["gamma"],
        'Normalized_LogFn_Lower_CI': normalized_log_results["lower_ci"]["gamma"],
        'Normalized_LogFn_Upper_CI': normalized_log_results["upper_ci"]["gamma"]
    })
    
    gamma_base = alt.Chart(gamma_data).encode(
        x=alt.X('Week:Q', title='Weeks')
    )
    
    power_gamma_area = gamma_base.mark_area(opacity=0.3, color=power_color).encode(
        y=alt.Y('PowerFn_Lower_CI:Q'),
        y2=alt.Y2('PowerFn_Upper_CI:Q')
    )
    power_gamma_line = gamma_base.mark_line(color=power_color).encode(
        y=alt.Y('PowerFn:Q', title='Goal Progress'),
        tooltip=['Week:Q', 'PowerFn:Q']
    )
    
    logistic_gamma_area = gamma_base.mark_area(opacity=0.3, color=logistic_color).encode(
        y=alt.Y('LogisticFn_Lower_CI:Q'),
        y2=alt.Y2('LogisticFn_Upper_CI:Q')
    )
    logistic_gamma_line = gamma_base.mark_line(color=logistic_color).encode(
        y=alt.Y('LogisticFn:Q'),
        tooltip=['Week:Q', 'LogisticFn:Q']
    )

    exponential_gamma_line = gamma_base.mark_line(color=exponential_color).encode(
        y=alt.Y('ExponentialFn:Q'),
        tooltip=['Week:Q', 'ExponentialFn:Q']
    )
    exponential_gamma_area = gamma_base.mark_area(opacity=0.3, color=exponential_color).encode(
        y=alt.Y('ExponentialFn_Lower_CI:Q'),
        y2=alt.Y2('ExponentialFn_Upper_CI:Q')
    )

    normalized_log_gamma_line = gamma_base.mark_line(color=normalized_log_color).encode(
        y=alt.Y('Normalized_LogFn:Q'),
        tooltip=['Week:Q', 'Normalized_LogFn:Q']
    )
    normalized_log_gamma_area = gamma_base.mark_area(opacity=0.3, color=normalized_log_color).encode(
        y=alt.Y('Normalized_LogFn_Lower_CI:Q'),
        y2=alt.Y2('Normalized_LogFn_Upper_CI:Q')
    )
    
    gamma_melt = pd.melt(
        gamma_data, 
        id_vars=['Week'], 
        value_vars=['PowerFn', 'LogisticFn', 'ExponentialFn'],
        var_name='Series', 
        value_name='Value'
    )
    
    gamma_legend = alt.Chart(gamma_melt).mark_line().encode(
        x='Week:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['PowerFn', 'LogisticFn', 'ExponentialFn'], 
                                       range=[power_color, logistic_color, exponential_color]),
                       legend=alt.Legend(title='', orient='top')),
    )
    
    gamma_chart = alt.layer(
        power_gamma_area, power_gamma_line,
        logistic_gamma_area, logistic_gamma_line,
        exponential_gamma_area, exponential_gamma_line,
        normalized_log_gamma_area, normalized_log_gamma_line,
        gamma_legend
    ).properties(
        width=400, 
        height=300,
        title='Goal Progress'
    )
    
    # ========== Chart 4: Cumulative tokens (vault) ==========
    token_data = pd.DataFrame({
        'Week': power_results["time"],
        'Power_Vault': power_results["median"]["cumulative_vault"]/1e6,
        'Power_Lower_Vault': power_results["lower_ci"]["cumulative_vault"]/1e6,
        'Power_Upper_Vault': power_results["upper_ci"]["cumulative_vault"]/1e6,
        'Power_Drained': power_results["median"]["cumulative_drained"]/1e6,
        'Power_Lower_Drained': power_results["lower_ci"]["cumulative_drained"]/1e6,
        'Power_Upper_Drained': power_results["upper_ci"]["cumulative_drained"]/1e6,
        'Logistic_Vault': logistic_results["median"]["cumulative_vault"]/1e6,
        'Logistic_Lower_Vault': logistic_results["lower_ci"]["cumulative_vault"]/1e6,
        'Logistic_Upper_Vault': logistic_results["upper_ci"]["cumulative_vault"]/1e6,
        'Logistic_Drained': logistic_results["median"]["cumulative_drained"]/1e6,
        'Logistic_Lower_Drained': logistic_results["lower_ci"]["cumulative_drained"]/1e6,
        'Logistic_Upper_Drained': logistic_results["upper_ci"]["cumulative_drained"]/1e6,
        'Exponential_Vault': exponential_results["median"]["cumulative_vault"]/1e6,
        'Exponential_Lower_Vault': exponential_results["lower_ci"]["cumulative_vault"]/1e6,
        'Exponential_Upper_Vault': exponential_results["upper_ci"]["cumulative_vault"]/1e6,
        'Exponential_Drained': exponential_results["median"]["cumulative_drained"]/1e6,
        'Exponential_Lower_Drained': exponential_results["lower_ci"]["cumulative_drained"]/1e6,
        'Normalized_Log_Vault': normalized_log_results["median"]["cumulative_vault"]/1e6,
        'Normalized_Log_Lower_Vault': normalized_log_results["lower_ci"]["cumulative_vault"]/1e6,
        'Normalized_Log_Upper_Vault': normalized_log_results["upper_ci"]["cumulative_vault"]/1e6,
        'Normalized_Log_Drained': normalized_log_results["median"]["cumulative_drained"]/1e6,
        'Normalized_Log_Lower_Drained': normalized_log_results["lower_ci"]["cumulative_drained"]/1e6,
        'Normalized_Log_Upper_Drained': normalized_log_results["upper_ci"]["cumulative_drained"]/1e6
    })
    
    token_base = alt.Chart(token_data).encode(
        x=alt.X('Week:Q', title='Weeks')
    )
    
    # Power Vault
    power_vault_area = token_base.mark_area(opacity=0.3, color=power_color).encode(
        y=alt.Y('Power_Lower_Vault:Q'),
        y2=alt.Y2('Power_Upper_Vault:Q')
    )
    power_vault_line = token_base.mark_line(color=power_color).encode(
        y=alt.Y('Power_Vault:Q', title='M-TIG')
    )
    # Drained (Power)
    power_drained_area = token_base.mark_area(opacity=0.3, color='orange', fillOpacity=0.1).encode(
        y=alt.Y('Power_Lower_Drained:Q'),
        y2=alt.Y2('Power_Upper_Drained:Q')
    )
    power_drained_line = token_base.mark_line(color='orange', strokeDash=[8, 4]).encode(
        y=alt.Y('Power_Drained:Q')
    )
    
    # Logistic Vault
    logistic_vault_area = token_base.mark_area(opacity=0.3, color=logistic_color).encode(
        y=alt.Y('Logistic_Lower_Vault:Q'),
        y2=alt.Y2('Logistic_Upper_Vault:Q')
    )
    logistic_vault_line = token_base.mark_line(color=logistic_color).encode(
        y=alt.Y('Logistic_Vault:Q')
    )
    # Drained (Logistic)
    logistic_drained_area = token_base.mark_area(opacity=0.3, color='blue', fillOpacity=0.1).encode(
        y=alt.Y('Logistic_Lower_Drained:Q'),
        y2=alt.Y2('Logistic_Upper_Drained:Q')
    )
    logistic_drained_line = token_base.mark_line(color='blue', strokeDash=[8, 4]).encode(
        y=alt.Y('Logistic_Drained:Q')
    )

    # Exponential Vault
    exponential_vault_area = token_base.mark_area(opacity=0.3, color=exponential_color).encode(
        y=alt.Y('Exponential_Lower_Vault:Q'),
        y2=alt.Y2('Exponential_Upper_Vault:Q')
    )
    exponential_vault_line = token_base.mark_line(color=exponential_color).encode(
        y=alt.Y('Exponential_Vault:Q')
    )
    # Drained (Exponential)
    exponential_drained_area = token_base.mark_area(opacity=0.3, color='green', fillOpacity=0.1).encode(
        y=alt.Y('Exponential_Lower_Drained:Q'),
        y2=alt.Y2('Exponential_Upper_Drained:Q')
    )
    exponential_drained_line = token_base.mark_line(color='green', strokeDash=[8, 4]).encode(
        y=alt.Y('Exponential_Drained:Q')
    )

    # Normalized Log Vault
    normalized_log_vault_area = token_base.mark_area(opacity=0.3, color=normalized_log_color).encode(
        y=alt.Y('Normalized_Log_Lower_Vault:Q'),
        y2=alt.Y2('Normalized_Log_Upper_Vault:Q')
    )
    normalized_log_vault_line = token_base.mark_line(color=normalized_log_color).encode(
        y=alt.Y('Normalized_Log_Vault:Q')
    )
    # Drained (Normalized Log)
    normalized_log_drained_area = token_base.mark_area(opacity=0.3, color='purple', fillOpacity=0.1).encode(
        y=alt.Y('Normalized_Log_Lower_Drained:Q'),
        y2=alt.Y2('Normalized_Log_Upper_Drained:Q')
    )
    normalized_log_drained_line = token_base.mark_line(color='purple', strokeDash=[8, 4]).encode(
        y=alt.Y('Normalized_Log_Drained:Q')
    )

    token_melt = pd.melt(
        token_data, 
        id_vars=['Week'], 
        value_vars=['Power_Vault', 'Power_Drained', 'Logistic_Vault', 'Logistic_Drained', 'Exponential_Vault', 'Exponential_Drained', 'Normalized_Log_Vault', 'Normalized_Log_Drained'],
        var_name='Series', 
        value_name='Value'
    )
    
    token_legend = alt.Chart(token_melt).mark_line().encode(
        x='Week:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['Power_Vault','Power_Drained','Logistic_Vault','Logistic_Drained', 'Exponential_Vault', 'Exponential_Drained', 'Normalized_Log_Vault', 'Normalized_Log_Drained'],
                                       range=[power_color, 'orange', logistic_color, 'blue', exponential_color, 'green', normalized_log_color, 'purple']),
                       legend=alt.Legend(title='', orient='top')),
        strokeDash=alt.condition(
            (alt.datum.Series == 'Power_Drained') | (alt.datum.Series == 'Logistic_Drained'),
            alt.value([8, 4]),
            alt.value([0])
        )
    )
    
    token_chart = alt.layer(
        power_vault_area, power_vault_line,
        power_drained_area, power_drained_line,
        logistic_vault_area, logistic_vault_line,
        logistic_drained_area, logistic_drained_line,
        exponential_vault_area, exponential_vault_line,
        exponential_drained_area, exponential_drained_line,
        normalized_log_vault_area, normalized_log_vault_line,
        normalized_log_drained_area, normalized_log_drained_line,
        token_legend
    ).properties(
        width=400, 
        height=300,
        title='Vault Dynamics'
    )
    
    # Layout: 2x2
    top_row = alt.hconcat(emission_chart, token_chart, spacing=0)
    bottom_row = alt.hconcat(event_chart, gamma_chart, spacing=0)
    final_chart = alt.vconcat(top_row, bottom_row, spacing=0)
    
    return {
        "emission_chart": emission_chart,
        "gamma_event_chart": event_chart,
        "token_chart": token_chart,
        "gamma_chart": gamma_chart,
        "final_chart": final_chart
    }


# ======================================================
# Main Streamlit App
# ======================================================
def main():
    st.title("TIG Emission Monte Carlo Simulation")
    
    # ------------- Sidebar ------------------
    st.sidebar.header("Simulation Parameters")
    
    # 1) Choose homogeneous vs non-homogeneous
    process_type = st.sidebar.radio("Poisson Process Type:", ["Non-homogeneous", "Homogeneous"], index=0)
    
    total_weeks = st.sidebar.slider("Total Weeks", min_value=800, max_value=2000, value=1500, step=52)
    
    if process_type == "Homogeneous":
        # We keep the same "mean_interarrival_time" approach
        mean_interarrival_time = st.sidebar.slider("Mean Interarrival Time (weeks)", 
                                                   min_value=1.0, max_value=52.0, value=10.0, step=1.0)
        custom_rate_func = None
    else:
        # Non-homogeneous: user enters a Python expression for lambda(t)
        st.sidebar.write("Define your rate function λ(t). For example:")
        st.sidebar.code("0.1*np.sin(t)**2", language="python")
        
        rate_str = st.sidebar.text_input("λ(t) =", value="0.1*np.sin(t)**2")
        
        # For safety, define a small function that returns eval of the string
        def user_rate_func(t):
            return eval(rate_str, {"t": t, "np": np, "__builtins__": {}})
        
        custom_rate_func = user_rate_func
        # We won't use mean_interarrival_time in non-homogeneous
        mean_interarrival_time = 1.0  # dummy, not used
    
    # 2) Gamma Function parameters
    st.sidebar.subheader("Power Law Parameters")
    alpha = st.sidebar.slider("Alpha Parameter", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    
    st.sidebar.subheader("Logistic Parameters")
    b = st.sidebar.slider("b-Logistic", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    c = st.sidebar.slider("c-Logistic", min_value=10, max_value=200, value=50, step=5)
    
    st.sidebar.subheader("Exponential Parameters")
    b_exp = st.sidebar.slider("b-Exponential", min_value=0.01, max_value=0.99, value=0.10, step=0.01)

    st.sidebar.subheader("Normalized Log Parameters")
    a_log = st.sidebar.slider("a", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    Nref_log = st.sidebar.slider("Nref", min_value=100, max_value=500, value=250, step=5)
    
    st.sidebar.subheader("Monte Carlo Settings")
    n_simulations = st.sidebar.slider("Number of Simulations", min_value=10, max_value=1000, value=100, step=10)
    
    run_button = st.sidebar.button("Run Simulations")
    
    # ------------- Run simulations on click ------------------
    if run_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
        
        start_time = time.time()
        
        # ===== Run simulations for Power Law =====
        status_text.text("Running Monte Carlo simulations for Power Law...")
        power_law_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_function_power,
            gamma_func_params=(alpha,),
            progress_callback=lambda p: update_progress(p * 0.5),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func
        )
        
        # ===== Run simulations for Logistic =====
        status_text.text("Running Monte Carlo simulations for Logistic...")
        logistic_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_logistic,
            gamma_func_params=(b, c),
            progress_callback=lambda p: update_progress(0.5 + p * 0.5),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func
        )

        # ===== Run simulations for Exponential =====
        status_text.text("Running Monte Carlo simulations for Exponential...")
        exponential_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_exponential,
            gamma_func_params=(b_exp,),
            progress_callback=lambda p: update_progress(0.5 + p * 0.5),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func
        )

        # ===== Run simulations for Normalized Log =====
        status_text.text("Running Monte Carlo simulations for Normalized Log...")
        normalized_log_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_normalized_log,
            gamma_func_params=(a_log, Nref_log),
            progress_callback=lambda p: update_progress(0.5 + p * 0.5),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func
        )
        
        
        end_time = time.time()
        status_text.text(f"Simulations completed in {end_time - start_time:.2f} seconds")
        
        # ------------- Display results -------------
        # st.subheader("Simulation Results")
        charts = plot_monte_carlo_results(power_law_results, logistic_results, exponential_results, normalized_log_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(charts["emission_chart"], use_container_width=True)
        with col2:
            st.altair_chart(charts["token_chart"], use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.altair_chart(charts["gamma_event_chart"], use_container_width=True)
        with col4:
            st.altair_chart(charts["gamma_chart"], use_container_width=True)
        
        st.subheader("Key Metrics at End of Simulation")
        
        # ------ Power Law metrics ------
        st.markdown("### Power Law")
        metrics_cols_power = st.columns(3)
        with metrics_cols_power[0]:
            st.metric("Median Total Minted (M-TIG)", 
                      f"{power_law_results['median']['cumulative_minted'][-1]/1e6:.2f}M")
        with metrics_cols_power[1]:
            st.metric("Median Vault Balance (M-TIG)", 
                      f"{power_law_results['median']['cumulative_vault'][-1]/1e6:.2f}M")
        with metrics_cols_power[2]:
            st.metric("Median Drained (M-TIG)", 
                      f"{power_law_results['median']['cumulative_drained'][-1]/1e6:.2f}M")
        
        # ------ Logistic metrics ------
        st.markdown("### Logistic")
        metrics_cols_logistic = st.columns(3)
        with metrics_cols_logistic[0]:
            st.metric("Median Total Minted (M-TIG)", 
                      f"{logistic_results['median']['cumulative_minted'][-1]/1e6:.2f}M")
        with metrics_cols_logistic[1]:
            st.metric("Median Vault Balance (M-TIG)", 
                      f"{logistic_results['median']['cumulative_vault'][-1]/1e6:.2f}M")
        with metrics_cols_logistic[2]:
            st.metric("Median Drained (M-TIG)", 
                      f"{logistic_results['median']['cumulative_drained'][-1]/1e6:.2f}M")

        # ------ Exponential metrics ------
        st.markdown("### Exponential")
        metrics_cols_exponential = st.columns(3)
        with metrics_cols_exponential[0]:
            st.metric("Median Total Minted (M-TIG)", 
                      f"{exponential_results['median']['cumulative_minted'][-1]/1e6:.2f}M")
        with metrics_cols_exponential[1]:
            st.metric("Median Vault Balance (M-TIG)", 
                      f"{exponential_results['median']['cumulative_vault'][-1]/1e6:.2f}M")
        with metrics_cols_exponential[2]:
            st.metric("Median Drained (M-TIG)", 
                      f"{exponential_results['median']['cumulative_drained'][-1]/1e6:.2f}M")

if __name__ == "__main__":
    main()
