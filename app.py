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
# 1. Homogeneous Poisson Process (updated to include initial_value)
# =====================================================
def generate_poisson_process(
    total_weeks: int,
    mean_interarrival_time: float,
    random_seed: int = 1234,
    initial_value: int = 0,
    vault_start_week: int = 0
) -> List[int]:
    np.random.seed(random_seed)
    
    interarrival_times = np.random.exponential(
        scale=mean_interarrival_time,
        size=total_weeks * 2
    )
    arrival_times = np.cumsum(interarrival_times)
    
    G = []
    for t in range(1, total_weeks + 1):
        if vault_start_week <= 0 or t >= vault_start_week:
            # After vault start: use Poisson process with initial value
            count = np.sum(arrival_times <= t)
            G.append(int(count) + initial_value)
        else:
            # Before vault start: linear growth from 0 to initial_value
            G.append(int(initial_value * t / vault_start_week))
    
    return G

# =====================================================
# 2. Non-homogeneous Poisson Process (updated to include initial_value)
# =====================================================
def generate_poisson_process_nonhomogeneous(
    total_weeks: int,
    rate_func: Callable[[float], float],
    random_seed: int = 1234,
    num_samples_for_max_rate: int = 1000,
    initial_value: int = 0,
    vault_start_week: int = 0
) -> List[int]:
    np.random.seed(random_seed)
    
    # Get max rate for rejection sampling
    sample_points = np.linspace(0, total_weeks, num_samples_for_max_rate + 1)
    lambda_vals = [rate_func(x) for x in sample_points]
    lambda_max = max(lambda_vals)
    
    # Generate candidate arrival times using homogeneous Poisson process
    arrival_times_candidate = []
    current_time = 0.0
    while True:
        interarrival = np.random.exponential(1.0 / lambda_max)
        current_time += interarrival
        if current_time > total_weeks:
            break
        arrival_times_candidate.append(current_time)
    
    # Thin using rejection sampling
    arrival_times = []
    for s in arrival_times_candidate:
        if np.random.rand() < (rate_func(s) / lambda_max):
            arrival_times.append(s)
    
    arrival_times = np.array(arrival_times)
    
    # Generate the cumulative counts
    G = []
    
    if vault_start_week <= 0:
        # If vault starts immediately or doesn't exist
        for t in range(1, total_weeks + 1):
            count = np.sum(arrival_times <= t)
            G.append(int(count) + initial_value)
    else:
        # Before vault start: linear growth from 0 to initial_value
        for t in range(1, vault_start_week):
            G.append(int(initial_value * t / vault_start_week))
        
        # At vault start week: exactly the initial_value (smooth transition)
        G.append(initial_value)
        
        # After vault start: continue Poisson process from this point
        # Only count arrivals after vault_start_week
        post_vault_arrivals = arrival_times[arrival_times > vault_start_week]
        for t in range(vault_start_week + 1, total_weeks + 1):
            # Add new arrivals to the initial_value
            new_arrivals = np.sum((vault_start_week < post_vault_arrivals) & (post_vault_arrivals <= t))
            G.append(initial_value + int(new_arrivals))
    
    return G

# ======================
# Gamma functions
# ======================
def gamma_function_power(n: int, alpha: float, n_offset: int = 0, y_offset: float = 0) -> float:
    if n < n_offset:
        # linearly increase from 0 to y_offset
        return y_offset + (n-n_offset) * (y_offset - 0) / (n_offset - 0)    
    return ((n-n_offset) ** alpha) / (1 + (n-n_offset) ** alpha) + y_offset

def gamma_logistic(n: int, b: float, c: float, n_offset: int = 0, y_offset: float = 0) -> float:
    if n < n_offset:
        # linearly increase from 0 to y_offset
        return y_offset + (n-n_offset) * (y_offset - 0) / (n_offset - 0)
    return 1 / (1 + np.exp(-b * (n - c - n_offset))) + y_offset

def gamma_exponential(n: int, b: float, n_offset: int = 0, y_offset: float = 0) -> float:
    if n < n_offset:
        # linearly increase from 0 to y_offset
        return y_offset + (n-n_offset) * (y_offset - 0) / (n_offset - 0)
    return 1-np.exp(-b * (n - n_offset)) + y_offset

def gamma_normalized_log(n: int, a: float, N_ref: float, n_offset: int = 0, y_offset: float = 0) -> float:
    if n < n_offset:
        # linearly increase from 0 to y_offset
        return y_offset + (n-n_offset) * (y_offset - 0) / (n_offset - 0)
    return np.log(1+a*(n-n_offset)) / np.log(1+a*N_ref) + y_offset

# ======================================================
# run_simulation (updated to include vault_start_week)
# ======================================================
def run_simulation(
    total_weeks: int,
    mean_interarrival_time: float,
    gamma_func,
    gamma_func_params: tuple,
    random_seed: int = 2345,
    use_nonhomogeneous: bool = False,
    custom_rate_func: Callable[[float], float] = None,
    initial_poisson_value: int = 0,
    vault_start_week: int = 0
) -> Dict[str, Any]:
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
        "cumulative_drained": np.zeros(total_weeks),
        "vault_start_week": vault_start_week
    }
    
    # Generate Poisson process G(t) with initial value and linear growth until vault_start_week
    if not use_nonhomogeneous:
        G_vals = generate_poisson_process(
            total_weeks, 
            mean_interarrival_time, 
            random_seed, 
            initial_poisson_value,
            vault_start_week
        )
    else:
        G_vals = generate_poisson_process_nonhomogeneous(
            total_weeks, 
            custom_rate_func, 
            random_seed,
            initial_value=initial_poisson_value,
            vault_start_week=vault_start_week
        )
    
    results["G"] = np.array(G_vals)
    
    # Apply vault / minted logic with vault start date
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
        
        # Check if vault is active (only when current week >= vault_start_week)
        vault_active = vault_start_week > 0 and t >= vault_start_week
        
        if start_draining_vault and vault_active:
            # Drain from vault
            drain_amount = min(mint_amt, vault_total)
            results["vault_drained"][idx] = drain_amount
            vault_total -= drain_amount
            drained_total += drain_amount
            
            # The drained amount is effectively minted
            results["minted"][idx] += drain_amount
            minted_total += drain_amount
        else:
            # If we're still in minting phase or vault not active yet
            if vault_active:
                # Vault is active, redirect some funds
                redirected = (1-results["gamma"][idx]) * delta_M if delta_M > 0 else 0
                results["redirected"][idx] = redirected
                
                results["vault"][idx] = redirected
                results["minted"][idx] = delta_M - redirected
                
                vault_total += redirected
                mint_amt = delta_M - redirected
            else:
                # No vault redirection - all tokens are minted directly
                results["minted"][idx] = delta_M
                mint_amt = delta_M
            
            minted_total += mint_amt
        
        # Update cumulative
        results["cumulative_vault"][idx] = vault_total
        results["cumulative_minted"][idx] = minted_total
        results["cumulative_drained"][idx] = drained_total
    
    return results


# ======================================================
# run_monte_carlo_simulations (updated with new parameters)
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
    custom_rate_func: Callable[[float], float] = None,
    initial_poisson_value: int = 0,
    vault_start_week: int = 0
) -> Dict[str, Any]:
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
            custom_rate_func=custom_rate_func,
            initial_poisson_value=initial_poisson_value,
            vault_start_week=vault_start_week
        )
        all_simulations.append(result)
        
        if progress_callback:
            progress_callback((i + 1) / n_simulations)
    
    aggregated_results = {
        "time": np.arange(1, total_weeks + 1),
        "M": np.zeros(total_weeks),
        "median": {},
        "lower_ci": {},
        "upper_ci": {},
        "vault_start_week": vault_start_week
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

# Helper function to add vault start line to charts
def add_vault_start_line(chart, vault_start_data):
    if vault_start_data is not None:
        vault_line = alt.Chart(vault_start_data).mark_rule(
            strokeDash=[6, 3],
            color='black'
        ).encode(
            x='Round:Q',
            tooltip=['Label:N']
        )
        return alt.layer(chart, vault_line)
    return chart

# ======================================================
# Plotting function (updated to show vault start line)
# ======================================================
def plot_monte_carlo_results(power_results: Dict[str, Any], logistic_results: Dict[str, Any], exponential_results: Dict[str, Any], normalized_log_results: Dict[str, Any]) -> Dict[str, alt.Chart]:
    power_color = "green"
    logistic_color = "purple"
    exponential_color = "red"
    normalized_log_color = "orange"
    
    # Get vault start week (should be the same for all results)
    vault_start_week = power_results.get("vault_start_week", 0)
    
    # Create vertical line data if vault_start_week > 0
    vault_start_data = None
    if vault_start_week > 0:
        vault_start_data = pd.DataFrame({
            'Round': [vault_start_week],
            'Label': [f'Vault Starts (Round {vault_start_week})']
        })
    
    # ========== Chart 1: Emission Schedule ==========
    emission_data = pd.DataFrame({
        'Round': power_results["time"],
        'M(t)': power_results["M"]/1e6,
        'Power M*(t)': power_results["median"]["cumulative_minted"]/1e6,
        'Power_Lower_CI': power_results["lower_ci"]["cumulative_minted"]/1e6,
        'Power_Upper_CI': power_results["upper_ci"]["cumulative_minted"]/1e6,
        'Logistic M*(t)': logistic_results["median"]["cumulative_minted"]/1e6,
        'Logistic_Lower_CI': logistic_results["lower_ci"]["cumulative_minted"]/1e6,
        'Logistic_Upper_CI': logistic_results["upper_ci"]["cumulative_minted"]/1e6,
        'Exponential M*(t)': exponential_results["median"]["cumulative_minted"]/1e6,
        'Exponential_Lower_CI': exponential_results["lower_ci"]["cumulative_minted"]/1e6,
        'Exponential_Upper_CI': exponential_results["upper_ci"]["cumulative_minted"]/1e6,
        'NormalizedLog M*(t)': normalized_log_results["median"]["cumulative_minted"]/1e6,
        'Normalized_Log_Lower_CI': normalized_log_results["lower_ci"]["cumulative_minted"]/1e6,
        'Normalized_Log_Upper_CI': normalized_log_results["upper_ci"]["cumulative_minted"]/1e6
    })
    
    emission_base = alt.Chart(emission_data).encode(
        x=alt.X('Round:Q', title='Rounds')
    )
    
    emission_line_base = emission_base.mark_line(color='blue').encode(
        y=alt.Y('M(t):Q', title='Cumulative M-TIG Minted'),
        tooltip=['Round:Q', 'M(t):Q']
    )
    
    power_emission_line = emission_base.mark_line(color=power_color).encode(
        y='Power_M*(t):Q',
        tooltip=['Round:Q', 'Power_M*(t):Q']
    )
    power_emission_area = emission_base.mark_area(opacity=0.3, color=power_color).encode(
        y='Power_Lower_CI:Q',
        y2='Power_Upper_CI:Q'
    )
    
    logistic_emission_line = emission_base.mark_line(color=logistic_color).encode(
        y='Logistic_M*(t):Q',
        tooltip=['Round:Q', 'Logistic_M*(t):Q']
    )
    logistic_emission_area = emission_base.mark_area(opacity=0.3, color=logistic_color).encode(
        y='Logistic_Lower_CI:Q',
        y2='Logistic_Upper_CI:Q'
    )

    exponential_emission_line = emission_base.mark_line(color=exponential_color).encode(
        y='Exponential_M*(t):Q',
        tooltip=['Round:Q', 'Exponential_M*(t):Q']
    )
    exponential_emission_area = emission_base.mark_area(opacity=0.3, color=exponential_color).encode(
        y='Exponential_Lower_CI:Q',
        y2='Exponential_Upper_CI:Q'
    )

    normalized_log_emission_line = emission_base.mark_line(color=normalized_log_color).encode(
        y='Normalized_Log_M*(t):Q',
        tooltip=['Round:Q', 'Normalized_Log_M*(t):Q']
    )
    normalized_log_emission_area = emission_base.mark_area(opacity=0.3, color=normalized_log_color).encode(
        y='Normalized_Log_Lower_CI:Q',
        y2='Normalized_Log_Upper_CI:Q'
    )

    emission_melt = pd.melt(
        emission_data, 
        id_vars=['Round'], 
        value_vars=['M(t)', 'Power M*(t)', 'Logistic M*(t)', 'Exponential M*(t)', 'NormalizedLog M*(t)'],
        var_name='Series', 
        value_name='Value'
    )
    
    emission_legend = alt.Chart(emission_melt).mark_line().encode(
        x='Round:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['M(t)', 'Power M*(t)', 'Logistic M*(t)', 'Exponential M*(t)', 'NormalizedLog M*(t)'], 
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
    
    # Add vault start line to emission chart
    emission_chart = add_vault_start_line(emission_chart, vault_start_data)
    
    # ========== Chart 2: Event count (G) ==========
    event_data = pd.DataFrame({
        'Round': power_results["time"],
        'Median_G': power_results["median"]["G"],
        'Lower_G': power_results["lower_ci"]["G"],
        'Upper_G': power_results["upper_ci"]["G"]
    })
    
    event_base = alt.Chart(event_data).encode(
        x=alt.X('Round:Q', title='Rounds'),
        y=alt.Y('Median_G:Q', title='Total Challenges')
    )
    event_area = event_base.mark_area(opacity=0.3, color='red').encode(
        y=alt.Y('Lower_G:Q'),
        y2=alt.Y2('Upper_G:Q')
    )
    event_line = event_base.mark_line(color='red').encode(
        tooltip=['Round:Q', 'Median_G:Q']
    )
    
    event_chart = alt.layer(
        event_area, event_line
    ).properties(
        width=400, 
        height=300,
        title='Total Challenges'
    )
    
    # Add vault start line to event chart
    event_chart = add_vault_start_line(event_chart, vault_start_data)
    
    # ========== Chart 3: Gamma chart ==========
    gamma_data = pd.DataFrame({
        'Round': power_results["time"],
        'Power': power_results["median"]["gamma"],
        'Power_Lower_CI': power_results["lower_ci"]["gamma"],
        'Power_Upper_CI': power_results["upper_ci"]["gamma"],
        'Logistic': logistic_results["median"]["gamma"],
        'Logistic_Lower_CI': logistic_results["lower_ci"]["gamma"],
        'Logistic_Upper_CI': logistic_results["upper_ci"]["gamma"],
        'Exponential': exponential_results["median"]["gamma"],
        'Exponential_Lower_CI': exponential_results["lower_ci"]["gamma"],
        'Exponential_Upper_CI': exponential_results["upper_ci"]["gamma"],
        'Normalized_Log': normalized_log_results["median"]["gamma"],
        'Normalized_Log_Lower_CI': normalized_log_results["lower_ci"]["gamma"],
        'Normalized_Log_Upper_CI': normalized_log_results["upper_ci"]["gamma"]
    })
    
    gamma_base = alt.Chart(gamma_data).encode(
        x=alt.X('Round:Q', title='Rounds')
    )
    
    power_gamma_area = gamma_base.mark_area(opacity=0.3, color=power_color).encode(
        y=alt.Y('Power_Lower_CI:Q'),
        y2=alt.Y2('Power_Upper_CI:Q')
    )
    power_gamma_line = gamma_base.mark_line(color=power_color).encode(
        y=alt.Y('Power:Q', title='Goal Progress'),
        tooltip=['Round:Q', 'Power:Q']
    )
    
    logistic_gamma_area = gamma_base.mark_area(opacity=0.3, color=logistic_color).encode(
        y=alt.Y('Logistic_Lower_CI:Q'),
        y2=alt.Y2('Logistic_Upper_CI:Q')
    )
    logistic_gamma_line = gamma_base.mark_line(color=logistic_color).encode(
        y=alt.Y('Logistic:Q'),
        tooltip=['Round:Q', 'Logistic:Q']
    )

    exponential_gamma_line = gamma_base.mark_line(color=exponential_color).encode(
        y=alt.Y('Exponential:Q'),
        tooltip=['Round:Q', 'Exponential:Q']
    )
    exponential_gamma_area = gamma_base.mark_area(opacity=0.3, color=exponential_color).encode(
        y=alt.Y('Exponential_Lower_CI:Q'),
        y2=alt.Y2('Exponential_Upper_CI:Q')
    )

    normalized_log_gamma_line = gamma_base.mark_line(color=normalized_log_color).encode(
        y=alt.Y('Normalized_Log:Q'),
        tooltip=['Round:Q', 'Normalized_Log:Q']
    )
    normalized_log_gamma_area = gamma_base.mark_area(opacity=0.3, color=normalized_log_color).encode(
        y=alt.Y('Normalized_Log_Lower_CI:Q'),
        y2=alt.Y2('Normalized_Log_Upper_CI:Q')
    )
    
    gamma_melt = pd.melt(
        gamma_data, 
        id_vars=['Round'], 
        value_vars=['Power', 'Logistic', 'Exponential', 'Normalized_Log'],
        var_name='Series', 
        value_name='Value'
    )
    
    gamma_legend = alt.Chart(gamma_melt).mark_line().encode(
        x='Round:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['Power', 'Logistic', 'Exponential', 'Normalized_Log'], 
                                       range=[power_color, logistic_color, exponential_color, normalized_log_color]),
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
    
    # Add vault start line to gamma chart
    gamma_chart = add_vault_start_line(gamma_chart, vault_start_data)
    
    # ========== Chart 4: Cumulative tokens (vault) ==========
    token_data = pd.DataFrame({
        'Round': power_results["time"],
        'Power': power_results["median"]["cumulative_vault"]/1e6,
        'Power_Lower_Vault': power_results["lower_ci"]["cumulative_vault"]/1e6,
        'Power_Upper_Vault': power_results["upper_ci"]["cumulative_vault"]/1e6,
        'Power_Drained': power_results["median"]["cumulative_drained"]/1e6,
        'Power_Lower_Drained': power_results["lower_ci"]["cumulative_drained"]/1e6,
        'Power_Upper_Drained': power_results["upper_ci"]["cumulative_drained"]/1e6,
        'Logistic': logistic_results["median"]["cumulative_vault"]/1e6,
        'Logistic_Lower_Vault': logistic_results["lower_ci"]["cumulative_vault"]/1e6,
        'Logistic_Upper_Vault': logistic_results["upper_ci"]["cumulative_vault"]/1e6,
        'Logistic_Drained': logistic_results["median"]["cumulative_drained"]/1e6,
        'Logistic_Lower_Drained': logistic_results["lower_ci"]["cumulative_drained"]/1e6,
        'Logistic_Upper_Drained': logistic_results["upper_ci"]["cumulative_drained"]/1e6,
        'Exponential': exponential_results["median"]["cumulative_vault"]/1e6,
        'Exponential_Lower_Vault': exponential_results["lower_ci"]["cumulative_vault"]/1e6,
        'Exponential_Upper_Vault': exponential_results["upper_ci"]["cumulative_vault"]/1e6,
        'Exponential_Drained': exponential_results["median"]["cumulative_drained"]/1e6,
        'Exponential_Lower_Drained': exponential_results["lower_ci"]["cumulative_drained"]/1e6,
        'Exponential_Upper_Drained': exponential_results["upper_ci"]["cumulative_drained"]/1e6,
        'NormalizedLog': normalized_log_results["median"]["cumulative_vault"]/1e6,
        'Normalized_Log_Lower_Vault': normalized_log_results["lower_ci"]["cumulative_vault"]/1e6,
        'Normalized_Log_Upper_Vault': normalized_log_results["upper_ci"]["cumulative_vault"]/1e6,
        'Normalized_Log_Drained': normalized_log_results["median"]["cumulative_drained"]/1e6,
        'Normalized_Log_Lower_Drained': normalized_log_results["lower_ci"]["cumulative_drained"]/1e6,
        'Normalized_Log_Upper_Drained': normalized_log_results["upper_ci"]["cumulative_drained"]/1e6
    })
    
    token_base = alt.Chart(token_data).encode(
        x=alt.X('Round:Q', title='Rounds')
    )
    
    # Power Vault
    power_vault_area = token_base.mark_area(opacity=0.3, color=power_color).encode(
        y=alt.Y('Power_Lower_Vault:Q'),
        y2=alt.Y2('Power_Upper_Vault:Q')
    )
    power_vault_line = token_base.mark_line(color=power_color).encode(
        y=alt.Y('Power_Vault:Q', title='M-TIG')
    )
    # # Drained (Power)
    # power_drained_area = token_base.mark_area(opacity=0.3, color='orange', fillOpacity=0.1).encode(
    #     y=alt.Y('Power_Lower_Drained:Q'),
    #     y2=alt.Y2('Power_Upper_Drained:Q')
    # )
    # power_drained_line = token_base.mark_line(color='orange', strokeDash=[8, 4]).encode(
    #     y=alt.Y('Power_Drained:Q')
    # )
    
    # Logistic Vault
    logistic_vault_area = token_base.mark_area(opacity=0.3, color=logistic_color).encode(
        y=alt.Y('Logistic_Lower_Vault:Q'),
        y2=alt.Y2('Logistic_Upper_Vault:Q')
    )
    logistic_vault_line = token_base.mark_line(color=logistic_color).encode(
        y=alt.Y('Logistic_Vault:Q')
    )
    # # Drained (Logistic)
    # logistic_drained_area = token_base.mark_area(opacity=0.3, color='blue', fillOpacity=0.1).encode(
    #     y=alt.Y('Logistic_Lower_Drained:Q'),
    #     y2=alt.Y2('Logistic_Upper_Drained:Q')
    # )
    # logistic_drained_line = token_base.mark_line(color='blue', strokeDash=[8, 4]).encode(
    #     y=alt.Y('Logistic_Drained:Q')
    # )

    # Exponential Vault
    exponential_vault_area = token_base.mark_area(opacity=0.3, color=exponential_color).encode(
        y=alt.Y('Exponential_Lower_Vault:Q'),
        y2=alt.Y2('Exponential_Upper_Vault:Q')
    )
    exponential_vault_line = token_base.mark_line(color=exponential_color).encode(
        y=alt.Y('Exponential_Vault:Q')
    )
    # # Drained (Exponential)
    # exponential_drained_area = token_base.mark_area(opacity=0.3, color='green', fillOpacity=0.1).encode(
    #     y=alt.Y('Exponential_Lower_Drained:Q'),
    #     y2=alt.Y2('Exponential_Upper_Drained:Q')
    # )
    # exponential_drained_line = token_base.mark_line(color='green', strokeDash=[8, 4]).encode(
    #     y=alt.Y('Exponential_Drained:Q')
    # )

    # Normalized Log Vault
    normalized_log_vault_area = token_base.mark_area(opacity=0.3, color=normalized_log_color).encode(
        y=alt.Y('Normalized_Log_Lower_Vault:Q'),
        y2=alt.Y2('Normalized_Log_Upper_Vault:Q')
    )
    normalized_log_vault_line = token_base.mark_line(color=normalized_log_color).encode(
        y=alt.Y('Normalized_Log_Vault:Q')
    )
    # # Drained (Normalized Log)
    # normalized_log_drained_area = token_base.mark_area(opacity=0.3, color='purple', fillOpacity=0.1).encode(
    #     y=alt.Y('Normalized_Log_Lower_Drained:Q'),
    #     y2=alt.Y2('Normalized_Log_Upper_Drained:Q')
    # )
    # normalized_log_drained_line = token_base.mark_line(color='purple', strokeDash=[8, 4]).encode(
    #     y=alt.Y('Normalized_Log_Drained:Q')
    # )

    token_melt = pd.melt(
        token_data, 
        id_vars=['Round'], 
        # value_vars=['Power_Vault','Power_Drained','Logistic_Vault','Logistic_Drained', 'Exponential_Vault', 'Exponential_Drained', 'Normalized_Log_Vault', 'Normalized_Log_Drained'],
        value_vars=['Power','Logistic', 'Exponential', 'NormalizedLog'],
        var_name='Series', 
        value_name='Value'
    )
    
    token_legend = alt.Chart(token_melt).mark_line().encode(
        x='Round:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['Power','Logistic','Exponential', 'NormalizedLog'],
                                       range=[power_color, logistic_color, exponential_color, normalized_log_color]),
                       legend=alt.Legend(title='', orient='top')),
        # strokeDash=alt.condition(
        #     "indexof(datum.Series, 'Drained') >= 0",
        #     alt.value([8, 4]),
        #     alt.value([0])
        # )
    )
    
    token_chart = alt.layer(
        power_vault_area, power_vault_line,
        # power_drained_area, power_drained_line,
        logistic_vault_area, logistic_vault_line,
        # logistic_drained_area, logistic_drained_line,
        exponential_vault_area, exponential_vault_line,
        # exponential_drained_area, exponential_drained_line,
        normalized_log_vault_area, normalized_log_vault_line,
        # normalized_log_drained_area, normalized_log_drained_line,
        token_legend
    ).properties(
        width=400, 
        height=300,
        title='Vault Dynamics'
    )
    
    # Add vault start line to token chart
    token_chart = add_vault_start_line(token_chart, vault_start_data)
    
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
# Main Streamlit App (updated with new UI controls)
# ======================================================
def main():
    st.title("TIG Emission Monte Carlo Simulation")
    
    # ------------- Sidebar ------------------
    st.sidebar.header("Simulation Parameters")
    
    # 1) Choose homogeneous vs non-homogeneous
    process_type = st.sidebar.radio("Poisson Process Type:", ["Non-homogeneous", "Homogeneous"], index=0)
    
    total_weeks = st.sidebar.slider("Total Rounds", min_value=800, max_value=2000, value=1500, step=52)
    
    # NEW: Add initial value for Poisson process
    initial_poisson_value = st.sidebar.number_input("Number of Challenges @ Vault Policy Start", 
                                                   min_value=0, max_value=100, value=4, step=1)
    
    # NEW: Add vault start week
    vault_start_week = st.sidebar.number_input("Vault Start Round", 
                                              min_value=60, max_value=total_weeks, value=62, step=1)
    gamma_start_y = st.sidebar.number_input("Gamma Start", min_value=0.0, max_value=0.99, value=0.02, step=0.01)
    
    if process_type == "Homogeneous":
        mean_interarrival_time = st.sidebar.slider("Mean Interarrival Time (Rounds)", 
                                                   min_value=1.0, max_value=52.0, value=10.0, step=1.0)
        custom_rate_func = None
    else:
        st.sidebar.write("Define your rate function λ(t). For example:")
        st.sidebar.code("0.015384615+0.00142012*t", language="python")
        
        rate_str = st.sidebar.text_input("λ(t) =", value="0.015384615+0.00142012*t")
        
        def user_rate_func(t):
            return eval(rate_str, {"t": t, "np": np, "__builtins__": {}})
        
        custom_rate_func = user_rate_func
        mean_interarrival_time = 1.0  # dummy, not used
    
    # 2) Gamma Function parameters
    st.sidebar.subheader("Power Law Parameters")
    alpha = st.sidebar.slider("Alpha Parameter", min_value=0.01, max_value=0.99, value=0.75, step=0.01)
    
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
        
        gamma_start_n = initial_poisson_value
        gamma_start_y = gamma_start_y
        # ===== Run simulations for Power Law =====
        status_text.text("Running Monte Carlo simulations for Power Law...")
        power_law_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_function_power,
            gamma_func_params=(alpha,gamma_start_n, gamma_start_y),
            progress_callback=lambda p: update_progress(p * 0.25),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func,
            initial_poisson_value=initial_poisson_value,
            vault_start_week=vault_start_week
        )
        
        # ===== Run simulations for Logistic =====
        status_text.text("Running Monte Carlo simulations for Logistic...")
        logistic_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_logistic,
            gamma_func_params=(b, c, gamma_start_n, gamma_start_y),
            progress_callback=lambda p: update_progress(0.25 + p * 0.25),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func,
            initial_poisson_value=initial_poisson_value,
            vault_start_week=vault_start_week
        )

        # ===== Run simulations for Exponential =====
        status_text.text("Running Monte Carlo simulations for Exponential...")
        exponential_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_exponential,
            gamma_func_params=(b_exp, gamma_start_n, gamma_start_y),
            progress_callback=lambda p: update_progress(0.5 + p * 0.25),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func,
            initial_poisson_value=initial_poisson_value,
            vault_start_week=vault_start_week
        )

        # ===== Run simulations for Normalized Log =====
        status_text.text("Running Monte Carlo simulations for Normalized Log...")
        normalized_log_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_normalized_log,
            gamma_func_params=(a_log, Nref_log, gamma_start_n, gamma_start_y),
            progress_callback=lambda p: update_progress(0.75 + p * 0.25),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func,
            initial_poisson_value=initial_poisson_value,
            vault_start_week=vault_start_week
        )
        
        
        end_time = time.time()
        status_text.text(f"Simulations completed in {end_time - start_time:.2f} seconds")
        
        # ------------- Display results -------------
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
            
        # ------ Normalized Log metrics ------
        st.markdown("### Normalized Log")
        metrics_cols_normalized_log = st.columns(3)
        with metrics_cols_normalized_log[0]:
            st.metric("Median Total Minted (M-TIG)", 
                      f"{normalized_log_results['median']['cumulative_minted'][-1]/1e6:.2f}M")
        with metrics_cols_normalized_log[1]:
            st.metric("Median Vault Balance (M-TIG)", 
                      f"{normalized_log_results['median']['cumulative_vault'][-1]/1e6:.2f}M")
        with metrics_cols_normalized_log[2]:
            st.metric("Median Drained (M-TIG)", 
                      f"{normalized_log_results['median']['cumulative_drained'][-1]/1e6:.2f}M")

if __name__ == "__main__":
    main()