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
    return ((n-n_offset) ** alpha) / (1 + (n-n_offset) ** alpha) * (1-y_offset) + y_offset

def gamma_logistic(n: int, b: float, c: float, n_offset: int = 0, y_offset: float = 0) -> float:
    if n < n_offset:
        # linearly increase from 0 to y_offset
        return y_offset + (n-n_offset) * (y_offset - 0) / (n_offset - 0)
    return (1 / (1 + np.exp(-b * (n - c - n_offset)))) * (1-y_offset) + y_offset

# def gamma_exponential(n: int, b: float, n_offset: int = 0, y_offset: float = 0) -> float:
#     if n < n_offset:
#         # linearly increase from 0 to y_offset
#         return y_offset + (n-n_offset) * (y_offset - 0) / (n_offset - 0)
#     return ( 1-np.exp(-b * (n - n_offset)) ) * (1-y_offset) + y_offset
def gamma_exponential(n: int, a: float, b: float, c: float, n_offset: int = 0, y_offset: float = 0) -> float:
    return a*(( 1-b*np.exp(-c * (n - n_offset)) ) * (1-y_offset) + y_offset)


def gamma_normalized_log(n: int, a: float, N_ref: float, n_offset: int = 0, y_offset: float = 0) -> float:
    if n < n_offset:
        # linearly increase from 0 to y_offset
        return y_offset + (n-n_offset) * (y_offset - 0) / (n_offset - 0)
    return (np.log(1+a*(n-n_offset)) / np.log(1+a*N_ref)) * (1-y_offset) + y_offset

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
        "active_side": np.zeros(total_weeks),
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
    finish_by = 806
    
    for t in range(1, total_weeks + 1):
        idx = t - 1
        
        # M(t)
        results["M"][idx] = M(t)
        delta_M = results["M"][idx] - (M(t-1) if t > 1 else 0)
        
        # gamma
        g_val = results["G"][idx]
        results["gamma"][idx] = gamma_func(g_val, *gamma_func_params)
        
        # Start draining if M(t) stops increasing
        if not start_draining_vault and results["gamma"][idx] > 1:
            start_draining_vault = True
        
        # Check if vault is active (only when current week >= vault_start_week)
        vault_active = vault_start_week > 0 and t >= vault_start_week
        
        if start_draining_vault and vault_active:
            # Drain from vault
            # drain_amount = min(mint_amt/2, vault_total)  # update with new gamma function
            if t < finish_by:
                if (results["gamma"][idx]-1) * delta_M > vault_total/(finish_by-t):
                    active = 0
                else:
                    active = 1
                drain_amount = max((results["gamma"][idx]-1) * delta_M, vault_total/(finish_by-t))
            else:
                active = 2
                drain_amount = vault_total
            results["vault_drained"][idx] = drain_amount
            vault_total -= drain_amount
            drained_total += drain_amount
            
            # The drained amount is effectively minted
            amt_emitted = drain_amount + delta_M
            results["minted"][idx] += amt_emitted
            results["active_side"][idx] = active
            minted_total += amt_emitted
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
            results["active_side"][idx] = -1
        
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
    
    aggregated_results["active_side"] = np.median(np.array([sim["active_side"] for sim in all_simulations]), axis=0)
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
def plot_monte_carlo_results(exponential_results: Dict[str, Any], gamma_func_params: tuple) -> Dict[str, alt.Chart]:
    power_color = "green"
    logistic_color = "purple"
    exponential_color = "red"
    normalized_log_color = "orange"
    
    # Get vault start week (should be the same for all results)
    vault_start_week = exponential_results.get("vault_start_week", 0)
    
    # Create vertical line data if vault_start_week > 0
    vault_start_data = None
    if vault_start_week > 0:
        vault_start_data = pd.DataFrame({
            'Round': [vault_start_week],
            'Label': [f'Vault Starts (Round {vault_start_week})']
        })
    
    # Create a shared color scale for Chart 1 (Emission Schedule)
    emission_color_scale = alt.Scale(
        domain=['Original Schedule', 'Exponential'],
        range=['blue', 'green']
    )
    
    # Create a separate color scale for Charts 3 and 4 (only Exponential)
    exponential_color_scale = alt.Scale(
        domain=['Exponential'],
        range=['green']
    )
    
    # Create shared legend configurations
    legend_config = alt.Legend(
        title='Models',
        orient='top',
        direction='horizontal',
        titleAnchor='middle'
    )
    
    exponential_legend_config = alt.Legend(
        title='Model',
        orient='top',
        direction='horizontal',
        titleAnchor='middle'
    )
    
    # Create a shared selection for all charts
    selection = alt.selection_point(
        name='legend_selection',
        fields=['Model'],
        bind='legend',
    )
    
    # ========== Chart 1: Emission Schedule ==========
    time_array = exponential_results["time"]
    n_times = len(time_array)

    # Create base emission data with median values
    emission_data = pd.DataFrame({
        'Round': exponential_results["time"],
        'Original Schedule': exponential_results["M"]/1e6,
        'Exponential': exponential_results["median"]["cumulative_minted"]/1e6,
        'Exponential_Lower': exponential_results["lower_ci"]["cumulative_minted"]/1e6,
        'Exponential_Upper': exponential_results["upper_ci"]["cumulative_minted"]/1e6,
    })

    # Melt the data for the main lines
    emission_lines = pd.melt(
        emission_data,
        id_vars=['Round'],
        value_vars=['Original Schedule', 'Exponential'],
        var_name='Model',
        value_name='Value'
    )

    # Create a separate DataFrame for confidence intervals (excluding Original Schedule)
    emission_ci = pd.DataFrame({
        'Round': exponential_results["time"],
        'Model': ['Exponential'] * len(exponential_results["time"]),
        'Lower_CI': exponential_results["lower_ci"]["cumulative_minted"]/1e6,
        'Upper_CI': exponential_results["upper_ci"]["cumulative_minted"]/1e6
    })

    # Create the emission chart with both lines and confidence intervals
    emission_area = alt.Chart(emission_ci).mark_area(opacity=0.3).encode(
        x='Round:Q',
        y='Lower_CI:Q',
        y2='Upper_CI:Q',
        color=alt.Color('Model:N', scale=emission_color_scale),
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0))
    )

    emission_line = alt.Chart(emission_lines).mark_line().encode(
        x=alt.X('Round:Q', title='Rounds'),
        y=alt.Y('Value:Q', title='Cumulative M-TIG Distributed'),
        color=alt.Color('Model:N', scale=emission_color_scale, legend=legend_config),
        tooltip=['Round:Q', 'Value:Q', 'Model:N'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0))
    )

    emission_chart = alt.layer(emission_area, emission_line).add_params(
        selection
    ).properties(
        width=400,
        height=300,
        title='Token Emission Schedule'
    )
    
    # Add vault start line to emission chart
    emission_chart = add_vault_start_line(emission_chart, vault_start_data)
    
    # ========== Chart 2: Event count (G) ==========
    # Filter data to only include up to 500 rounds
    event_data = pd.DataFrame({
        'Round': exponential_results["time"],
        'Median_G': exponential_results["median"]["G"],
        'Lower_G': exponential_results["lower_ci"]["G"],
        'Upper_G': exponential_results["upper_ci"]["G"]
    })
    
    # Filter to only first 500 rounds
    event_data = event_data[event_data['Round'] <= 500]
    
    event_base = alt.Chart(event_data).encode(
        x=alt.X('Round:Q', title='Rounds', scale=alt.Scale(domain=[0, 500])),
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
    # First create the base data with all values
    gamma_data = pd.DataFrame({
        'Round': exponential_results["time"],
        'Exponential': exponential_results["median"]["gamma"],
        'Exponential_Lower': exponential_results["lower_ci"]["gamma"],
        'Exponential_Upper': exponential_results["upper_ci"]["gamma"],
    })

    # Melt the data for the main lines
    gamma_lines = pd.melt(
        gamma_data,
        id_vars=['Round'],
        value_vars=['Exponential'],
        var_name='Model',
        value_name='Value'
    )

    # Create a separate DataFrame for confidence intervals
    gamma_ci = pd.DataFrame({
        'Round': np.tile(exponential_results["time"], 1),
        'Model': np.repeat(['Exponential'], len(exponential_results["time"])),
        'Lower_CI': np.concatenate([
            exponential_results["lower_ci"]["gamma"],
        ]),
        'Upper_CI': np.concatenate([
            exponential_results["upper_ci"]["gamma"]
        ])
    })

    # Create the gamma chart with both lines and confidence intervals
    gamma_area = alt.Chart(gamma_ci).mark_area(opacity=0.3).encode(
        x='Round:Q',
        y='Lower_CI:Q',
        y2='Upper_CI:Q',
        color=alt.Color('Model:N', scale=exponential_color_scale),
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0))
    )

    gamma_line = alt.Chart(gamma_lines).mark_line().encode(
        x=alt.X('Round:Q', title='Rounds'),
        y=alt.Y('Value:Q', title='Goal Progress'),
        color=alt.Color('Model:N', scale=exponential_color_scale, legend=exponential_legend_config),
        tooltip=['Round:Q', 'Value:Q', 'Model:N'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0))
    )

    gamma_chart = alt.layer(gamma_area, gamma_line).add_params(
        selection
    ).properties(
        width=400,
        height=300,
        title='Goal Progress'
    )
    
    # Add vault start line to gamma chart
    gamma_chart = add_vault_start_line(gamma_chart, vault_start_data)
    
    # ========== Chart 4: Cumulative tokens (vault) ==========
    # Create the base data with all values
    token_data = pd.DataFrame({
        'Round': exponential_results["time"],
        'Exponential': exponential_results["median"]["cumulative_vault"]/1e6,
        'Exponential_Lower': exponential_results["lower_ci"]["cumulative_vault"]/1e6,
        'Exponential_Upper': exponential_results["upper_ci"]["cumulative_vault"]/1e6,
    })

    # Melt the data for the main lines
    token_lines = pd.melt(
        token_data,
        id_vars=['Round'],
        value_vars=['Exponential'],
        var_name='Model',
        value_name='Value'
    )

    # Create a separate DataFrame for confidence intervals
    token_ci = pd.DataFrame({
        'Round': np.tile(exponential_results["time"], 1),
        'Model': np.repeat(['Exponential'], len(exponential_results["time"])),
        'Lower_CI': np.concatenate([
            exponential_results["lower_ci"]["cumulative_vault"]/1e6,
        ]),
        'Upper_CI': np.concatenate([
            exponential_results["upper_ci"]["cumulative_vault"]/1e6,
        ])
    })

    # Create the token chart with both lines and confidence intervals
    token_area = alt.Chart(token_ci).mark_area(opacity=0.3).encode(
        x='Round:Q',
        y='Lower_CI:Q',
        y2='Upper_CI:Q',
        color=alt.Color('Model:N', scale=exponential_color_scale),
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0))
    )

    token_line = alt.Chart(token_lines).mark_line().encode(
        x=alt.X('Round:Q', title='Rounds'),
        y=alt.Y('Value:Q', title='M-TIG'),
        color=alt.Color('Model:N', scale=exponential_color_scale, legend=exponential_legend_config),
        tooltip=['Round:Q', 'Value:Q', 'Model:N'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0))
    )

    token_chart = alt.layer(token_area, token_line).add_params(
        selection
    ).properties(
        width=400,
        height=300,
        title='Vault Dynamics'
    )
    
    # Add vault start line to token chart
    token_chart = add_vault_start_line(token_chart, vault_start_data)
    
    # ========== Chart 5: Gamma Function Plot ==========
    # Create data for gamma function plot
    n_values = np.arange(0, 250)  # Show up to 500 challenges
    gamma_values = []
    for nn in n_values:
        gamma_values.append(gamma_exponential(nn, *gamma_func_params))
    
    gamma_func_data = pd.DataFrame({
        'n': n_values,
        'Gamma': gamma_values
    })
    
    gamma_func_chart = alt.Chart(gamma_func_data).mark_line().encode(
        x=alt.X('n:Q', title='Number of Challenges (n)'),
        y=alt.Y('Gamma:Q', title='Gamma Value'),
        tooltip=['n:Q', 'Gamma:Q']
    ).properties(
        width=800,  # Full width to match other plots
        height=300,
        title='Gamma Function'
    )
    
    # Layout: 3 rows of charts
    top_row = alt.hconcat(emission_chart, token_chart, spacing=20)
    middle_row = alt.hconcat(event_chart, gamma_chart, spacing=20)
    bottom_row = gamma_func_chart
    final_chart = alt.vconcat(top_row, middle_row, bottom_row, spacing=20).configure_view(
        strokeWidth=0
    )
    
    return {
        "emission_chart": emission_chart,
        "gamma_event_chart": event_chart,
        "token_chart": token_chart,
        "gamma_chart": gamma_chart,
        "gamma_func_chart": gamma_func_chart,
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
    
    st.sidebar.subheader("Exponential Parameters")
    b_exp = st.sidebar.number_input("b-Exponential", 
                                  min_value=0.0001, 
                                  max_value=1.0, 
                                  value=0.10, 
                                  step=0.0001, 
                                  format="%.4f")

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
        
        # ===== Run simulations for Exponential =====
        status_text.text("Running Monte Carlo simulations for Exponential...")
        exponential_results = run_monte_carlo_simulations(
            n_simulations=n_simulations,
            total_weeks=total_weeks,
            mean_interarrival_time=mean_interarrival_time,
            gamma_func=gamma_exponential,
            gamma_func_params=(b_exp, 0, 0),
            progress_callback=lambda p: update_progress(0.5 + p * 0.25),
            use_nonhomogeneous=(process_type == "Non-homogeneous"),
            custom_rate_func=custom_rate_func,
            initial_poisson_value=initial_poisson_value,
            vault_start_week=vault_start_week
        )
        
        end_time = time.time()
        status_text.text(f"Simulations completed in {end_time - start_time:.2f} seconds")
        
        # ------------- Display results -------------
        charts = plot_monte_carlo_results(exponential_results, (b_exp, gamma_start_n, gamma_start_y))
        
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
        
        col5, col6 = st.columns(2)
        with col5:
            st.altair_chart(charts["gamma_func_chart"], use_container_width=True)
        # with col6:
        #     st.altair_chart(charts["final_chart"], use_container_width=True)

        st.subheader("Key Metrics")
        
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