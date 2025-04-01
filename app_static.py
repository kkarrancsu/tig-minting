import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from typing import Dict, List, Callable, Any
import time

# Set Altair max rows limit to handle large datasets
st.set_page_config(layout="wide")
alt.data_transformers.disable_max_rows()

# Original functions from your code
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

def generate_poisson_process(
    total_weeks: int, 
    mean_interarrival_time: float, 
    random_seed: int = 1234
) -> List[int]:
    np.random.seed(random_seed)
    
    interarrival_times = np.random.exponential(
        scale=mean_interarrival_time, 
        size=total_weeks * 2
    )
    
    arrival_times = np.cumsum(interarrival_times)
    
    G = []
    for t in range(1, total_weeks + 1):
        count = np.sum(arrival_times <= t)
        G.append(int(count))
    
    return G

# Different gamma functions
def gamma_function_power(n: int, alpha: float) -> float:
    return (n ** alpha) / (1 + n ** alpha)

def gamma_logistic(n: int, b: float, c: float) -> float:
    return 1 / (1 + np.exp(-b * (n - c)))

def run_simulation(
    total_weeks: int,
    mean_interarrival_time: float,
    gamma_func,
    gamma_func_params: tuple,
    random_seed: int = 2345
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
        "cumulative_drained": np.zeros(total_weeks)
    }
    
    results["G"] = generate_poisson_process(total_weeks, mean_interarrival_time, random_seed)
    
    vault_total = 0
    minted_total = 0
    drained_total = 0
    start_draining_vault = False
    mint_amt = 0
    
    for t in range(1, total_weeks + 1):
        idx = t - 1
        
        results["M"][idx] = M(t)
        
        delta_M = results["M"][idx] - (M(t-1) if t > 1 else 0)
        
        results["gamma"][idx] = gamma_func(results["G"][idx], *gamma_func_params)
        
        if not start_draining_vault and delta_M == 0:
            start_draining_vault = True
            
        if start_draining_vault:
            drain_amount = min(mint_amt, vault_total)
            results["vault_drained"][idx] = drain_amount
            vault_total -= drain_amount
            drained_total += drain_amount
            
            results["minted"][idx] += drain_amount
            minted_total += drain_amount
        else:
            redirected = results["gamma"][idx] * delta_M if delta_M > 0 else 0
            results["redirected"][idx] = redirected
            
            results["vault"][idx] = redirected
            results["minted"][idx] = delta_M - redirected
            
            vault_total += redirected
            mint_amt = delta_M - redirected
            minted_total += mint_amt
        
        results["cumulative_vault"][idx] = vault_total
        results["cumulative_minted"][idx] = minted_total
        results["cumulative_drained"][idx] = drained_total
    
    return results

def run_monte_carlo_simulations(
    n_simulations: int,
    total_weeks: int,
    mean_interarrival_time: float,
    gamma_func: Callable,
    gamma_func_params: tuple,
    progress_callback: Callable[[float], None] = None,
    random_seed: int = 2345
) -> Dict[str, Any]:
    all_simulations = []
    
    for i in range(n_simulations):
        seed = random_seed + i
        result = run_simulation(total_weeks, mean_interarrival_time, gamma_func, gamma_func_params, seed)
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
    
    aggregated_results["M"] = all_simulations[0]["M"]
    
    keys_for_stats = ["G", "gamma", "vault", "minted", "redirected", "vault_drained",
                     "cumulative_vault", "cumulative_minted", "cumulative_drained"]
    
    for key in keys_for_stats:
        all_values = np.array([sim[key] for sim in all_simulations])
        
        aggregated_results["median"][key] = np.median(all_values, axis=0)
        aggregated_results["lower_ci"][key] = np.percentile(all_values, 5, axis=0)
        aggregated_results["upper_ci"][key] = np.percentile(all_values, 95, axis=0)
    
    return aggregated_results

def plot_monte_carlo_results(power_results: Dict[str, Any], logistic_results: Dict[str, Any]) -> Dict[str, alt.Chart]:
    # Define color scheme for consistent colors across plots
    power_color = "green"
    logistic_color = "purple"
    
    # Plot 1: Token emission
    emission_data = pd.DataFrame({
        'Week': power_results["time"],
        'M(t)': power_results["M"]/1e6,
        'Power_M*(t)': power_results["median"]["cumulative_minted"]/1e6,
        'Power_Lower_CI': power_results["lower_ci"]["cumulative_minted"]/1e6,
        'Power_Upper_CI': power_results["upper_ci"]["cumulative_minted"]/1e6,
        'Logistic_M*(t)': logistic_results["median"]["cumulative_minted"]/1e6,
        'Logistic_Lower_CI': logistic_results["lower_ci"]["cumulative_minted"]/1e6,
        'Logistic_Upper_CI': logistic_results["upper_ci"]["cumulative_minted"]/1e6
    })
    
    emission_base = alt.Chart(emission_data).encode(
        x=alt.X('Week:Q', title='Weeks')
    )
    
    # Base emission line (same for both models)
    emission_line_base = emission_base.mark_line(color='blue').encode(
        y=alt.Y('M(t):Q', title='Cumulative M-TIG Minted'),
        tooltip=['Week:Q', 'M(t):Q']
    )
    
    # Power Law emission
    power_emission_line = emission_base.mark_line(color=power_color).encode(
        y='Power_M*(t):Q',
        tooltip=['Week:Q', 'Power_M*(t):Q']
    )
    
    power_emission_area = emission_base.mark_area(opacity=0.3, color=power_color).encode(
        y='Power_Lower_CI:Q',
        y2='Power_Upper_CI:Q'
    )
    
    # Logistic emission
    logistic_emission_line = emission_base.mark_line(color=logistic_color).encode(
        y='Logistic_M*(t):Q',
        tooltip=['Week:Q', 'Logistic_M*(t):Q']
    )
    
    logistic_emission_area = emission_base.mark_area(opacity=0.3, color=logistic_color).encode(
        y='Logistic_Lower_CI:Q',
        y2='Logistic_Upper_CI:Q'
    )
    
    # Convert wide to long format for legend
    emission_melt = pd.melt(
        emission_data, 
        id_vars=['Week'], 
        value_vars=['M(t)', 'Power_M*(t)', 'Logistic_M*(t)'],
        var_name='Series', 
        value_name='Value'
    )
    
    # Create a chart with legend
    emission_legend = alt.Chart(emission_melt).mark_line().encode(
        x='Week:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['M(t)', 'Power_M*(t)', 'Logistic_M*(t)'], 
                                     range=['blue', power_color, logistic_color]),
                       legend=alt.Legend(title='', orient='top')),
    )
    
    emission_chart = alt.layer(
        power_emission_area, logistic_emission_area, 
        emission_line_base, power_emission_line, logistic_emission_line,
        emission_legend
    ).properties(
        width=400, 
        height=300,
        title='Token Emission Schedule'
    )
    
    # Plot 2: Event count chart - this is the same for both models since they use the same G
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
    
    # Plot 3: Gamma chart - comparing both gamma functions
    gamma_data = pd.DataFrame({
        'Week': power_results["time"],
        'Power_Median_Gamma': power_results["median"]["gamma"],
        'Power_Lower_Gamma': power_results["lower_ci"]["gamma"],
        'Power_Upper_Gamma': power_results["upper_ci"]["gamma"],
        'Logistic_Median_Gamma': logistic_results["median"]["gamma"],
        'Logistic_Lower_Gamma': logistic_results["lower_ci"]["gamma"],
        'Logistic_Upper_Gamma': logistic_results["upper_ci"]["gamma"]
    })
    
    gamma_base = alt.Chart(gamma_data).encode(
        x=alt.X('Week:Q', title='Weeks')
    )
    
    # Power Law gamma
    power_gamma_area = gamma_base.mark_area(opacity=0.3, color=power_color).encode(
        y=alt.Y('Power_Lower_Gamma:Q'),
        y2=alt.Y2('Power_Upper_Gamma:Q')
    )
    
    power_gamma_line = gamma_base.mark_line(color=power_color).encode(
        y=alt.Y('Power_Median_Gamma:Q', title='Goal Progress'),
        tooltip=['Week:Q', 'Power_Median_Gamma:Q']
    )
    
    # Logistic gamma
    logistic_gamma_area = gamma_base.mark_area(opacity=0.3, color=logistic_color).encode(
        y=alt.Y('Logistic_Lower_Gamma:Q'),
        y2=alt.Y2('Logistic_Upper_Gamma:Q')
    )
    
    logistic_gamma_line = gamma_base.mark_line(color=logistic_color).encode(
        y=alt.Y('Logistic_Median_Gamma:Q'),
        tooltip=['Week:Q', 'Logistic_Median_Gamma:Q']
    )
    
    # Convert wide to long format for legend
    gamma_melt = pd.melt(
        gamma_data, 
        id_vars=['Week'], 
        value_vars=['Power_Median_Gamma', 'Logistic_Median_Gamma'],
        var_name='Series', 
        value_name='Value'
    )
    
    gamma_legend = alt.Chart(gamma_melt).mark_line().encode(
        x='Week:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['Power_Median_Gamma', 'Logistic_Median_Gamma'], 
                                      range=[power_color, logistic_color]),
                       legend=alt.Legend(title='', orient='top')),
    )
    
    gamma_chart = alt.layer(
        power_gamma_area, power_gamma_line,
        logistic_gamma_area, logistic_gamma_line,
        gamma_legend
    ).properties(
        width=400, 
        height=300,
        title='Goal Progress'
    )
    
    # Plot 4: Cumulative tokens - vault dynamics
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
        'Logistic_Upper_Drained': logistic_results["upper_ci"]["cumulative_drained"]/1e6
    })
    
    token_base = alt.Chart(token_data).encode(
        x=alt.X('Week:Q', title='Weeks')
    )
    
    # Power Law Vault
    power_vault_area = token_base.mark_area(opacity=0.3, color=power_color).encode(
        y=alt.Y('Power_Lower_Vault:Q'),
        y2=alt.Y2('Power_Upper_Vault:Q')
    )
    
    power_vault_line = token_base.mark_line(color=power_color).encode(
        y=alt.Y('Power_Vault:Q', title='M-TIG')
    )
    
    # Power Law Drained
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
    
    # Logistic Drained
    logistic_drained_area = token_base.mark_area(opacity=0.3, color='blue', fillOpacity=0.1).encode(
        y=alt.Y('Logistic_Lower_Drained:Q'),
        y2=alt.Y2('Logistic_Upper_Drained:Q')
    )
    
    logistic_drained_line = token_base.mark_line(color='blue', strokeDash=[8, 4]).encode(
        y=alt.Y('Logistic_Drained:Q')
    )
    
    # Convert to long format for legend
    token_melt = pd.melt(
        token_data, 
        id_vars=['Week'], 
        value_vars=['Power_Vault', 'Power_Drained', 'Logistic_Vault', 'Logistic_Drained'],
        var_name='Series', 
        value_name='Value'
    )
    
    token_legend = alt.Chart(token_melt).mark_line().encode(
        x='Week:Q',
        y='Value:Q',
        color=alt.Color('Series:N', 
                       scale=alt.Scale(domain=['Power_Vault', 'Power_Drained', 'Logistic_Vault', 'Logistic_Drained'], 
                                      range=[power_color, 'orange', logistic_color, 'blue']),
                       legend=alt.Legend(title='', orient='top')),
        strokeDash=alt.condition(
            (alt.datum.Series == 'Power_Drained') | (alt.datum.Series == 'Logistic_Drained'),
            alt.value([8, 4]),  # More pronounced dotted line for drained
            alt.value([0])      # Solid line for vault
        )
    )
    
    token_chart = alt.layer(
        power_vault_area, power_vault_line, 
        power_drained_area, power_drained_line,
        logistic_vault_area, logistic_vault_line,
        logistic_drained_area, logistic_drained_line,
        token_legend
    ).properties(
        width=400, 
        height=300,
        title='Vault Dynamics'
    )
    
    # Create 2x2 grid
    top_row = alt.hconcat(emission_chart, token_chart, spacing=0)
    bottom_row = alt.hconcat(event_chart, gamma_chart, spacing=0)
    
    # Combine rows vertically
    final_chart = alt.vconcat(top_row, bottom_row, spacing=0)
    
    return {
        "emission_chart": emission_chart,
        "gamma_event_chart": event_chart,
        "token_chart": token_chart,
        "gamma_chart": gamma_chart,
        "final_chart": final_chart
    }

def main():
    st.title("Token Emission Monte Carlo Simulation")
    
    st.sidebar.header("Simulation Parameters")
    
    total_weeks = st.sidebar.slider("Total Weeks", min_value=800, max_value=2000, value=1500, step=52)
    
    mean_interarrival_time = st.sidebar.slider("Mean Interarrival Time (weeks)", 
                                             min_value=5.0, max_value=52.0, value=10.0, step=1.0)
    
    st.sidebar.subheader("Power Law Parameters")
    alpha = st.sidebar.slider("Alpha Parameter", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    
    st.sidebar.subheader("Logistic Parameters")
    b = st.sidebar.slider("b Parameter", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    c = st.sidebar.slider("c Parameter", min_value=10, max_value=200, value=50, step=5)
    
    gamma_options = {
        "Power Law": gamma_function_power,
        "Logistic": gamma_logistic,
    }
    
    n_simulations = st.sidebar.slider("Number of Simulations", min_value=10, max_value=1000, value=100, step=10)
    
    run_button = st.sidebar.button("Run Simulations")
    
    if run_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
        
        status_text.text("Running Monte Carlo simulations for Power Law...")
        start_time = time.time()
        
        # Run simulations for Power Law
        power_law_results = run_monte_carlo_simulations(
            n_simulations, 
            total_weeks, 
            mean_interarrival_time, 
            gamma_function_power,
            (alpha,),
            lambda p: update_progress(p * 0.5)  # First half of progress
        )
        
        status_text.text("Running Monte Carlo simulations for Logistic...")
        
        # Run simulations for Logistic
        logistic_results = run_monte_carlo_simulations(
            n_simulations, 
            total_weeks, 
            mean_interarrival_time, 
            gamma_logistic,
            (b, c),
            lambda p: update_progress(0.5 + p * 0.5)  # Second half of progress
        )
        
        end_time = time.time()
        status_text.text(f"Simulations completed in {end_time - start_time:.2f} seconds")
        
        st.subheader("Simulation Results")
        charts = plot_monte_carlo_results(power_law_results, logistic_results)
        
        # Create 2x2 grid layout
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
        
        # Power Law metrics
        st.markdown("### Power Law")
        metrics_cols_power = st.columns(3)
        
        with metrics_cols_power[0]:
            st.metric("Median Total Minted (M-TIG)", 
                    f"{power_law_results['median']['cumulative_minted'][-1]/1e6:.2f}M")
        
        with metrics_cols_power[1]:
            st.metric("Median Vault Balance (M-TIG)", 
                    f"{power_law_results['median']['cumulative_vault'][-1]/1e6:.2f}M")
        
        with metrics_cols_power[2]:
            st.metric("Median Drained from Vault (M-TIG)", 
                    f"{power_law_results['median']['cumulative_drained'][-1]/1e6:.2f}M")
        
        # Logistic metrics
        st.markdown("### Logistic")
        metrics_cols_logistic = st.columns(3)
        
        with metrics_cols_logistic[0]:
            st.metric("Median Total Minted (M-TIG)", 
                    f"{logistic_results['median']['cumulative_minted'][-1]/1e6:.2f}M")
        
        with metrics_cols_logistic[1]:
            st.metric("Median Vault Balance (M-TIG)", 
                    f"{logistic_results['median']['cumulative_vault'][-1]/1e6:.2f}M")
        
        with metrics_cols_logistic[2]:
            st.metric("Median Drained from Vault (M-TIG)", 
                    f"{logistic_results['median']['cumulative_drained'][-1]/1e6:.2f}M")

if __name__ == "__main__":
    main()