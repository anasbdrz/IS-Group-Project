# streamlit run s:/Uni/IS/worldcup/app.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats # CRITICAL FOR WHOLE NUMBERS
import graphviz

# 1. Page Configuration
st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")

# --- DATA PROCESSING SECTION ---
@st.cache_data
def load_and_prep_data(filepath, hist_rank_filepath, current_rank_filepath):
    print("\n" + "="*40)
    print("STAGE 1: INITIALIZING DATA PIPELINE")
    
    # 1. Load all three datasets
    df = pd.read_csv(filepath)
    hist_rankings = pd.read_csv(hist_rank_filepath)
    current_rankings = pd.read_csv(current_rank_filepath)
    
    # Force nanosecond precision [ns] so the merge_asof function doesn't crash
    df['date'] = pd.to_datetime(df['date']).astype('datetime64[ns]')
    
    # --- 🚨 STANDARDIZE THE HISTORICAL RANKINGS 🚨 ---
    # Convert dates and rename columns to match our pipeline logic
    hist_rankings['date'] = pd.to_datetime(hist_rankings['date']).astype('datetime64[ns]')
    clean_hist_rankings = hist_rankings[['date', 'team', 'rank']].dropna()
    clean_hist_rankings.rename(columns={'date': 'rank_date', 'team': 'country_full'}, inplace=True)
    clean_hist_rankings = clean_hist_rankings.sort_values('rank_date')
    
    # --- 🚨 THE NAME FIX DICTIONARY 🚨 ---
    name_mappings = {
        "USA": "United States",
        "IR Iran": "Iran",                
        "South Korea": "Korea Republic",  
        "Cape Verde": "Cabo Verde"        
    }
    df['home_team'] = df['home_team'].replace(name_mappings)
    df['away_team'] = df['away_team'].replace(name_mappings)
    
    # Apply name fixes to both ranking datasets
    clean_hist_rankings['country_full'] = clean_hist_rankings['country_full'].replace(name_mappings)
    current_rankings['team'] = current_rankings['team'].replace(name_mappings)
    
    # 2. Split the dataset
    schedule_2026 = df.tail(72).copy()
    historical_df = df.iloc[:-72].copy()
    recent_df = historical_df[historical_df['date'] >= '2018-08-01'].copy()
    recent_df = recent_df.sort_values('date')
    
    # 3. Time-Travel Merge: Attach Historical Ranks
    recent_df = pd.merge_asof(
        recent_df, clean_hist_rankings,
        left_on='date', right_on='rank_date',
        left_by='home_team', right_by='country_full',
        direction='backward' 
    )
    recent_df.rename(columns={'rank': 'home_rank'}, inplace=True)

    recent_df = pd.merge_asof(
        recent_df, clean_hist_rankings,
        left_on='date', right_on='rank_date',
        left_by='away_team', right_by='country_full',
        direction='backward'
    )
    recent_df.rename(columns={'rank': 'away_rank'}, inplace=True)

    recent_df['home_rank'] = recent_df['home_rank'].fillna(100)
    recent_df['away_rank'] = recent_df['away_rank'].fillna(100)

    # 4. Melt the Data for Statsmodels
    home = recent_df[['date', 'home_team', 'away_team', 'home_score', 'tournament', 'neutral', 'home_rank', 'away_rank']].copy()
    home.columns = ['date', 'team', 'opponent', 'goals', 'tournament', 'neutral', 'team_rank', 'opponent_rank']
    home['is_home'] = ~home['neutral']
    home['rank_diff'] = home['team_rank'] - home['opponent_rank']

    away = recent_df[['date', 'away_team', 'home_team', 'away_score', 'tournament', 'neutral', 'away_rank', 'home_rank']].copy()
    away.columns = ['date', 'team', 'opponent', 'goals', 'tournament', 'neutral', 'team_rank', 'opponent_rank']
    away['is_home'] = False
    away['rank_diff'] = away['team_rank'] - away['opponent_rank']

    training_data = pd.concat([home, away], ignore_index=True).dropna(subset=['goals'])

    # 5. Sparsity Filter
    team_counts = training_data['team'].value_counts()
    established_teams = team_counts[team_counts >= 5].index
    training_data = training_data[
        training_data['team'].isin(established_teams) & 
        training_data['opponent'].isin(established_teams)
    ]

    print(f"-> Pipeline Complete! Loaded {len(training_data)} rows for training.")
    print("="*40 + "\n")

    # 6. Build the 2026 Dictionary
    current_rank_dict = dict(zip(current_rankings['team'], current_rankings['rank']))

    return training_data, schedule_2026, current_rank_dict

# Load both datasets into memory (Done ONLY ONCE)
train_data, future_schedule, current_rank_dict = load_and_prep_data(
    r'S:\Uni\IS\worldcup\results.csv', 
    r'S:\Uni\IS\worldcup\fifa_mens_rank.csv', 
    r'S:\Uni\IS\worldcup\fifa_2026_rank.csv' # Change this to whatever your 2026 file is named!
)

# --- THE AI ENGINE ---
@st.cache_resource
def train_poisson_model(df):
    print("STAGE 2: TRAINING POISSON MODEL")
    print("-> Fitting Generalized Linear Model (GLM) with FIFA Rankings...")
    
    # 🚨 THE MAGIC INGREDIENT: We added + rank_diff
    formula = "goals ~ team + opponent + is_home + rank_diff"
    
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
    
    print("-> Training Complete! AI Brain is active.")
    print("="*40 + "\n")
    
    return model

# Train the model 
poisson_model = train_poisson_model(train_data)
st.success("Model trained successfully with FIFA Rankings!")

# 2. Main Header (Sidebar removed!)
st.title("🏆 World Cup 2026 Predictor")
st.markdown("Predicting the 2026 expanded 48-team tournament using Poisson Regression.")

# 3. Create the Tabs
tab1, tab_3rd, tab_r32, tab_r16, tab_qf, tab_sf, tab_3rd_match, tab_final = st.tabs([
    "📊 Groups", "🥉 3rd Place", "🔥 R32", "⚔️ R16", 
    "🛡️ Quarters", "🏆 Semis", "🥉 Bronze", "👑 Final"
])

# --- PANDAS HIGHLIGHTING FUNCTIONS ---
def highlight_group(row):
    if row.name < 2:    # Index 0 and 1 (Top 2 teams)
        return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row) # Light Green
    elif row.name == 3: # Index 3 (Last place)
        return ['background-color: rgba(220, 53, 69, 0.2)'] * len(row) # Light Red
    return [''] * len(row) # 3rd place gets no color in the main group

def highlight_third_place(row):
    if row.name < 8:    # Top 8 advance
        return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row) 
    return ['background-color: rgba(220, 53, 69, 0.2)'] * len(row)     # Bottom 4 eliminated

# Initialize session state variables for knockouts if they don't exist yet
if 'ro16_qualifiers' not in st.session_state:
    st.session_state.ro16_qualifiers = []
if 'qf_qualifiers' not in st.session_state:
    st.session_state.qf_qualifiers = []

# --- TAB 1: GROUP STAGE ---
with tab1:
    st.header("World Cup 2026: Group Stage Simulation")
    
    if st.button("Simulate Group Stage Scores"):
        print("STAGE 3: RUNNING SINGLE WORLD CUP SIMULATION")
        predictions_list = []
        
        for count, (index, row) in enumerate(future_schedule.iterrows()):
            h_team = row['home_team']
            a_team = row['away_team']
            is_neutral = row['neutral']
            
            # --- 🚨 FETCH CURRENT FIFA RANKS 🚨 ---
            # If a team isn't found in the ranking dictionary, default them to rank 100
            h_rank = current_rank_dict.get(h_team, 100)
            a_rank = current_rank_dict.get(a_team, 100)
            
            # Set up the data EXACTLY how the new formula expects it
            h_data = pd.DataFrame({
                'team': [h_team], 
                'opponent': [a_team], 
                'is_home': [not is_neutral],
                'rank_diff': [h_rank - a_rank] # Home perspective
            })
            
            a_data = pd.DataFrame({
                'team': [a_team], 
                'opponent': [h_team], 
                'is_home': [False],
                'rank_diff': [a_rank - h_rank] # Away perspective
            })
            
            try:
                # 1. AI Predicts Raw Expected Goals
                raw_h_xg = poisson_model.predict(h_data).values[0]
                raw_a_xg = poisson_model.predict(a_data).values[0]
                
                # --- 🧠 REALISTIC FOOTBALL LOGIC ENGINE 🧠 ---
                
                # Rule 1: Tournament Deflation (World Cup games are tighter than qualifiers)
                h_xg = raw_h_xg * 0.85
                a_xg = raw_a_xg * 0.85
                
                # Rule 2: The "Cancel Out" Cap
                # If total expected goals are too high, they are likely fighting for possession. 
                # We scale them down proportionally.
                total_xg = h_xg + a_xg
                max_match_xg = 4.0 # A match almost never averages more than 4 expected goals
                
                if total_xg > max_match_xg:
                    scale_factor = max_match_xg / total_xg
                    h_xg = h_xg * scale_factor
                    a_xg = a_xg * scale_factor
                
                # Rule 3: The Hard Floor (No team has literally 0 chance to score)
                h_xg = max(h_xg, 0.1)
                a_xg = max(a_xg, 0.1)
                
                # 2. Simulate Whole Numbers using SciPy
                sim_h_score = stats.poisson.rvs(mu=h_xg)
                sim_a_score = stats.poisson.rvs(mu=a_xg)
                
                # Rule 4: The Outlier Pruner 
                if sim_h_score > 5: sim_h_score = 5
                if sim_a_score > 5: sim_a_score = 5
                
                # --- 🚨 NEW: SEPARATE MATH FROM DISPLAY 🚨 ---
                predictions_list.append({
                    'Home': h_team,
                    'Home_Goals': sim_h_score, # Raw integer for the Standings Calculator
                    'Away_Goals': sim_a_score, # Raw integer for the Standings Calculator
                    'Result': f"({h_xg:.2f}) {sim_h_score} - {sim_a_score} ({a_xg:.2f})",
                    'Away': a_team
                })
            except Exception as e:
                print(f"CRASH {h_team} vs {a_team}: {e}")
                
        results_df = pd.DataFrame(predictions_list)

        # --- SAFETY NET CHECK ---
        if len(results_df) == 0:
            st.error("🚨 Simulation failed! All matches crashed. Check your Anaconda Prompt terminal for the 'CRASH' messages to see which team names are mismatched.")
        else:
            # --- GRAPH THEORY: EXTRACTING THE GROUPS ---
            match_network = {}
            for _, row in future_schedule.iterrows():
                h, a = row['home_team'], row['away_team']
                if h not in match_network: match_network[h] = set()
                if a not in match_network: match_network[a] = set()
                match_network[h].add(a)
                match_network[a].add(h)
                
            visited = set()
            inferred_groups = []
            for team in match_network:
                if team not in visited:
                    group = set()
                    queue = [team]
                    while queue:
                        curr = queue.pop(0)
                        if curr not in visited:
                            visited.add(curr)
                            group.add(curr)
                            queue.extend(list(match_network[curr]))
                    inferred_groups.append(sorted(list(group)))
            
            inferred_groups.sort(key=lambda x: x[0])
            
            # --- 3. RENDERING THE UI & CALCULATING STANDINGS ---
            st.success("Simulation Complete! Scroll down to see the official standings.")
            group_letters = "ABCDEFGHIJKL"
            
            # 🚨 NEW: Create a list to catch the 3rd place teams!
            # 🚨 NEW: Dictionaries to catch teams for the Knockouts
            group_winners = {}
            group_runners_up = {}
            all_third_place = []
            
            for i, group_teams in enumerate(inferred_groups):
                st.markdown(f"### Group {group_letters[i]}")
                
                group_matches = results_df[
                    results_df['Home'].isin(group_teams) & 
                    results_df['Away'].isin(group_teams)
                ]
                
                standings = {team: {'Played': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0} for team in group_teams}
                
                for _, row in group_matches.iterrows():
                    home = row['Home']
                    away = row['Away']
                    h_goals = row['Home_Goals']
                    a_goals = row['Away_Goals']
                    
                    standings[home]['Played'] += 1
                    standings[away]['Played'] += 1
                    standings[home]['GF'] += h_goals
                    standings[away]['GF'] += a_goals
                    standings[home]['GA'] += a_goals
                    standings[away]['GA'] += h_goals
                    
                    if h_goals > a_goals:
                        standings[home]['W'] += 1
                        standings[home]['Pts'] += 3
                        standings[away]['L'] += 1
                    elif h_goals < a_goals:
                        standings[away]['W'] += 1
                        standings[away]['Pts'] += 3
                        standings[home]['L'] += 1
                    else:
                        standings[home]['D'] += 1
                        standings[away]['D'] += 1
                        standings[home]['Pts'] += 1
                        standings[away]['Pts'] += 1
                
                standings_df = pd.DataFrame.from_dict(standings, orient='index')
                standings_df['GD'] = standings_df['GF'] - standings_df['GA']
                standings_df = standings_df.sort_values(by=['Pts', 'GD', 'GF'], ascending=[False, False, False]).reset_index()
                standings_df.rename(columns={'index': 'Team'}, inplace=True)
                
                # --- 🚨 NEW: CATCH THE 3RD PLACE TEAM ---
                # Index 2 is the 3rd place team. We copy them and tag which Group they came from!
                third_place_team = standings_df.iloc[2].copy()
                third_place_team['Group'] = group_letters[i] 
                all_third_place.append(third_place_team)
                
                # Save the 1st and 2nd place teams using the group letter as the key!
                group_winners[group_letters[i]] = standings_df.iloc[0]['Team']
                group_runners_up[group_letters[i]] = standings_df.iloc[1]['Team']
                # ----------------------------------------
                
                # Render the 2-Column UI with Color Coding
                col1, col2 = st.columns([1.5, 1]) 
                
                with col1:
                    st.caption("Group Standings")
                    # Apply the highlight function we wrote earlier!
                    styled_standings = standings_df.style.apply(highlight_group, axis=1)
                    st.dataframe(styled_standings, hide_index=True, width='stretch')
                
                with col2:
                    st.caption("Match Results (Expected Goals)")
                    st.dataframe(group_matches[['Home', 'Result', 'Away']], hide_index=True, width='stretch')
                
                st.divider() 

# --- TAB 1.5: THIRD PLACE STANDINGS ---
with tab_3rd:
    st.header("🥉 Third-Place Team Rankings")
    st.markdown("The **top 8** teams from this table will qualify for the Round of 32.")
    
    # We only want to build this table if the simulation has actually run!
    if 'all_third_place' in locals() and len(all_third_place) > 0:
        
        # 1. Build the master dataframe from our caught teams
        third_place_df = pd.DataFrame(all_third_place)
        
        # 2. Reorder columns so 'Group' is right next to 'Team'
        cols = ['Team', 'Group', 'Played', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
        third_place_df = third_place_df[cols]
        
        # 3. Sort by FIFA tie-breaker rules (Points -> Goal Difference -> Goals For)
        third_place_df = third_place_df.sort_values(by=['Pts', 'GD', 'GF'], ascending=[False, False, False]).reset_index(drop=True)
        
        # 4. Apply the styling (Top 8 Green, Bottom 4 Red)
        styled_third_place = third_place_df.style.apply(highlight_third_place, axis=1)
        
        # 5. Display the beautiful table!
        st.dataframe(styled_third_place, hide_index=True, width='stretch', height=460)
        
    else:
        st.info("👈 Run the Group Stage simulation first to generate the Third-Place rankings!")

# --- THE KNOCKOUT ENGINE ---
def simulate_knockout(home_team, away_team, poisson_model, current_rank_dict):
    
    # 1. Look up the ranks using the raw, clean team names
    h_rank = current_rank_dict.get(home_team, 100)
    a_rank = current_rank_dict.get(away_team, 100)
    
    # 2. Build the DataFrames for the AI
    h_data = pd.DataFrame({'team': [home_team], 'opponent': [away_team], 'is_home': [True], 'rank_diff': [h_rank - a_rank]})
    a_data = pd.DataFrame({'team': [away_team], 'opponent': [home_team], 'is_home': [False], 'rank_diff': [a_rank - h_rank]})
    
    # 3. Predict Expected Goals (with Tournament Deflation)
    h_xg = max(poisson_model.predict(h_data).values[0] * 0.85, 0.1)
    a_xg = max(poisson_model.predict(a_data).values[0] * 0.85, 0.1)
    
    # 4. Roll the Dice
    h_score = stats.poisson.rvs(mu=h_xg)
    a_score = stats.poisson.rvs(mu=a_xg)
    
    # 5. Outlier Pruner
    if h_score > 5: h_score = 5
    if a_score > 5: a_score = 5
    
    # 6. Determine Winner & Format Result String
    result_str = f"({h_xg:.2f}) {h_score} - {a_score} ({a_xg:.2f})"
    winner = home_team if h_score > a_score else away_team
    
    # --- 🚨 PENALTY SHOOTOUT LOGIC 🚨 ---
    if h_score == a_score:
        # Higher ranked team has a 60% chance to win the shootout
        h_win_prob = 0.6 if h_rank < a_rank else 0.4
        import random
        if random.random() < h_win_prob:
            winner = home_team
            result_str = f"{h_score} - {a_score} ({winner} wins on Pens)"
        else:
            winner = away_team
            result_str = f"{h_score} - {a_score} ({winner} wins on Pens)"
            
    return winner, result_str

# --- 🚨 3RD PLACE ALLOCATION ALGORITHM 🚨 ---
def get_3rd_place(available_teams, preferred_groups):
    # Finds the highest-ranked 3rd place team from the preferred groups
    for pref in preferred_groups:
        for i, team in enumerate(available_teams):
            if team['Group'] == pref:
                return available_teams.pop(i)['Team']
    # Fallback: If no team from preferred groups exists, take the best remaining
    return available_teams.pop(0)['Team']

# --- TAB 2: ROUND OF 32 ---
with tab_r32:
    st.header("🔥 Round of 32")
    
    if 'third_place_df' in locals() or ('all_third_place' in locals() and len(all_third_place) > 0): 
        available_3rds = third_place_df.head(8).to_dict('records')
        
        ro32_fixtures = [
            ("Match 73", group_runners_up['A'], group_runners_up['B']),
            ("Match 74", group_winners['E'], get_3rd_place(available_3rds, ['A','B','C','D','F'])),
            ("Match 75", group_winners['F'], group_runners_up['C']),
            ("Match 76", group_winners['C'], group_runners_up['F']),
            ("Match 77", group_winners['I'], get_3rd_place(available_3rds, ['C','D','F','G','H'])),
            ("Match 78", group_runners_up['E'], group_runners_up['I']),
            ("Match 79", group_winners['A'], get_3rd_place(available_3rds, ['C','E','F','H','I'])),
            ("Match 80", group_winners['L'], get_3rd_place(available_3rds, ['E','H','I','J','K'])),
            ("Match 81", group_winners['D'], get_3rd_place(available_3rds, ['B','E','F','I','J'])),
            ("Match 82", group_winners['G'], get_3rd_place(available_3rds, ['A','E','H','I','J'])),
            ("Match 83", group_runners_up['K'], group_runners_up['L']),
            ("Match 84", group_winners['H'], group_runners_up['J']),
            ("Match 85", group_winners['B'], get_3rd_place(available_3rds, ['E','F','G','I','J'])),
            ("Match 86", group_winners['J'], group_runners_up['H']),
            ("Match 87", group_winners['K'], get_3rd_place(available_3rds, ['D','E','I','J','L'])),
            ("Match 88", group_runners_up['D'], group_runners_up['G']),
        ]
        
        ro32_results = []
        local_ro16_list = [] 
        
        for match_num, home, away in ro32_fixtures:
            winner, score = simulate_knockout(home, away, poisson_model, current_rank_dict)
            ro32_results.append({"Match": match_num, "Team 1": home, "Result": score, "Team 2": away, "Advancing": winner})
            local_ro16_list.append(winner)
            
        st.session_state.ro16_qualifiers = local_ro16_list
        df_r32 = pd.DataFrame(ro32_results)
        
        # 🚨 SAVE THE DATAFRAME GLOBALLY SO THE R16 TAB CAN DRAW IT LATER 🚨
        st.session_state.df_r32 = df_r32 
        
        # --- TOP ROW: The Data Table ---
        st.subheader("📋 Match Data")
        st.dataframe(df_r32, hide_index=True, width='stretch')
        st.divider() 
        
        # --- BOTTOM ROW: The Visual Tree (Split Layout) ---
        st.subheader("🌳 Advancing Tree")
        
        col_tree_left, col_tree_right = st.columns(2)
        import graphviz
        
        # --- LEFT SIDE (Matches 1-8) ---
        with col_tree_left:
            dot_L = graphviz.Digraph()
            # 🚨 ADDED outputorder='edgesfirst' 🚨
            dot_L.attr(rankdir='LR', bgcolor='transparent', splines='ortho', ranksep='0.4', nodesep='0.05', outputorder='edgesfirst')
            
            dot_L.node_attr.update(shape='box', style='rounded,filled', fillcolor='#262730', 
                                   color='#4f8bf9', fontcolor='white', fontsize='11', 
                                   fixedsize='true', width='1.5', height='0.4')
            dot_L.edge_attr.update(color='white', penwidth='1.5', dir='none')
            
            for index, row in df_r32.iloc[:8].iterrows():
                id_t1, id_t2, id_winner = f"R32_{row['Team 1']}", f"R32_{row['Team 2']}", f"R16_{row['Advancing']}"
                
                dot_L.node(id_t1, row['Team 1'])
                dot_L.node(id_t2, row['Team 2'])
                dot_L.node(id_winner, row['Advancing'], fillcolor='#17b169') 
                
                dot_L.edge(id_t1, id_winner, tailport='e', headport='w')
                dot_L.edge(id_t2, id_winner, tailport='e', headport='w')
            st.graphviz_chart(dot_L, use_container_width=True)

        # --- RIGHT SIDE (Matches 9-16) ---
        with col_tree_right:
            dot_R = graphviz.Digraph()
            # 🚨 ADDED outputorder='edgesfirst' 🚨
            dot_R.attr(rankdir='RL', bgcolor='transparent', splines='ortho', ranksep='0.4', nodesep='0.05', outputorder='edgesfirst')
            
            dot_R.node_attr.update(shape='box', style='rounded,filled', fillcolor='#262730', 
                                   color='#4f8bf9', fontcolor='white', fontsize='11', 
                                   fixedsize='true', width='1.5', height='0.4')
            dot_R.edge_attr.update(color='white', penwidth='1.5', dir='none')
            
            for index, row in df_r32.iloc[8:].iterrows():
                id_t1, id_t2, id_winner = f"R32_{row['Team 1']}", f"R32_{row['Team 2']}", f"R16_{row['Advancing']}"
                
                dot_R.node(id_t1, row['Team 1'])
                dot_R.node(id_t2, row['Team 2'])
                dot_R.node(id_winner, row['Advancing'], fillcolor='#17b169') 
                
                dot_R.edge(id_t1, id_winner, tailport='w', headport='e')
                dot_R.edge(id_t2, id_winner, tailport='w', headport='e')
            st.graphviz_chart(dot_R, use_container_width=True)

# --- TAB 3: ROUND OF 16 ---
with tab_r16:
    st.header("⚔️ Round of 16")
    
    if 'ro16_qualifiers' in st.session_state and len(st.session_state.ro16_qualifiers) >= 16:
        q = st.session_state.ro16_qualifiers[:16]
        
        r16_fixtures = [
            ("Match 89", q[0], q[2]),   
            ("Match 90", q[1], q[4]),   
            ("Match 91", q[3], q[5]),   
            ("Match 92", q[6], q[7]),   
            ("Match 93", q[10], q[11]), 
            ("Match 94", q[8], q[9]),   
            ("Match 95", q[13], q[15]), 
            ("Match 96", q[12], q[14]), 
        ]
        
        r16_results = []
        local_qf_list = []
        
        for match_num, home, away in r16_fixtures:
            winner, score = simulate_knockout(home, away, poisson_model, current_rank_dict)
            r16_results.append({"Match": match_num, "Team 1": home, "Result": score, "Team 2": away, "Advancing": winner})
            local_qf_list.append(winner)
            
        st.session_state.qf_qualifiers = local_qf_list
        df_r16 = pd.DataFrame(r16_results)
        
        # --- TOP ROW: The Data Table ---
        st.subheader("📋 Match Data")
        st.dataframe(df_r16, hide_index=True, width='stretch')
        st.divider() 
        
        # --- BOTTOM ROW: The Expanding Visual Tree (Split Layout) ---
        st.subheader("🌳 Tournament Progression")
        
        col_tree_left, col_tree_right = st.columns(2)
        import graphviz
        
        # --- LEFT SIDE (First 4 Matches of R16) ---
        with col_tree_left:
            dot_L = graphviz.Digraph()
            # 🚨 ADDED outputorder='edgesfirst' 🚨
            dot_L.attr(rankdir='LR', bgcolor='transparent', splines='ortho', ranksep='0.4', nodesep='0.05', outputorder='edgesfirst')
            
            dot_L.node_attr.update(shape='box', style='rounded,filled', fillcolor='#262730', 
                                   color='#4f8bf9', fontcolor='white', fontsize='11', 
                                   fixedsize='true', width='1.5', height='0.4')
            dot_L.edge_attr.update(color='white', penwidth='1.5', dir='none')
            
            if 'df_r32' in st.session_state:
                for index, row in st.session_state.df_r32.iloc[:8].iterrows():
                    id_t1, id_t2, id_winner = f"R32_{row['Team 1']}", f"R32_{row['Team 2']}", f"R16_{row['Advancing']}"
                    dot_L.node(id_t1, row['Team 1'])
                    dot_L.node(id_t2, row['Team 2'])
                    dot_L.node(id_winner, row['Advancing'], fillcolor='#262730') 
                    dot_L.edge(id_t1, id_winner, tailport='e', headport='w')
                    dot_L.edge(id_t2, id_winner, tailport='e', headport='w')

            for index, row in df_r16.iloc[:4].iterrows():
                id_home, id_away, id_qf = f"R16_{row['Team 1']}", f"R16_{row['Team 2']}", f"QF_{row['Advancing']}"
                dot_L.node(id_qf, row['Advancing'], fillcolor='#17b169') 
                dot_L.edge(id_home, id_qf, tailport='e', headport='w')
                dot_L.edge(id_away, id_qf, tailport='e', headport='w')
            st.graphviz_chart(dot_L, use_container_width=True)

        # --- RIGHT SIDE (Last 4 Matches of R16) ---
        with col_tree_right:
            dot_R = graphviz.Digraph()
            # 🚨 ADDED outputorder='edgesfirst' 🚨
            dot_R.attr(rankdir='RL', bgcolor='transparent', splines='ortho', ranksep='0.4', nodesep='0.05', outputorder='edgesfirst')
            
            dot_R.node_attr.update(shape='box', style='rounded,filled', fillcolor='#262730', 
                                   color='#4f8bf9', fontcolor='white', fontsize='11', 
                                   fixedsize='true', width='1.5', height='0.4')
            dot_R.edge_attr.update(color='white', penwidth='1.5', dir='none')
            
            if 'df_r32' in st.session_state:
                for index, row in st.session_state.df_r32.iloc[8:].iterrows():
                    id_t1, id_t2, id_winner = f"R32_{row['Team 1']}", f"R32_{row['Team 2']}", f"R16_{row['Advancing']}"
                    dot_R.node(id_t1, row['Team 1'])
                    dot_R.node(id_t2, row['Team 2'])
                    dot_R.node(id_winner, row['Advancing'], fillcolor='#262730') 
                    dot_R.edge(id_t1, id_winner, tailport='w', headport='e')
                    dot_R.edge(id_t2, id_winner, tailport='w', headport='e')

            for index, row in df_r16.iloc[4:].iterrows():
                id_home, id_away, id_qf = f"R16_{row['Team 1']}", f"R16_{row['Team 2']}", f"QF_{row['Advancing']}"
                dot_R.node(id_qf, row['Advancing'], fillcolor='#17b169') 
                dot_R.edge(id_home, id_qf, tailport='w', headport='e')
                dot_R.edge(id_away, id_qf, tailport='w', headport='e')
            st.graphviz_chart(dot_R, use_container_width=True)