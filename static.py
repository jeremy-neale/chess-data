# This file generates the static plots
# Jeremy Neale

# ========================= Sources ==========================

# https://dash.plotly.com/dash-core-components/tabs

# ====================== Imports ====================================

import kagglehub
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================== Constants and Setup =================

PLOTS_DIR = 'plots/static'
os.makedirs(PLOTS_DIR, exist_ok=True)

source = 'https://www.kaggle.com/datasets/arevel/chess-games'
url = 'https://www.kaggle.com/api/v1/datasets/download/arevel/chess-games'

pd.set_option('display.max_columns', None)

# =================== StaticPlots Class ====================

class StaticPlots:

    def __init__(self):
        # Convert to dataframe
        csv_path = "chess_games.csv"
        self.df = pd.read_csv(csv_path, nrows=50000)

    # =============== Plot Creation =========================

    def make_static_plots(self):
        print("Plotting static plots...")
        self.make_line_plot()
        self.make_group_bar_plot()
        self.make_count_plot()
        self.make_pie_chart()
        self.make_avg_elo_distplot()
        self.make_pair_plot()
        self.make_heat_map()
        self.make_hist_kde()
        self.make_kde()
        self.make_reg_plot()
        self.make_box_plot()
        self.make_boxen_plot()
        self.make_area_plot()
        self.make_violin_plot()
        self.make_joint_plot()
        self.make_rug_plot()
        self.make_3d_plot()
        self.make_cluster_map()
        self.make_hexbin()
        self.make_strip_plot()
        self.make_swarm_plot()

    def make_swarm_plot(self):
        plt.figure(figsize=(10, 6))
        sns.swarmplot(
            data=self.df,
            x='Result-string',
            y='total_checks',
            size = 3
        )
        plt.title('Swarm Plot: Total Checks by Game Result')
        plt.xlabel('Game Result')
        plt.ylabel('Total Checks')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/swarmplot_result_checks.png")
        plt.close()

    def make_strip_plot(self):
        plt.figure(figsize=(12, 6))
        sns.stripplot(
            data=self.df,
            x='Event',
            y='total_captures',
            alpha=0.5
        )
        plt.xlabel('Game Type')
        plt.ylabel('Total Captures')
        plt.title('Strip Plot: Total Captures by Game Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/stripplot_event_captures.png")
        plt.close()

    def make_hexbin(self):
        plt.figure(figsize=(10, 6))
        plt.hexbin(
            self.df['avg_elo'],
            self.df['first_castle_turn'],
            gridsize=30,
            cmap='viridis',
            mincnt=1
        )
        plt.colorbar(label='Count in bin')
        plt.xlabel('Average Elo')
        plt.ylabel('First Castle Turn')
        plt.title('Hexbin Plot: Average Elo vs First Castle Turn')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/hexbin_avgelo_castleturn.png")
        plt.close()

    def make_cluster_map(self):
        cols = [
            'avg_elo',
            'total_captures',
            'total_checks',
            'first_castle_turn',
            'first_capture_turn',
            'first_queen_move_turn'
        ]
        cleaned = self.df[cols].dropna()

        sns.clustermap(
            cleaned,
            standard_scale=1,
            cmap='coolwarm',
            figsize=(10, 10)
        )

        plt.suptitle("Clustermap of Chess Game Features", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/clustermap_features.png")
        plt.close()

    def make_3d_plot(self):
        # Was having issues with the 3d plot with contour so I just did a scatter :(
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        x = self.df['avg_elo']
        y = self.df['first_castle_turn']
        z = self.df['total_captures']

        ax.scatter(x, y, z, c=z, cmap='coolwarm', alpha=0.7)

        ax.set_xlabel('Average Elo')
        ax.set_ylabel('First Castle Turn')
        ax.set_zlabel('Total Captures')
        ax.set_title('3D Scatter: Elo vs Castle Turn vs Captures')

        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/3d_elo_castle_capture.png")
        plt.close()

    def make_rug_plot(self):
        plt.figure(figsize=(10, 2))
        sns.rugplot(
            data=self.df,
            x='total_checks',
            lw=1,
            alpha=0.7,
            color='green'
        )
        plt.title('Rug Plot of Total Checks per Game', fontsize=14)
        plt.xlabel('Total Checks')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/rugplot_total_checks.png')
        plt.close()

    def make_joint_plot(self):
        # Main (center) join plot
        joint_plot = sns.jointplot(
            data=self.df,
            x='avg_elo',
            y='first_castle_turn',
            kind='scatter',
            height=8,
            color='gray'
        )

        # Add KDE contours on top of scatter
        sns.kdeplot(
            data=self.df,
            x='avg_elo',
            y='first_castle_turn',
            ax=joint_plot.ax_joint,
            levels=5,
            linewidths=1.5,
            color='blue'
        )

        plt.suptitle('Joint Plot of Avg Elo vs First Castle Turn', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/joint_avg_elo_castle.png')
        plt.close()

    def make_violin_plot(self):
        plt.figure(figsize=(12, 6))
        sns.violinplot(
            data=self.df,
            x='Event',
            y='first_queen_move_turn'
        )
        plt.title('Turns Until a Queen is Moved by Game Type', fontsize=14, pad=15)
        plt.xlabel('Game Type')
        plt.ylabel('Turn of First Queen Move')
        plt.xticks(rotation=20)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/violin_queen_move_by_event.png')
        plt.close()

    def make_area_plot(self):
        plt.figure(figsize=(10, 6))

        # Sort df by result
        sorted_df = self.df.sort_values(by='Result-string')

        # Use seaborn to get a smooth KDE area plot
        sns.kdeplot(
            data=sorted_df,
            x='avg_elo',
            hue='Result-string',
            fill=True,
            common_norm=False,
            alpha=0.5,
            linewidth=2
        )

        plt.title('Elo Distribution by Game Result (Area Plot)', fontsize=14, pad=15)
        plt.xlabel('Average Elo')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/area_plot_avg_elo_by_result.png')
        plt.close()

    def make_box_plot(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[['total_captures', 'total_checks']])
        plt.title('Box Plot of Total Captures and Checks')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/box_plot_captures_checks.png')
        plt.close()

    def make_boxen_plot(self):
        plt.figure(figsize=(10, 6))
        sns.boxenplot(data=self.df[['total_captures', 'total_checks']])
        plt.title('Boxen Plot of Total Captures and Checks')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/boxen_plot_caputures_checks.png')
        plt.close()

    def make_reg_plot(self):
        plt.figure(figsize=(10, 6))

        # Drop rows where no one castled
        df_castle = self.df.dropna(subset=['first_castle_turn'])

        sns.regplot(
            data=df_castle,
            x='avg_elo',
            y='first_castle_turn',
            scatter_kws={'alpha': 0.5, 'color': 'teal'},
            line_kws={'color': 'darkred', 'linewidth': 2}
        )

        plt.title("Avg Elo vs. First Castling Turn", fontsize=14, pad=15)
        plt.xlabel("Average Elo")
        plt.ylabel("Turn Number When Castling First Occurs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/regplot_avg_elo_vs_castle.png')
        plt.close()

    def make_kde(self):
        plt.figure(figsize=(10, 6))

        sns.kdeplot(
            self.df['avg_elo'].dropna(),
            fill=True,
            alpha=0.6,
            linewidth=2
        )
        sns.color_palette("inferno")

        plt.title("KDE Plot of Average Elo", fontsize=14, pad=15)
        plt.xlabel("Average Elo")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/kde_avg_elo.png')
        plt.close()

    def make_qq(self):
        plt.figure(figsize=(10, 8))
        qqplot(self.df['avg_elo'].dropna(), line='s')
        plt.title("QQ Plot of Average Elo", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/qqplot_avg_elo.png')
        plt.close()

    def make_line_plot(self):
        counts = self.df['first_castle_turn'].value_counts().sort_index()
        plt.plot(counts.index, counts.values)
        plt.xlabel('Number of turns until a player castles')
        plt.ylabel('Number of games')
        plt.title('Turns Until a Player Castles')
        plt.grid(True)
        plt.tight_layout()
        figname = 'line_castle_vs_games.png'
        plt.savefig(f'{PLOTS_DIR}/{figname}')
        plt.close()

    def make_group_bar_plot(self):
        # Count values for each type
        white_counts = self.df['white_castle'].value_counts(dropna=False)
        black_counts = self.df['black_castle'].value_counts(dropna=False)

        labels = ['Kingside (short)', 'Queenside (long)', 'Never castled']

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, white_counts.values, width, label='White', color='#dddddd')
        ax.bar(x + width / 2, black_counts.values, width, label='Black', color='black')

        ax.set_xlabel('Castling Type')
        ax.set_ylabel('Number of Games')
        ax.set_title('Castling Type by Color (Grouped Bar Plot)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/grouped_castling_bar.png')
        plt.close()


    def make_count_plot(self):
        sns.countplot(
            x='Result-string',
            data=self.df,
            palette={
                'white': '#dddddd',
                'black': 'black',
                'draw': 'blue'
            }
        )
        plt.title('Match result countplot')
        plt.xlabel('Result')
        plt.ylabel('Number of Games')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/countplot_white_black_wins_results.png')
        plt.close()

    # Makes 3 pie charts based on who castles, brings queen out, and captures first
    def make_pie_chart(self):
        fig, axes = plt.subplots(1, 3, figsize=(16, 8))

        pie_configs = [
            ('first_castle_color', 'Who Castled First', axes[0]),
            ('first_capture_color', 'Who Captured First', axes[1]),
            ('first_queen_move_color', 'Who Moved Queen First', axes[2])
        ]

        label_order = ['white', 'black', 'neither']
        label_names = ['White', 'Black', 'Neither']
        color_map = {'white': '#dddddd', 'black': 'black', 'neither': 'blue'}

        for col, title, ax in pie_configs:
            series = self.df[col].fillna('neither')
            counts = series.value_counts()

            sizes = [counts.get(label, 0) for label in label_order]
            colors = [color_map[label] for label in label_order]

            ax.pie(
                sizes,
                labels=label_names,
                colors=colors,
                autopct='%1.2f%%',
                pctdistance=1.2,
                labeldistance=1.4
            )
            ax.set_title(title, fontsize=25, pad=42)

        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/pies_first_castle_capture_queen.png')
        plt.close()

    def make_avg_elo_distplot(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.df,
            x='avg_elo',
            kde=True,
            bins=30,
            color='green',
            edgecolor='black',
            linewidth=1
        )

        plt.title('Distribution of Average Elo of each Match')
        plt.xlabel('Average Elo Rating')
        plt.ylabel('Number of Games')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/distplot_avg_elo.png')
        plt.close()

    def make_pair_plot(self):
        cols = [
            'avg_elo',
            'first_queen_move_turn',
            'first_capture_turn',
            'first_castle_turn',
            'Result-string'  # For hue
        ]

        df_clean = self.df[cols].dropna()

        # Create the pair plot
        pair = sns.pairplot(
            df_clean,
            hue='Result-string',
            palette={'white': 'grey', 'black': 'black', 'draw': 'blue'},
            diag_kind='kde',
            plot_kws=dict(alpha=0.6)
        )

        pair.figure.suptitle('Pair Plot of Elo and Game Timing Features', y=1.02, fontsize=14)
        pair.figure.tight_layout()

        # Legen - put in top right
        pair._legend.set_bbox_to_anchor((1, 1))
        pair._legend.set_title("Result")

        pair.savefig(f'{PLOTS_DIR}/pairplot_elo_timing_result.png')
        plt.close()

    def make_heat_map(self):
        cols = ['avg_elo', 'first_queen_move_turn', 'first_capture_turn', 'first_castle_turn']
        corr_matrix = self.df[cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            cbar=True
        )

        plt.title('Correlation Heatmap of First Events and Average Elo')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/heatmap_elo_timing.png')
        plt.close()

    def make_hist_kde(self):
        plt.figure(figsize=(10, 6))

        sns.histplot(
            data=self.df,
            x='avg_elo',
            kde=True,  # Overlay KDE
            bins=30,  # Number of histogram bins
            color='steelblue',
            edgecolor='black',
            linewidth=1.2
        )

        plt.title('Distribution of Average Elo with KDE', fontsize=16)
        plt.xlabel('Average Elo')
        plt.ylabel('Count')
        plt.tight_layout()

        plt.savefig(f'{PLOTS_DIR}/hist_kde_avg_elo.png')
        plt.close()


    # ================= Feature Engineering =========================

    def feature_engineering(self):
        print("Feature engineering...")
        self.make_queen_columns()
        self.make_capture_columns()
        self.make_castle_columns()
        self.fix_results()
        self.df['avg_elo'] = (self.df['WhiteElo'] + self.df['BlackElo']) / 2
        self.make_total_capture_columns()
        self.make_check_columns()

    def make_total_capture_columns(self):
        self.df[['num_white_captures', 'num_black_captures']] = self.df['AN'].apply(
            lambda x: pd.Series(count_occurrence(x, 'x')))
        self.df['total_captures'] = self.df['num_white_captures'] + self.df['num_black_captures']

    def make_check_columns(self):
        self.df[['num_white_checks', 'num_black_checks']] = self.df['AN'].apply(
            lambda x: pd.Series(count_occurrence(x, '+')))
        self.df['total_checks'] = self.df['num_white_checks'] + self.df['num_black_checks']

    def make_queen_columns(self):
        self.df['first_white_queen_move'] = self.df['AN'].apply(lambda x: first_move_pattern(x, 'Q', 'white')[2])
        self.df['first_black_queen_move'] = self.df['AN'].apply(lambda x: first_move_pattern(x, 'Q', 'black')[2])
        self.df['first_queen_move_turn'] = self.df['AN'].apply(lambda x: first_move_pattern(x, 'Q', 'either')[2])
        self.df['first_queen_move_color'] = self.df['AN'].apply(lambda x: first_move_color(x, 'Q'))

    def make_capture_columns(self):
        self.df['first_white_capture'] = self.df['AN'].apply(lambda x: first_move_pattern(x, 'x', 'white')[2])
        self.df['first_black_capture'] = self.df['AN'].apply(lambda x: first_move_pattern(x, 'x', 'black')[2])
        self.df['first_capture_turn'] = self.df['AN'].apply(lambda x: first_move_pattern(x, 'x', 'either')[2])
        self.df['first_capture_color'] = self.df['AN'].apply(lambda x: first_move_color(x, 'x'))

    def make_castle_columns(self):
        self.df['first_castle_turn'] = self.df['AN'].apply(lambda x: first_move_pattern(x, 'O-O', 'either')[2])
        self.df['first_castle_color'] = self.df['AN'].apply(lambda x: first_move_color(x, 'O-O'))
        self.df[['white_castle', 'black_castle']] = self.df['AN'].apply(lambda x: pd.Series(detect_castling_type(x)))

    def fix_results(self):
        result_map = {
            '1-0': 'white',
            '0-1': 'black',
            '1/2-1/2': 'draw'
        }

        self.df['Result-string'] = self.df['Result'].map(result_map)

# ======================== Helper Functions ===================================

# This function takes a string of Chess moves in the format:
# 1. e4 e5 2. Kf3 Kc6 3. ...
# Then, if the pattern parameter is found in the sequence of moves,
# It returns the white move, the black move, and the turn number
def first_move_pattern(moves, pattern, color):
    moves = moves.split('.')

    for i, move in enumerate(moves):
        if i == 0 or pattern not in move:
            continue
        # Find both moves for the turn
        split_moves = move.split()
        if len(split_moves) != 3:
            continue
        white_move = split_moves[0]
        black_move = split_moves[1]

        # Separate cases based on the 'color' param
        if color == 'white':
            if pattern in white_move:
                return white_move, None, i
        elif color == 'black':
            if pattern in black_move:
                return None, black_move, i
        elif color == 'either':
            if pattern in white_move or pattern in black_move:
                return white_move, black_move, i

    return None, None, None

# Returns the COLOR of which player matches the pattern first
def first_move_color(moves, pattern):
    result = first_move_pattern(moves, pattern, 'either')

    if result is None:
        return None

    white_move, black_move, turn_number = result

    if turn_number is None:
        return None
    elif pattern in white_move:
        return 'white'
    elif pattern in black_move:
        return 'black'
    else:
        return None


# Similar to the move_pattern function, but specific to castle
def detect_castling_type(moves):
    white_castle = None
    black_castle = None

    turns = moves.split('.')
    for i, turn in enumerate(turns):
        if i == 0:
            continue
        parts = turn.strip().split()

        if len(parts) != 3:
            continue

        white_move = parts[0]
        black_move = parts[1]

        if white_castle is None:
            if 'O-O-O' in white_move:
                white_castle = 'queenside'
            elif 'O-O' in white_move:
                white_castle = 'kingside'

        if black_castle is None:
            if 'O-O-O' in black_move:
                black_castle = 'queenside'
            elif 'O-O' in black_move:
                black_castle = 'kingside'

        # Save time if both have been found
        if white_castle and black_castle:
            break

    return white_castle, black_castle

def count_occurrence(moves, occurrence):
    white_count = 0
    black_count = 0

    moves = moves.split('.')

    for i, move in enumerate(moves):
        if i == 0 or occurrence not in move:
            continue
        # Find both moves for the turn
        split_moves = move.split()
        if len(split_moves) != 3:
            continue
        white_move = split_moves[0]
        black_move = split_moves[1]

        # Look for matches
        if occurrence in white_move:
            white_count += 1
        if occurrence in black_move:
            black_count += 1

    return white_count, black_count

# ======================================================

# Functionality
plots = StaticPlots()
plots.feature_engineering()

print(plots.df.head())
# plots.df.to_csv('chess_games_50k-updated.csv', index=False)
plots.make_static_plots()
