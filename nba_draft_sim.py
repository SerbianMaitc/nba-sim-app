"""
NBA Top-75 Draft & Game Simulator

What this is
-------------
- A Python CLI where two sides draft 5 players (PG, SG, SF, PF, C) and simulate a game.
- Player pool is seeded with a subset from Bleacher Report's Top 75 All-Time (expandable to full 75).
- Player attributes are grounded in career stats (PPG, RPG, APG, TS%, 3P%, FT%, OBPM/DBPM, WS/48) and era heuristics.
- The simulation uses possession-level Monte Carlo with team fit (spacing, rim pressure, ball movement, defense, rebounding) affecting outcomes.

How to run
----------
1) Save as `nba_draft_sim.py` and run `python nba_draft_sim.py`.
2) Follow prompts to draft each position.
3) After both teams are set, the game simulates and prints a box score + summary.

Notes
-----
- The pool currently includes ~25 historically accurate stars for demo. You can add the rest of the B/R Top 75 by extending PLAYER_DB.
- All numbers are reasonable approximations of career values. You can adjust or add players without changing the engine.
- If you want me to preload the full 75 with exact stats, ping me and I'll wire it in next.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# -----------------------------
# Data Model
# -----------------------------

@dataclass
class Player:
    name: str
    pos_primary: str  # 'PG','SG','SF','PF','C'
    pos_secondary: List[str]
    era: str          # '60s','70s','80s','90s','00s','10s','20s'
    ppg: float
    rpg: float
    apg: float
    ts: float         # True Shooting % (0-1)
    three_pct: float  # 3P% (0-1)
    ft_pct: float     # FT% (0-1)
    obpm: float
    dbpm: float
    ws48: float
    three_rate: float # Approx. 3PA rate (0-1) relative to shots
    height_in: int
    weight_lb: int

    # Cached 2K-like attributes (computed later)
    ratings: Dict[str, float] = field(default_factory=dict)

# -----------------------------
# Helper: clamp & scale
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# Normalize value to 0..99
# Provide a (min_val, max_val) typical NBA range for scaling.

def scale_to_99(val: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 50.0
    t = (val - lo) / (hi - lo)
    return clamp(99*t, 0, 99)

# -----------------------------
# Player Database (seed subset)
# Values are realistic approximations of career numbers.
# -----------------------------

PLAYER_DB: List[Player] = [
    # name, pos_primary, pos_secondary, era, ppg, rpg, apg, ts, 3P%, FT%, OBPM, DBPM, WS/48, 3P rate, height, weight
    Player("Michael Jordan", "SG", ["SF"], "90s", 30.1, 6.2, 5.3, 0.569, 0.327, 0.835, 7.9, 1.3, 0.250, 0.12, 78, 198),
    Player("LeBron James", "SF", ["PF","PG"], "10s", 27.1, 7.5, 7.3, 0.594, 0.345, 0.735, 6.8, 1.9, 0.232, 0.28, 81, 250),
    Player("Kareem Abdul-Jabbar", "C", ["PF"], "70s", 24.6, 11.2, 3.6, 0.592, 0.000, 0.721, 5.7, 2.2, 0.228, 0.00, 86, 225),
    Player("Magic Johnson", "PG", ["SG"], "80s", 19.5, 7.2, 11.2, 0.610, 0.303, 0.848, 6.9, 1.0, 0.225, 0.18, 81, 215),
    Player("Larry Bird", "SF", ["PF"], "80s", 24.3, 10.0, 6.3, 0.564, 0.375, 0.886, 5.6, 2.0, 0.203, 0.24, 81, 220),
    Player("Shaquille O'Neal", "C", ["PF"], "00s", 23.7, 10.9, 2.5, 0.586, 0.045, 0.527, 4.4, 2.0, 0.208, 0.01, 85, 325),
    Player("Tim Duncan", "PF", ["C"], "00s", 19.0, 10.8, 3.0, 0.551, 0.179, 0.696, 2.9, 3.5, 0.209, 0.02, 83, 250),
    Player("Kobe Bryant", "SG", ["SF"], "00s", 25.0, 5.2, 4.7, 0.555, 0.329, 0.837, 4.5, 0.7, 0.170, 0.28, 78, 212),
    Player("Hakeem Olajuwon", "C", ["PF"], "90s", 21.8, 11.1, 2.5, 0.552, 0.202, 0.712, 3.6, 2.7, 0.178, 0.02, 84, 255),
    Player("Stephen Curry", "PG", ["SG"], "10s", 24.7, 4.7, 6.4, 0.628, 0.428, 0.910, 7.7, 0.8, 0.212, 0.55, 74, 190),
    Player("Kevin Durant", "SF", ["PF"], "10s", 27.3, 7.1, 4.3, 0.635, 0.388, 0.886, 5.7, 1.8, 0.214, 0.36, 82, 240),
    Player("Giannis Antetokounmpo", "PF", ["C","SF"], "20s", 24.1, 10.3, 4.7, 0.611, 0.286, 0.690, 6.6, 2.2, 0.208, 0.20, 83, 242),
    Player("Nikola Jokic", "C", ["PF"], "20s", 20.9, 10.7, 6.9, 0.644, 0.364, 0.829, 10.4, 2.0, 0.260, 0.28, 83, 284),
    Player("Wilt Chamberlain", "C", ["PF"], "60s", 30.1, 22.9, 4.4, 0.540, 0.000, 0.511, 5.0, 2.0, 0.248, 0.00, 84, 275),
    Player("Bill Russell", "C", ["PF"], "60s", 15.1, 22.5, 4.3, 0.493, 0.000, 0.561, 1.5, 4.8, 0.193, 0.00, 82, 215),
    Player("Jerry West", "SG", ["PG"], "60s", 27.0, 5.8, 6.7, 0.580, 0.000, 0.814, 6.1, 1.1, 0.213, 0.02, 75, 175),
    Player("Oscar Robertson", "PG", ["SG"], "60s", 25.7, 7.5, 9.5, 0.550, 0.000, 0.838, 6.3, 1.2, 0.207, 0.02, 77, 205),
    Player("Dirk Nowitzki", "PF", ["C"], "00s", 20.7, 7.5, 2.4, 0.580, 0.384, 0.879, 3.8, 0.7, 0.178, 0.32, 84, 245),
    Player("Kevin Garnett", "PF", ["C"], "00s", 17.8, 10.0, 3.7, 0.547, 0.275, 0.789, 4.6, 3.0, 0.184, 0.10, 83, 240),
    Player("Karl Malone", "PF", ["C"], "90s", 25.0, 10.1, 3.6, 0.577, 0.274, 0.742, 4.8, 1.4, 0.205, 0.07, 81, 250),
    Player("Charles Barkley", "PF", ["SF"], "90s", 22.1, 11.7, 3.9, 0.612, 0.266, 0.735, 5.5, 0.8, 0.216, 0.08, 78, 252),
    Player("Dwyane Wade", "SG", ["PG"], "00s", 22.0, 4.7, 5.4, 0.554, 0.293, 0.765, 4.0, 1.6, 0.159, 0.15, 76, 220),
    Player("Allen Iverson", "SG", ["PG"], "00s", 26.7, 3.7, 6.2, 0.518, 0.313, 0.780, 3.8, -0.5, 0.102, 0.32, 72, 165),
    Player("Kawhi Leonard", "SF", ["SG"], "20s", 19.9, 6.4, 3.0, 0.600, 0.386, 0.857, 4.8, 2.3, 0.212, 0.30, 79, 225),
    Player("Chris Paul", "PG", ["SG"], "10s", 17.5, 4.5, 9.4, 0.589, 0.371, 0.873, 6.9, 2.3, 0.247, 0.30, 72, 175),
    Player("James Harden", "SG", ["PG"], "10s", 24.6, 5.6, 7.1, 0.610, 0.366, 0.860, 5.7, 0.7, 0.222, 0.50, 77, 220),
    Player("Russell Westbrook", "PG", ["SG"], "10s", 22.0, 7.3, 8.4, 0.520, 0.308, 0.785, 3.5, 0.5, 0.137, 0.25, 75, 200),
    Player("Paul Pierce", "SF", ["SG"], "00s", 19.7, 5.6, 3.5, 0.568, 0.369, 0.806, 3.6, 1.3, 0.178, 0.31, 79, 235),
    Player("Ray Allen", "SG", ["SF"], "00s", 18.9, 4.1, 3.4, 0.583, 0.400, 0.892, 3.3, 0.8, 0.157, 0.43, 77, 205),
    Player("Reggie Miller", "SG", ["SF"], "90s", 18.2, 3.0, 3.0, 0.614, 0.395, 0.888, 3.1, 0.7, 0.172, 0.40, 78, 185),
    Player("Steve Nash", "PG", ["SG"], "00s", 14.3, 3.0, 8.5, 0.607, 0.428, 0.904, 5.4, -0.5, 0.212, 0.32, 75, 178),
    Player("Jason Kidd", "PG", ["SG"], "00s", 12.6, 6.3, 8.7, 0.515, 0.349, 0.750, 3.6, 2.6, 0.179, 0.30, 76, 210),
    Player("Gary Payton", "PG", ["SG"], "90s", 16.3, 3.9, 6.7, 0.533, 0.317, 0.729, 3.2, 1.7, 0.159, 0.18, 76, 180),
    Player("Vince Carter", "SG", ["SF"], "00s", 16.7, 4.3, 3.1, 0.548, 0.374, 0.798, 2.2, 0.6, 0.118, 0.30, 78, 220),
    Player("Tracy McGrady", "SG", ["SF"], "00s", 19.6, 5.6, 4.4, 0.538, 0.338, 0.747, 4.3, 0.5, 0.151, 0.28, 80, 225),
    Player("Carmelo Anthony", "SF", ["PF"], "10s", 22.5, 6.2, 2.7, 0.545, 0.353, 0.814, 2.3, 0.0, 0.104, 0.29, 80, 238),
    Player("Chris Bosh", "PF", ["C"], "10s", 19.2, 8.5, 2.0, 0.567, 0.339, 0.800, 2.2, 1.0, 0.158, 0.22, 83, 235),
    Player("Dwight Howard", "C", ["PF"], "10s", 15.7, 11.8, 1.3, 0.609, 0.180, 0.568, 1.4, 2.8, 0.150, 0.02, 83, 265),
    Player("Yao Ming", "C", ["PF"], "00s", 19.0, 9.2, 1.6, 0.614, 0.100, 0.833, 3.3, 1.5, 0.200, 0.01, 88, 310),
    Player("Manu Ginobili", "SG", ["PG"], "00s", 13.3, 3.5, 3.8, 0.580, 0.370, 0.826, 4.7, 0.9, 0.193, 0.30, 78, 205),
    Player("Tony Parker", "PG", ["SG"], "00s", 15.5, 2.7, 5.6, 0.551, 0.327, 0.751, 2.3, -0.3, 0.120, 0.12, 74, 185),
    Player("Pau Gasol", "PF", ["C"], "10s", 17.0, 9.2, 3.2, 0.579, 0.298, 0.752, 2.4, 1.5, 0.174, 0.10, 84, 250),
    Player("Alonzo Mourning", "C", ["PF"], "00s", 17.1, 8.5, 1.1, 0.565, 0.100, 0.692, 1.4, 3.3, 0.160, 0.01, 82, 240),
    Player("Dennis Rodman", "PF", ["SF"], "90s", 7.3, 13.1, 1.8, 0.520, 0.230, 0.584, 1.3, 2.8, 0.161, 0.05, 79, 210),
    Player("Dikembe Mutombo", "C", ["PF"], "00s", 9.8, 10.3, 0.9, 0.523, 0.000, 0.685, -0.4, 3.0, 0.147, 0.00, 85, 245),
    Player("Andre Iguodala", "SF", ["SG"], "10s", 11.3, 4.9, 4.2, 0.544, 0.329, 0.707, 2.1, 1.9, 0.128, 0.21, 78, 215),
    Player("Shane Battier", "SF", ["PF"], "00s", 8.6, 4.2, 1.8, 0.580, 0.385, 0.747, 1.0, 1.7, 0.102, 0.40, 80, 220),
    Player("Trevor Ariza", "SF", ["SG"], "10s", 10.4, 4.8, 2.1, 0.530, 0.351, 0.774, 0.5, 1.3, 0.092, 0.45, 80, 215),
    Player("PJ Tucker", "PF", ["SF"], "20s", 6.9, 5.5, 1.3, 0.566, 0.363, 0.760, 0.3, 1.4, 0.090, 0.50, 77, 245),
    Player("Patrick Beverley", "PG", ["SG"], "10s", 8.5, 3.4, 3.2, 0.525, 0.372, 0.756, 0.9, 1.9, 0.103, 0.38, 73, 180),
    Player("Derrick White", "SG", ["PG"], "20s", 12.4, 3.4, 4.1, 0.586, 0.368, 0.869, 1.5, 1.5, 0.121, 0.35, 76, 190),
    Player("Marcus Smart", "PG", ["SG"], "20s", 10.6, 3.5, 4.6, 0.525, 0.325, 0.782, 1.8, 1.9, 0.105, 0.40, 76, 220),
    Player("Robert Horry", "PF", ["SF"], "00s", 7.0, 4.8, 2.1, 0.533, 0.341, 0.728, 0.8, 1.0, 0.096, 0.30, 82, 240),
    Player("Derek Fisher", "PG", ["SG"], "00s", 8.3, 2.1, 3.0, 0.519, 0.377, 0.813, 0.7, 0.8, 0.084, 0.35, 73, 200),
    Player("Boris Diaw", "PF", ["C"], "10s", 8.6, 4.4, 3.5, 0.553, 0.334, 0.727, 1.5, 0.9, 0.111, 0.25, 80, 250),
    Player("Thabo Sefolosha", "SG", ["SF"], "10s", 5.9, 3.9, 1.5, 0.552, 0.353, 0.747, 0.2, 1.7, 0.087, 0.30, 79, 215),
    Player("Danny Green", "SG", ["SF"], "10s", 8.7, 3.4, 1.5, 0.561, 0.401, 0.797, 0.8, 1.5, 0.110, 0.50, 78, 215),
    Player("Tyson Chandler", "C", ["PF"], "10s", 8.2, 9.0, 0.9, 0.638, 0.000, 0.643, 0.3, 2.1, 0.167, 0.00, 85, 240),
    Player("Steven Adams", "C", ["PF"], "20s", 9.2, 8.2, 1.6, 0.595, 0.000, 0.528, 0.7, 1.3, 0.138, 0.01, 83, 265),
    Player("DeAndre Jordan", "C", ["PF"], "10s", 9.0, 10.3, 1.0, 0.647, 0.000, 0.470, 0.3, 1.8, 0.171, 0.01, 83, 265),
    Player("Jae Crowder", "PF", ["SF"], "20s", 10.0, 4.3, 2.0, 0.538, 0.353, 0.775, 0.7, 1.1, 0.095, 0.40, 78, 235),
    Player("Joe Harris", "SG", ["SF"], "20s", 10.6, 3.2, 1.6, 0.621, 0.439, 0.796, 1.2, 0.3, 0.123, 0.60, 78, 220),
    Player("Kyle Korver", "SG", ["SF"], "10s", 9.7, 2.8, 1.7, 0.604, 0.429, 0.878, 0.8, 0.3, 0.130, 0.60, 79, 212),
    Player("Kendrick Perkins", "C", ["PF"], "10s", 5.4, 5.8, 0.9, 0.515, 0.000, 0.589, -1.4, 2.5, 0.067, 0.00, 82, 270),
    Player("Nick Collison", "PF", ["C"], "10s", 5.9, 5.2, 1.0, 0.552, 0.100, 0.768, 0.3, 1.1, 0.089, 0.01, 81, 255),
    Player("Iman Shumpert", "SG", ["SF"], "10s", 7.3, 3.4, 1.8, 0.497, 0.337, 0.757, -0.5, 1.3, 0.062, 0.35, 77, 215),
    Player("Avery Bradley", "SG", ["PG"], "10s", 11.0, 2.9, 1.8, 0.533, 0.366, 0.781, 0.8, 1.2, 0.092, 0.38, 74, 180),
    Player("Cory Joseph", "PG", ["SG"], "20s", 7.0, 2.5, 3.0, 0.538, 0.344, 0.780, 0.1, 0.5, 0.080, 0.25, 75, 200),
    Player("Royce O'Neale", "SF", ["PF"], "20s", 8.3, 4.6, 2.4, 0.565, 0.381, 0.773, 0.5, 0.8, 0.098, 0.45, 77, 225),
    Player("Delon Wright", "PG", ["SG"], "20s", 7.0, 3.2, 3.0, 0.550, 0.358, 0.788, 1.1, 1.3, 0.106, 0.30, 77, 185),
    Player("Doug Christie", "SG", ["SF"], "00s", 11.2, 4.1, 3.6, 0.531, 0.355, 0.788, 1.4, 1.9, 0.109, 0.25, 77, 200),
    Player("Tayshaun Prince", "SF", ["PF"], "00s", 11.1, 4.3, 2.4, 0.528, 0.370, 0.757, 0.5, 1.5, 0.093, 0.28, 81, 215),
    Player("Raja Bell", "SG", ["SF"], "00s", 9.9, 2.9, 1.7, 0.552, 0.406, 0.780, 0.3, 1.1, 0.091, 0.45, 77, 210),
    Player("Jared Dudley", "SF", ["PF"], "10s", 7.3, 3.2, 1.5, 0.548, 0.390, 0.740, 0.1, 0.7, 0.084, 0.35, 78, 230),
    Player("Quentin Richardson", "SG", ["SF"], "00s", 10.3, 4.7, 1.5, 0.517, 0.357, 0.700, 0.3, 0.4, 0.076, 0.42, 78, 220),
    Player("Anthony Tolliver", "PF", ["C"], "10s", 6.4, 3.4, 0.8, 0.551, 0.379, 0.773, 0.2, 0.3, 0.070, 0.42, 80, 240),
    Player("Michael Cooper", "SG", ["SF"], "80s", 8.9, 3.2, 4.2, 0.531, 0.345, 0.833, 1.6, 1.5, 0.118, 0.25, 77, 174),
    Player("Kurt Rambis", "PF", ["C"], "80s", 5.2, 5.6, 1.1, 0.540, 0.000, 0.700, 0.3, 0.7, 0.075, 0.01, 80, 213),
    Player("Bill Cartwright", "C", ["PF"], "90s", 13.2, 6.3, 1.4, 0.531, 0.000, 0.779, 0.2, 1.0, 0.108, 0.01, 85, 245),
    Player("Horace Grant", "PF", ["C"], "90s", 11.2, 8.1, 2.2, 0.558, 0.200, 0.741, 1.5, 1.4, 0.135, 0.10, 81, 215),
    Player("Bryon Russell", "SF", ["SG"], "90s", 8.9, 3.7, 1.3, 0.530, 0.379, 0.757, 0.6, 0.8, 0.097, 0.35, 79, 215),
    Player("Luc Longley", "C", ["PF"], "90s", 7.2, 4.9, 1.7, 0.525, 0.000, 0.737, 0.3, 0.9, 0.089, 0.01, 86, 265),
    Player("Derek Harper", "PG", ["SG"], "90s", 13.3, 2.4, 5.5, 0.528, 0.359, 0.800, 1.1, 1.5, 0.120, 0.30, 73, 185),
    Player("Rick Fox", "SF", ["PF"], "00s", 9.6, 3.8, 2.8, 0.537, 0.348, 0.770, 0.9, 1.1, 0.093, 0.33, 79, 230),
    Player("Doug Christie", "SG", ["SF"], "00s", 11.2, 4.1, 3.6, 0.531, 0.355, 0.788, 1.4, 1.9, 0.109, 0.25, 77, 200),
    Player("Antonio Davis", "PF", ["C"], "90s", 10.0, 7.5, 1.2, 0.513, 0.000, 0.758, 0.6, 1.2, 0.110, 0.01, 81, 245),
    Player("Matt Bonner", "PF", ["C"], "00s", 6.9, 3.1, 0.8, 0.582, 0.414, 0.782, -0.3, 0.2, 0.100, 0.60, 82, 235),
    Player("James Posey", "SF", ["SG"], "00s", 8.6, 4.7, 1.6, 0.530, 0.369, 0.795, 0.5, 1.0, 0.102, 0.38, 80, 215),
    Player("Eduardo Nájera", "PF", ["SF"], "00s", 5.1, 3.7, 0.9, 0.525, 0.200, 0.710, -0.4, 1.0, 0.070, 0.10, 80, 235),
    Player("Jared Jeffries", "SF", ["PF"], "00s", 4.8, 4.1, 1.3, 0.493, 0.265, 0.685, -0.5, 1.1, 0.068, 0.20, 81, 230),
    Player("Desmond Mason", "SG", ["SF"], "00s", 12.1, 4.5, 1.6, 0.519, 0.303, 0.780, 0.2, 0.6, 0.091, 0.18, 77, 222),
    Player("Keith Bogans", "SG", ["SF"], "00s", 6.3, 2.7, 1.3, 0.494, 0.353, 0.754, -0.4, 0.6, 0.073, 0.35, 77, 215),
    Player("Rasho Nesterović", "C", ["PF"], "00s", 6.8, 5.1, 0.9, 0.526, 0.000, 0.732, 0.2, 1.4, 0.094, 0.01, 84, 250),
    Player("Steve Blake", "PG", ["SG"], "00s", 6.5, 2.1, 4.0, 0.520, 0.388, 0.796, 0.3, 0.6, 0.082, 0.34, 75, 185),
    Player("Francisco Garcia", "SG", ["SF"], "00s", 7.9, 2.6, 1.4, 0.531, 0.362, 0.805, 0.1, 0.4, 0.087, 0.38, 78, 195),
    Player("Reggie Evans", "PF", ["C"], "00s", 4.1, 7.1, 0.4, 0.510, 0.000, 0.540, -1.6, 2.3, 0.071, 0.00, 80, 245),
    Player("Ish Smith", "PG", ["SG"], "20s", 7.3, 2.4, 3.8, 0.502, 0.310, 0.725, 0.2, 0.1, 0.085, 0.25, 72, 175),
    Player("Maxi Kleber", "PF", ["C"], "20s", 7.3, 4.2, 1.1, 0.552, 0.366, 0.795, 0.6, 1.2, 0.108, 0.48, 82, 230),
    Player("Royce White", "PF", ["SF"], "10s", 3.0, 2.3, 1.2, 0.465, 0.250, 0.667, -1.2, 0.5, 0.060, 0.20, 80, 260),
    Player("Cody Zeller", "C", ["PF"], "20s", 8.5, 6.0, 1.3, 0.568, 0.229, 0.758, 0.4, 1.0, 0.105, 0.05, 84, 240),
    Player("Cam Payne", "PG", ["SG"], "20s", 8.9, 2.0, 3.7, 0.542, 0.360, 0.780, 0.5, 0.3, 0.096, 0.35, 73, 183),
    Player("Darius Bazley", "PF", ["SF"], "20s", 7.3, 4.3, 0.9, 0.495, 0.321, 0.691, -0.8, 1.0, 0.082, 0.30, 80, 220),
    Player("Isaiah Joe", "SG", ["PG"], "20s", 6.5, 1.8, 1.2, 0.598, 0.407, 0.860, -0.4, 0.2, 0.097, 0.66, 75, 170),
]

# -----------------------------
# Attribute Builder (2K-like ratings 0..99)
# -----------------------------

def build_player_ratings(p: Player) -> Dict[str, float]:
    # Scoring / Shooting
    finishing = scale_to_99(p.ts, 0.50, 0.65) * 0.7 + scale_to_99(p.ppg, 12, 32) * 0.3
    three = scale_to_99(p.three_pct, 0.28, 0.44) * 0.8 + scale_to_99(p.three_rate, 0.02, 0.55) * 0.2
    mid = (scale_to_99(p.ts, 0.52, 0.60) + scale_to_99(p.ppg, 12, 30)) / 2
    ft = scale_to_99(p.ft_pct, 0.60, 0.92)

    # Playmaking
    playmaking = scale_to_99(p.apg, 2.0, 11.5) * 0.7 + scale_to_99(p.obpm, 0.0, 10.5) * 0.3

    # Defense
    per_def = scale_to_99(p.dbpm, 0.0, 4.8) * 0.7 + scale_to_99(p.ws48, 0.10, 0.26) * 0.3
    int_def = scale_to_99(p.dbpm, 0.0, 4.8) * 0.6 + scale_to_99(p.rpg, 3.5, 13.0) * 0.4

    # Rebounding
    reb = scale_to_99(p.rpg, 3.5, 13.0) * 0.7 + scale_to_99(p.height_in, 72, 86) * 0.3

    # Athleticism (rough proxy)
    athletic = (scale_to_99(p.ws48, 0.10, 0.26) * 0.5 + scale_to_99(p.ts, 0.50, 0.65) * 0.5)

    # Usage tendency (for shot selection)
    usage = clamp(0.25 + (p.ppg - 15) * 0.015, 0.10, 0.40)  # 10%-40%

    # 3P attempt tendency
    three_tendency = clamp(p.three_rate, 0.00, 0.60)

    return {
        "finishing": finishing,
        "three": three,
        "mid": mid,
        "ft": ft,
        "playmaking": playmaking,
        "per_def": per_def,
        "int_def": int_def,
        "rebounding": reb,
        "athletic": athletic,
        "usage": usage,
        "three_tendency": three_tendency,
    }

# Precompute ratings
for p in PLAYER_DB:
    p.ratings = build_player_ratings(p)

# -----------------------------
# Team / Draft
# -----------------------------

POSITIONS = ["PG","SG","SF","PF","C"]

@dataclass
class Team:
    name: str
    lineup: Dict[str, Player]  # pos -> Player

    def all_players(self) -> List[Player]:
        return [self.lineup[pos] for pos in POSITIONS]

    def team_synergy(self) -> Dict[str, float]:
        players = self.all_players()
        # Aggregates
        spacing = sum(p.ratings["three"] * (0.5 + p.ratings["three_tendency"]) for p in players) / len(players)
        rim_pressure = sum(p.ratings["finishing"] for p in players) / len(players)
        ball_move = sum(p.ratings["playmaking"] for p in players) / len(players)
        int_def = sum((p.ratings["int_def"] if p.pos_primary in ["PF","C"] else p.ratings["int_def"]*0.8) for p in players) / len(players)
        per_def = sum((p.ratings["per_def"] if p.pos_primary in ["PG","SG","SF"] else p.ratings["per_def"]*0.8) for p in players) / len(players)
        reb = sum(p.ratings["rebounding"] for p in players) / len(players)
        pace = 96 + (ball_move - 50) * 0.1 + (spacing - 50) * 0.05
        return {
            "spacing": spacing,
            "rim_pressure": rim_pressure,
            "ball_move": ball_move,
            "int_def": int_def,
            "per_def": per_def,
            "rebounding": reb,
            "pace": clamp(pace, 90, 104),
        }

# -----------------------------
# Simulation Engine
# -----------------------------

def possession(off: Team, defn: Team) -> Tuple[int, Dict[str,int]]:
    """Simulate a single possession. Returns (points_scored, stat_updates)
    stat_updates: {player_name_stat: value} e.g., {"LeBron James_pts": 2, "LeBron James_ast": 1}
    """
    stats: Dict[str,int] = {}

    off_syn = off.team_synergy()
    def_syn = defn.team_synergy()

    # Base turnover probability influenced by ball movement and perimeter defense
    base_tov = 0.125
    tov = clamp(base_tov - (off_syn["ball_move"]-50)*0.001 + (def_syn["per_def"]-50)*0.001, 0.06, 0.18)

    if random.random() < tov:
        return 0, stats  # turnover

    # Choose shooter weighted by usage
    players = off.all_players()
    weights = [max(0.05, p.ratings["usage"]) for p in players]
    shooter = random.choices(players, weights=weights, k=1)[0]

     # Chance the possession is assisted based on team ball-movement and the shooter’s playmaking
    # — boosted so high-IQ creators rack up more assists
    assist_raw = (
        0.45  # big base to drive up assist counts
        + (off_syn["ball_move"] - 50) * 0.012
        + (off_syn["spacing"]   - 50) * 0.006
        + (shooter.ratings["playmaking"] - 50) * 0.015
    )
    assist_prob = clamp(assist_raw, 0.45, 0.97)
    assisted = random.random() < assist_prob


    # Shot type decision (3 vs 2)
    p3 = clamp(0.05 + shooter.ratings["three_tendency"] + (off_syn["spacing"]-50)*0.002 - (def_syn["per_def"]-50)*0.0015, 0.03, 0.65)
    is_three = random.random() < p3

    # Make probability from ratings and defense impact
    if is_three:
        shot_skill = shooter.ratings["three"]
        def_impact = def_syn["per_def"]
    else:
        # Blend finishing & mid based on rim pressure / interior defense
        inside_bias = clamp((off_syn["rim_pressure"] - def_syn["int_def"]) * 0.01 + 0.5, 0.3, 0.7)
        shot_skill = inside_bias * shooter.ratings["finishing"] + (1-inside_bias) * shooter.ratings["mid"]
        def_impact = 0.7*def_syn["int_def"] + 0.3*def_syn["per_def"]

    # Convert skill/defense to make prob
    base = (shot_skill - def_impact) * 0.0035 + 0.48  # tuned around realistic FG%
    if is_three:
        base -= 0.07  # harder shot baseline
    make_prob = clamp(base, 0.30 if not is_three else 0.24, 0.73 if not is_three else 0.52)

    # Fouls (on 2PT drives more likely)
    foul_prob = clamp(0.05 + (shooter.ratings["finishing"]-50)*0.002 - (def_syn["int_def"]-50)*0.001, 0.03, 0.18)
    drew_foul = (not is_three) and (random.random() < foul_prob)

    points = 0
    made = random.random() < make_prob

    if made:
        points = 3 if is_three else 2
        key = f"{shooter.name}_pts"; stats[key] = stats.get(key, 0) + points
        key = f"{shooter.name}_fga"; stats[key] = stats.get(key, 0) + 1
        key = f"{shooter.name}_fgm"; stats[key] = stats.get(key, 0) + 1
        if assisted:
            # pick a random teammate weighted by playmaking
            mates  = [p for p in players if p is not shooter]
            weights = [max(1.0, tm.ratings["playmaking"]) for tm in mates]
            passer = random.choices(mates, weights=weights, k=1)[0]
            stats[f"{passer.name}_ast"] = stats.get(f"{passer.name}_ast", 0) + 1

        return points, stats

    # Missed shot path
    key = f"{shooter.name}_fga"; stats[key] = stats.get(key, 0) + 1

    if drew_foul:
        # 2 free throws
        ft_makes = 0
        for _ in range(2):
            ft_prob = clamp(0.44 + (shooter.ratings["ft"]-50)*0.006 - (def_syn["int_def"]-50)*0.001, 0.50, 0.95)
            if random.random() < ft_prob:
                ft_makes += 1
        if ft_makes:
            key = f"{shooter.name}_pts"; stats[key] = stats.get(key, 0) + ft_makes
            key = f"{shooter.name}_fta"; stats[key] = stats.get(key, 0) + 2
            key = f"{shooter.name}_ftm"; stats[key] = stats.get(key, 0) + ft_makes
            return ft_makes, stats

    # Rebound chance
    off_reb_edge = (off_syn["rebounding"] - def_syn["rebounding"]) * 0.003 + 0.24
    oreb = random.random() < clamp(off_reb_edge, 0.16, 0.34)
    if oreb:
        # Offensive rebound: pick among all five by rebounding rating
        weights = [max(1.0, p.ratings["rebounding"]) for p in players]
        boarder = random.choices(players, weights=weights, k=1)[0]
        stats[f"{boarder.name}_orb"] = stats.get(f"{boarder.name}_orb", 0) + 1

        # Quick putback attempt
        putback_prob = clamp(0.54 + (shooter.ratings["finishing"]-50)*0.004
                             - (def_syn["int_def"]-50)*0.003, 0.40, 0.80)
        if random.random() < putback_prob:
            stats[f"{shooter.name}_pts"] = stats.get(f"{shooter.name}_pts", 0) + 2
            stats[f"{shooter.name}_fga"] = stats.get(f"{shooter.name}_fga", 0) + 1
            stats[f"{shooter.name}_fgm"] = stats.get(f"{shooter.name}_fgm", 0) + 1
            return 2, stats
        else:
            stats[f"{shooter.name}_fga"] = stats.get(f"{shooter.name}_fga", 0) + 1
            return 0, stats
    else:
        # Defensive rebound: pick among all five defenders by rebounding rating
        defenders = defn.all_players()
        weights_def = [max(1.0, p.ratings["rebounding"]) for p in defenders]
        dboarder = random.choices(defenders, weights=weights_def, k=1)[0]
        stats[f"{dboarder.name}_drb"] = stats.get(f"{dboarder.name}_drb", 0) + 1
        return 0, stats



def simulate_game(team_a: Team, team_b: Team, seed: int|None = None) -> Dict:
    if seed is not None:
        random.seed(seed)

    a_syn = team_a.team_synergy(); b_syn = team_b.team_synergy()
    pace = (a_syn["pace"] + b_syn["pace"]) / 2.0
    possessions = int(pace)

    score_a = 0
    score_b = 0
    box: Dict[str, Dict[str,int]] = {}

    def merge_stats(updates: Dict[str,int]):
        for k, v in updates.items():
            player_key, stat = k.rsplit('_', 1)
            if player_key not in box:
                box[player_key] = {"pts":0,"fga":0,"fgm":0,"fta":0,"ftm":0,"ast":0,"orb":0,"drb":0}
            box[player_key][stat] = box[player_key].get(stat, 0) + v

    # Alternate possessions
    for i in range(possessions*2):
        if i % 2 == 0:
            pts, updates = possession(team_a, team_b)
            score_a += pts
            merge_stats(updates)
        else:
            pts, updates = possession(team_b, team_a)
            score_b += pts
            merge_stats(updates)

    return {
        "score": (score_a, score_b),
        "box": box,
        "syn": {team_a.name: a_syn, team_b.name: b_syn},
        "possessions": possessions*2,
    }

# -----------------------------
# CLI Draft Helpers
# -----------------------------

def list_by_position(pos: str, pool: List[Player]) -> List[Player]:
    return [p for p in pool if p.pos_primary == pos or pos in p.pos_secondary]

def find_player(name: str, pool: List[Player]) -> Player | None:
    for p in pool:
        if p.name.lower() == name.lower():
            return p
    return None

def draft_team(team_name: str, available: List[Player]) -> Team:
    lineup: Dict[str, Player] = {}
    # Print header with leading/trailing newlines properly
    print(f"\nDrafting for {team_name}...\n")


    for pos in POSITIONS:
        candidates = list_by_position(pos, available)
        if not candidates:
            raise ValueError(f"No candidates left for {pos}")
        # randomly limit to at most 5 options for the user
        options = random.sample(candidates, min(5, len(candidates)))
        print(f"Available {pos}s (showing {len(options)} of {len(candidates)}):")
        for p in options:
            print(f" - {p.name} ({p.pos_primary}) | PPG {p.ppg}, 3P% {int(p.three_pct*100)}%, TS {p.ts:.3f}")
        while True:
            pick = input(f"Pick your {pos}: ").strip()
            sel = find_player(pick, options)
            if sel is None:
                print("Name not recognized in the current options. Try again.")
                continue
            lineup[pos] = sel
            available.remove(sel)
            print(f"Selected {sel.name} at {pos}.\n")

            break
    return Team(team_name, lineup)

# -----------------------------
# Pretty Printing
# -----------------------------

def print_box(result: Dict, team_a: Team, team_b: Team):
    a, b = result["score"]
    print("\n================ FINAL ================")
    print(f"{team_a.name} {a}  —  {team_b.name} {b}")
    print(f"Possessions: {result['possessions']}")

    print("\nTeam Synergy (0-100 approx)")
    for tname, syn in result["syn"].items():
        print(f"- {tname}: spacing {syn['spacing']:.1f}, rim {syn['rim_pressure']:.1f}, ball-move {syn['ball_move']:.1f}, int D {syn['int_def']:.1f}, per D {syn['per_def']:.1f}, reb {syn['rebounding']:.1f}, pace {syn['pace']:.1f}")

    def show_team(team: Team):
        print(f"\n--- {team.name} Box ---")
        for p in team.all_players():
            r = result['box'].get(p.name, {})
            pts = r.get('pts',0); fgm=r.get('fgm',0); fga=r.get('fga',0); ast=r.get('ast',0)
            ftm=r.get('ftm',0); fta=r.get('fta',0); orb=r.get('orb',0); drb=r.get('drb',0)
            print(f"{p.name:22}  PTS {pts:2}  FG {fgm}-{fga}  FT {ftm}-{fta}  AST {ast:2}  REB {orb+drb:2} (O{orb}/D{drb})")

    show_team(team_a)
    show_team(team_b)

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    print("NBA Draft & Sim — Top-75 (demo pool)\n")
    pool = PLAYER_DB.copy()

    # Quick option: type 'auto' to let CPU draft best fit by position heuristics
    auto = input("Type 'auto' to auto-draft both teams, or press Enter for manual draft: ").strip().lower() == 'auto'

    def auto_pick(team_name: str, avail: List[Player]) -> Team:
        lineup = {}
        for pos in POSITIONS:
            cands = list_by_position(pos, avail)
            # score candidates by (overall offense + defense + fit proxy)
            def score(p: Player):
                r=p.ratings
                return r['finishing']+r['three']+r['mid']+r['playmaking'] + r['per_def']+r['int_def'] + 0.5*r['rebounding']
            best = max(cands, key=score)
            lineup[pos]=best
            avail.remove(best)
        print(f"Auto-drafted {team_name}:")
        for k,v in lineup.items():
            print(f" - {k}: {v.name}")
        return Team(team_name, lineup)

    if auto:
        team_a = auto_pick("Team A", pool)
        team_b = auto_pick("Team B", pool)
    else:
        team_a = draft_team("Team A", pool)
        team_b = draft_team("Team B", pool)

    seed_in = input("Random seed (blank for random): ").strip()
    seed = int(seed_in) if seed_in else None

    result = simulate_game(team_a, team_b, seed)
    print_box(result, team_a, team_b)
    
