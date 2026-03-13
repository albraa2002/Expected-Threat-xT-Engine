# 🎯 Expected Threat (xT) Spatial Engine & Pass Evaluation Dashboard

## 📌 Project Overview
Traditional football metrics like goals or assists fail to capture the value of players who orchestrate the build-up phase. This Data Science project implements an **Expected Threat (xT) Spatial Engine**, a highly advanced predictive metric that assigns a probability value to every coordinate on the pitch, evaluating how much a specific pass increases the team's chances of scoring.

## 🚀 Key Analytical Value
* **Spatial Mathematical Modeling:** Divided the pitch into a 12x8 coordinate grid, generating a synthetic xT matrix where values exponentially increase towards the opponent's penalty box.
* **Action Valuation (xT Added):** Developed a Python algorithm to calculate the `xT_Added` for individual passes (xT of end location minus xT of start location), objectively quantifying a playmaker's ability to break defensive lines.
* **Advanced Tactical UI:** Built an interactive Plotly dashboard featuring a spatial heatmap, directional pass vectors (color-coded by positive/negative xT impact), and an xT Leaderboard.

## 🛠️ Tech Stack
* **Domain:** Sports Data Science, Advanced Football Analytics (xT)
* **Language:** Python 3.x
* **Data Manipulation:** `Pandas`, `NumPy`
* **Visualization:** `Plotly` (Heatmaps, Annotations, Vectors)

## 👤 Author
**Albaraa Ehab**
* **Role:** Data Analyst / Sports Analytics
