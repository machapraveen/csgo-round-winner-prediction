# CSGO Round Winner Prediction: Machine Learning Analysis

A comprehensive machine learning project that predicts the outcome of Counter-Strike: Global Offensive (CS:GO) rounds based on in-game statistics. The project compares multiple ML algorithms including K-Nearest Neighbors, Random Forest, and Neural Networks to achieve up to 83% accuracy.

## Author
**Macha Praveen**

## Overview

This project analyzes CS:GO match data to predict which team (Terrorist or Counter-Terrorist) will win a round based on various in-game features such as player health, equipment, weapons, and tactical situations. The dataset contains over 122,000 game rounds with detailed statistics.

### Model Performance Comparison
- **K-Nearest Neighbors (Optimized)**: 78.8% accuracy
- **Random Forest**: 83.1% accuracy (best performer)
- **Neural Network (Deep)**: 76.2% accuracy
- **Neural Network (Shallow)**: 75.5% accuracy

## Project Structure

```
CSGO Round Winner Prediction/
├── CSGO Prediction.ipynb    # Main analysis notebook
└── README.md               # This file
```

## Dataset Analysis

### Data Source
The project uses the CS:GO dataset from OpenML containing 122,410 round records with comprehensive game statistics.

### Key Features (Top Correlations with Round Winner)
```python
# Top 10 features most correlated with round winner
t_win                           1.000000
ct_armor                        0.336382  # Counter-Terrorist armor count
ct_helmets                      0.308255  # Counter-Terrorist helmet count
t_helmets                       0.297458  # Terrorist helmet count
ct_defuse_kits                  0.291557  # Counter-Terrorist defuse kit count
t_armor                         0.290753  # Terrorist armor count
ct_grenade_flashbang            0.253868  # CT flashbang grenades
ct_players_alive                0.216798  # CT players still alive
ct_grenade_smokegrenade         0.209975  # CT smoke grenades
ct_weapon_awp                   0.198626  # CT AWP sniper rifles
```

### Dataset Statistics
- **Records**: 122,410 rounds
- **Features**: 20 selected features after correlation analysis
- **Target Balance**: ~51% T wins, ~49% CT wins (well balanced)
- **Data Types**: Equipment counts, player health, weapon counts, grenade inventory

## Implementation Details

### 1. Data Preprocessing
```python
# Download and parse OpenML dataset
url = "https://www.openml.org/data/download/22102255/dataset"
r = requests.get(url, allow_redirects=True)

# Extract features and convert to CSV format
df = pd.read_csv("df.csv")
df["t_win"] = df.round_winner.astype("category").cat.codes

# Feature selection based on correlation (>0.15 threshold)
correlations = df[columns+["t_win"]].corr()
selected_columns = [col for col in columns+['t_win'] 
                   if abs(correlations[col]['t_win']) > 0.15]
```

### 2. K-Nearest Neighbors with Hyperparameter Tuning
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_neighbors": list(range(5, 17, 2)),
    "weights": ["uniform", "distance"]
}

knn = KNeighborsClassifier(n_jobs=-1)
clf = RandomizedSearchCV(knn, param_grid, n_jobs=-1, n_iter=3, cv=3)
clf.fit(X_train_scaled, y_train)

# Best parameters: n_neighbors=15, weights='distance'
# Accuracy: 78.8%
```

### 3. Random Forest Implementation
```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_jobs=-1)
forest.fit(X_train_scaled, y_train)

# Accuracy: 83.1% (best performing model)
```

### 4. Deep Neural Network
```python
from tensorflow import keras

# Deep network architecture
model = keras.models.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(100, activation="relu"), 
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", 
              optimizer="adam", 
              metrics=["accuracy"])

# Training with early stopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5)
model.fit(X_train_scaled_train, y_train_train, 
          epochs=30, 
          callbacks=[early_stopping_cb], 
          validation_data=(X_valid, y_valid))

# Accuracy: 76.2%
```

### 5. Shallow Neural Network
```python
# Simpler architecture for comparison
model = keras.models.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# Accuracy: 75.5%
```

## Installation & Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow requests
```

### Required Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms
- **tensorflow/keras**: Deep learning framework
- **requests**: HTTP library for data download

### Running the Analysis

1. **Open Jupyter Notebook**:
```bash
jupyter notebook "CSGO Prediction.ipynb"
```

2. **Execute cells sequentially** to:
   - Download and preprocess the dataset
   - Perform feature selection and correlation analysis
   - Train and evaluate multiple models
   - Generate visualizations and performance metrics

## Key Features Analysis

### Equipment Impact
The analysis reveals that equipment significantly impacts round outcomes:
- **Armor and Helmets**: Strong positive correlation with winning
- **Defuse Kits**: Critical for CT success
- **AWP Sniper Rifles**: High-impact weapons for both teams

### Tactical Elements
- **Player Count**: Number of alive players strongly predicts success
- **Bomb Status**: Whether bomb is planted affects round dynamics
- **Grenades**: Flashbangs and smoke grenades provide tactical advantage

### Weapon Distribution
```python
# Key weapons identified in feature selection
weapons_analyzed = [
    'ct_weapon_ak47', 't_weapon_ak47',    # Assault rifles
    'ct_weapon_awp', 't_weapon_awp',      # Sniper rifles  
    'ct_weapon_m4a4',                     # CT-specific rifle
    'ct_weapon_sg553', 't_weapon_sg553',  # Scoped rifles
    'ct_weapon_usps'                      # CT pistol
]
```

## Model Evaluation

### Performance Metrics
| Model | Accuracy | Best Use Case |
|-------|----------|---------------|
| Random Forest | 83.1% | Best overall performance, handles feature interactions well |
| KNN (Optimized) | 78.8% | Good baseline, simple interpretation |
| Deep NN | 76.2% | Complex patterns but prone to overfitting |
| Shallow NN | 75.5% | Fastest training, reasonable performance |

### Feature Importance Insights
1. **Equipment** (armor, helmets, defuse kits) most predictive
2. **Team composition** (players alive) critical
3. **Weapon advantages** (AWP, AK47) significant
4. **Tactical items** (grenades) provide edge

## Technical Implementation

### Data Pipeline
```python
# Complete preprocessing pipeline
def preprocess_data():
    # 1. Download raw ARFF format data
    # 2. Parse attributes and convert to CSV
    # 3. Create binary target variable
    # 4. Select features based on correlation
    # 5. Scale features for ML algorithms
    # 6. Split train/validation/test sets
    return X_train_scaled, X_test_scaled, y_train, y_test
```

### Model Training Strategy
- **Cross-validation**: 3-fold CV for hyperparameter tuning
- **Early stopping**: Prevents neural network overfitting
- **Feature scaling**: StandardScaler for distance-based algorithms
- **Stratified sampling**: Maintains target class balance

## Results & Insights

### Key Findings
1. **Equipment superiority matters**: Teams with better equipment (armor, helmets) win more rounds
2. **Player count is crucial**: Even small player advantages translate to higher win probability
3. **Weapon meta is important**: Certain weapons (AWP, AK47) provide significant advantages
4. **Tactical utility usage**: Proper grenade usage correlates with round success

### Model Selection Recommendation
**Random Forest** emerges as the best model because:
- Highest accuracy (83.1%)
- Handles feature interactions naturally
- Robust to overfitting
- Provides feature importance insights
- No need for feature scaling

## Applications

### Potential Use Cases
1. **Esports Analytics**: Team performance analysis and strategy optimization
2. **Game Balance**: Identifying overpowered weapons or equipment
3. **Coaching Tools**: Data-driven tactical decision making
4. **Betting Odds**: Real-time round outcome prediction
5. **Player Development**: Understanding factors that lead to round wins

### Real-world Impact
This analysis demonstrates how machine learning can provide actionable insights in competitive gaming, helping teams make data-driven tactical decisions and improving overall gameplay strategies.

## Future Enhancements

### Potential Improvements
1. **Real-time Integration**: Live game state prediction
2. **Map-specific Models**: Different strategies per map
3. **Time Series Analysis**: Round progression patterns  
4. **Player-specific Features**: Individual skill ratings
5. **Ensemble Methods**: Combining multiple algorithms

### Advanced Features
- Economic state modeling (team money)
- Positional data analysis
- Communication pattern analysis
- Historical performance integration

## Technical Notes

- **Memory Usage**: ~18MB for full dataset
- **Processing Time**: ~5-10 minutes for complete analysis
- **Model Persistence**: Neural networks saved as .h5 files
- **Scalability**: Handles 100K+ records efficiently
- **Cross-platform**: Compatible with Windows/Linux/macOS