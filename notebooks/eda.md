# EDA Checklist

## 1. Load Data
df = pd.read_csv("data/spam.csv")
## 2. Basic Checks
- head()
- shape
- info()
- columns
- nulls

## 3. Visualizations
- spam vs ham countplot
- message length distributions
- histograms of numeric features
- word clouds (optional)

## 4. Text analysis
- top bigrams
- most frequent spam keywords