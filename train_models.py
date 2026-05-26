import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# 1. Load Data
df = pd.read_csv('asthma_disease_data.csv')

# 2. Data Preparation
# Drop administrative/irrelevant columns
X = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
y = df['Diagnosis']

# Train/Test Split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Crucial for SVM and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize Models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=6, random_state=42),
    'Support Vector Machine': SVC(class_weight='balanced', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
}

# 4. Train and Evaluate Models
results = []
trained_models = {}

print("Training models...")
for name, model in models.items():
    # Use scaled features for Logistic Regression and SVM, unscaled for trees (but scaling is fine for all)
    # Let's use scaled features for consistency
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })
    
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 5. Display Results Table
df_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("\n" + "="*50)
print("             MODEL COMPARISON RESULTS             ")
print("="*50)
print(df_results.to_string(index=False))
print("="*50)

# Save results to csv
df_results.to_csv('model_comparison_results.csv', index=False)

# 6. Feature Importances for the Best Model (Random Forest or Gradient Boosting)
# Let's plot and save feature importance for Random Forest (often the best model with class weights)
rf_model = trained_models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns
rf_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importances.values[:10], y=rf_importances.index[:10], palette='viridis')
plt.title('Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 10 Important Features in Random Forest:")
print(rf_importances.head(10).to_string())

# Also evaluate Gradient Boosting feature importances
gb_model = trained_models['Gradient Boosting']
gb_importances = pd.Series(gb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 10 Important Features in Gradient Boosting:")
print(gb_importances.head(10).to_string())
