from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Datasæt som har været igennem databehandling
data = pd.read_csv("Alpha_dataset.csv")

# Load the trained Random Forest model
model_RF = joblib.load("stress_model_RF.pkl")
model_LR = joblib.load("stress_model_LR.pkl")
model_KNN = joblib.load("stress_model_KNN.pkl")

# Ændre i datasættet ved at droppe Stress Binary også sæt den til at classificere den efter
X = data.drop(columns=["Stress_Binary"])  # Features
y = data["Stress_Binary"]  # True labels

# Modeller sættes til at finde Stress_Binary i datasæt
predictions_RF = model_RF.predict(X)
predictions_LR = model_LR.predict(X)
predictions_KNN = model_KNN.predict(X)

# Kombiner alle modellerne, så deres predcitions bliver brugt til at finde Stress_Binary
combined_predictions = (predictions_RF + predictions_LR + predictions_KNN) >= 2

# Evaluere den kombinerede model
accuracy = accuracy_score(y, combined_predictions)
precision = precision_score(y, combined_predictions, average='binary')
recall = recall_score(y, combined_predictions, average='binary')
f1 = f1_score(y, combined_predictions, average='binary')
conf_matrix = confusion_matrix(y, combined_predictions)

# Print results
print("Combined Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Stressed", "Stressed"],
            yticklabels=["Not Stressed", "Stressed"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Combined Model')
plt.show()
