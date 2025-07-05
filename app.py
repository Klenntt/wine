import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class WineQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wine Quality Classifier")
        self.data = None
        self.pipeline = None
        self.columns = []

        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.root, text="Load CSV", command=self.load_csv).grid(row=0, column=0, padx=10, pady=10)
        tk.Button(self.root, text="Show Stats", command=self.show_stats).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Train Model", command=self.train_model).grid(row=0, column=2, padx=10, pady=10)
        tk.Button(self.root, text="Predict", command=self.predict_quality).grid(row=0, column=3, padx=10, pady=10)

        self.entries_frame = tk.LabelFrame(self.root, text="Enter Feature Values")
        self.entries_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        self.data = pd.read_csv(file_path)

        if 'quality' not in self.data.columns:
            messagebox.showerror("Error", "CSV must contain a 'quality' column")
            return

        self.data['quality_class'] = self.data['quality'].apply(lambda x: 1 if x >= 7 else 0)
        self.columns = [col for col in self.data.columns if col not in ['quality', 'quality_class']]

        for widget in self.entries_frame.winfo_children():
            widget.destroy()

        self.entries = {}
        for idx, col in enumerate(self.columns):
            tk.Label(self.entries_frame, text=col).grid(row=idx, column=0, sticky="w")
            entry = tk.Entry(self.entries_frame)
            entry.grid(row=idx, column=1)
            self.entries[col] = entry

        messagebox.showinfo("Loaded", f"Data loaded with {self.data.shape[0]} rows.")

    def show_stats(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first.")
            return
        print("Missing values per column:\n", self.data.isnull().sum())

        plt.figure(figsize=(10, 6))
        sns.countplot(x='quality_class', data=self.data)
        plt.title('Wine Quality Distribution')
        plt.xlabel('Class (0 = Not Good, 1 = Good)')
        plt.ylabel('Count')
        plt.show()

        plt.figure(figsize=(12, 8))
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def train_model(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first.")
            return

        X = self.data[self.columns]
        y = self.data['quality_class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ])

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print(f"Accuracy: {acc:.2f}")
        messagebox.showinfo("Training Complete", f"Model trained with accuracy: {acc:.2f}")

        # Plot Feature Importance
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({'Feature': self.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.show()

        joblib.dump(self.pipeline, "wine_quality_model.pkl")

    def predict_quality(self):
        if self.pipeline is None:
            messagebox.showwarning("Warning", "Train the model first.")
            return

        try:
            input_data = [float(self.entries[col].get()) for col in self.columns]
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for all fields.")
            return

        prediction = self.pipeline.predict([input_data])[0]
        result = "Good Quality (1)" if prediction == 1 else "Not Good (0)"
        messagebox.showinfo("Prediction Result", f"The predicted wine quality is: {result}")

# Run the app
root = tk.Tk()
app = WineQualityApp(root)
root.mainloop()
