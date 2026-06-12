import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import urllib.request

# Matplotlib embedding in Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Custom modules
from src.preprocess import train_test_split, StandardScaler, MinMaxScaler
from src.knn import KNNClassifier
from src.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class KNNDashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("K-Nearest Neighbors Scratch Classifier & Heart Disease Diagnosis")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # State variables
        self.dataset_path = "heart_disease.csv"
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        
        # Theme configuration (Dark Slate / Tech theme)
        self.bg_color = "#111827"       # Slate 900
        self.panel_color = "#1f2937"    # Slate 800
        self.accent_color = "#3b82f6"   # Blue 500
        self.text_color = "#f9fafb"     # Slate 50
        self.muted_text = "#9ca3af"     # Slate 400
        
        self.configure(bg=self.bg_color)
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure styles
        self.style.configure(".", background=self.bg_color, foreground=self.text_color)
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("Panel.TFrame", background=self.panel_color)
        
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 10))
        self.style.configure("Panel.TLabel", background=self.panel_color, foreground=self.text_color, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background=self.panel_color, foreground=self.text_color, font=("Segoe UI", 12, "bold"))
        self.style.configure("Title.TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 16, "bold"))
        
        self.style.configure("TButton", background=self.accent_color, foreground=self.text_color, font=("Segoe UI", 10, "bold"), borderwidth=0)
        self.style.map("TButton", background=[("active", "#2563eb"), ("disabled", "#4b5563")])
        
        self.style.configure("Secondary.TButton", background="#4b5563", foreground=self.text_color, font=("Segoe UI", 10), borderwidth=0)
        self.style.map("Secondary.TButton", background=[("active", "#374151")])
        
        self.style.configure("TRadiobutton", background=self.panel_color, foreground=self.text_color, font=("Segoe UI", 10))
        self.style.configure("TCombobox", fieldbackground="#374151", background=self.panel_color, foreground=self.text_color)
        
        # Build UI layout
        self.create_widgets()
        
        # Download and load data by default if possible
        self.after(500, self.auto_load_data)

    def create_widgets(self):
        # 1. Top Bar
        top_bar = ttk.Frame(self)
        top_bar.pack(fill=tk.X, padx=15, pady=10)
        
        title_lbl = ttk.Label(top_bar, text="KNN CLASS CLASSIFIER & DIAGNOSTIC SYSTEM", style="Title.TLabel")
        title_lbl.pack(side=tk.LEFT)
        
        self.status_lbl = ttk.Label(top_bar, text="Status: Ready", style="TLabel", foreground=self.accent_color)
        self.status_lbl.pack(side=tk.RIGHT, pady=5)
        
        # Main Body Splitter
        main_pane = ttk.Frame(self)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        # Three Columns: 1. Control Panel (Left), 2. Visualizations (Middle), 3. Patient Diagnosis (Right)
        
        # --- Column 1: Control Panel ---
        ctrl_frame = ttk.Frame(main_pane, style="Panel.TFrame", width=280)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), pady=10)
        ctrl_frame.pack_propagate(False)
        
        # Section Header
        lbl = ttk.Label(ctrl_frame, text="Model Settings", style="Header.TLabel")
        lbl.pack(anchor=tk.W, padx=15, pady=15)
        
        # Divider
        self.draw_divider(ctrl_frame)
        
        # K value Selector
        ttk.Label(ctrl_frame, text="Number of Neighbors (K):", style="Panel.TLabel").pack(anchor=tk.W, padx=15, pady=(10, 5))
        self.k_val = tk.IntVar(value=5)
        k_spin = ttk.Spinbox(ctrl_frame, from_=1, to=50, textvariable=self.k_val, width=10, font=("Segoe UI", 10))
        k_spin.pack(anchor=tk.W, padx=15, pady=5)
        
        # Distance Metric Selector
        ttk.Label(ctrl_frame, text="Distance Metric:", style="Panel.TLabel").pack(anchor=tk.W, padx=15, pady=(10, 5))
        self.metric_val = tk.StringVar(value="Euclidean")
        metric_combo = ttk.Combobox(ctrl_frame, textvariable=self.metric_val, values=["Euclidean", "Manhattan", "Minkowski"], state="readonly", width=18)
        metric_combo.pack(anchor=tk.W, padx=15, pady=5)
        
        # Scaling Selector
        ttk.Label(ctrl_frame, text="Feature Scaling:", style="Panel.TLabel").pack(anchor=tk.W, padx=15, pady=(10, 5))
        self.scaling_val = tk.StringVar(value="StandardScaler")
        r1 = ttk.Radiobutton(ctrl_frame, text="Standardization (Z-score)", variable=self.scaling_val, value="StandardScaler", style="TRadiobutton")
        r1.pack(anchor=tk.W, padx=15, pady=2)
        r2 = ttk.Radiobutton(ctrl_frame, text="Min-Max Normalization", variable=self.scaling_val, value="MinMaxScaler", style="TRadiobutton")
        r2.pack(anchor=tk.W, padx=15, pady=2)
        r3 = ttk.Radiobutton(ctrl_frame, text="No Scaling (Raw)", variable=self.scaling_val, value="None", style="TRadiobutton")
        r3.pack(anchor=tk.W, padx=15, pady=2)
        
        # Split ratio slider
        ttk.Label(ctrl_frame, text="Test Set Ratio:", style="Panel.TLabel").pack(anchor=tk.W, padx=15, pady=(15, 5))
        self.test_ratio = tk.DoubleVar(value=0.2)
        ratio_scale = ttk.Scale(ctrl_frame, from_=0.1, to=0.5, variable=self.test_ratio, orient=tk.HORIZONTAL)
        ratio_scale.pack(fill=tk.X, padx=15, pady=2)
        self.ratio_lbl = ttk.Label(ctrl_frame, text="Test size: 20%", style="Panel.TLabel", foreground=self.muted_text)
        self.ratio_lbl.pack(anchor=tk.W, padx=15)
        ratio_scale.bind("<Motion>", lambda e: self.ratio_lbl.configure(text=f"Test size: {int(self.test_ratio.get()*100)}%"))
        
        self.draw_divider(ctrl_frame)
        
        # Action Buttons
        self.btn_load = ttk.Button(ctrl_frame, text="Reload Dataset", command=self.load_dataset_action, style="Secondary.TButton")
        self.btn_load.pack(fill=tk.X, padx=15, pady=(15, 5))
        
        self.btn_train = ttk.Button(ctrl_frame, text="Train & Evaluate", command=self.train_and_evaluate, style="TButton")
        self.btn_train.pack(fill=tk.X, padx=15, pady=5)
        
        self.btn_tune = ttk.Button(ctrl_frame, text="Run Hyperparameter Tuning", command=self.run_tuning_plot, style="Secondary.TButton")
        self.btn_tune.pack(fill=tk.X, padx=15, pady=5)

        # --- Column 2: Dashboard Visualizations (Middle) ---
        vis_frame = ttk.Frame(main_pane, style="Panel.TFrame")
        vis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Upper panel for stats, lower panel for plots
        self.stats_lbl_var = tk.StringVar(value="Load and Train model to view statistics...")
        stats_box = ttk.Label(vis_frame, textvariable=self.stats_lbl_var, style="Panel.TLabel", font=("Consolas", 9), anchor=tk.W, justify=tk.LEFT)
        stats_box.pack(fill=tk.X, padx=15, pady=15)
        
        self.plot_panel = ttk.Frame(vis_frame)
        self.plot_panel.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.fig, self.ax = plt.subplots(figsize=(5, 4), facecolor="#1f2937")
        self.ax.set_facecolor("#1f2937")
        self.ax.tick_params(colors="#f9fafb")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Column 3: Patient Diagnosis Panel (Right) ---
        diag_frame = ttk.Frame(main_pane, style="Panel.TFrame", width=320)
        diag_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0), pady=10)
        diag_frame.pack_propagate(False)
        
        ttk.Label(diag_frame, text="Interactive Patient Diagnosis", style="Header.TLabel").pack(anchor=tk.W, padx=15, pady=15)
        self.draw_divider(diag_frame)
        
        # Form Container (Scrollable)
        form_canvas = tk.Canvas(diag_frame, bg=self.panel_color, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(diag_frame, orient="vertical", command=form_canvas.yview)
        scrollable_frame = ttk.Frame(form_canvas, style="Panel.TFrame")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: form_canvas.configure(scrollregion=form_canvas.bbox("all"))
        )
        form_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=290)
        form_canvas.configure(yscrollcommand=scrollbar.set)
        
        form_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Form Fields
        # 1. Age
        ttk.Label(scrollable_frame, text="Age (years):", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.ent_age = ttk.Spinbox(scrollable_frame, from_=1, to=120, width=15)
        self.ent_age.set(50)
        self.ent_age.pack(anchor=tk.W, pady=2)
        
        # 2. Sex
        ttk.Label(scrollable_frame, text="Sex:", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.cb_sex = ttk.Combobox(scrollable_frame, values=["Male", "Female"], state="readonly", width=15)
        self.cb_sex.set("Male")
        self.cb_sex.pack(anchor=tk.W, pady=2)
        
        # 3. Chest Pain Type (cp)
        ttk.Label(scrollable_frame, text="Chest Pain Type (CP):", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.cb_cp = ttk.Combobox(scrollable_frame, values=["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"], state="readonly", width=25)
        self.cb_cp.set("0: Typical Angina")
        self.cb_cp.pack(anchor=tk.W, pady=2)
        
        # 4. Resting Blood Pressure (trestbps)
        ttk.Label(scrollable_frame, text="Resting BP (mm Hg):", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.ent_bps = ttk.Spinbox(scrollable_frame, from_=80, to=220, width=15)
        self.ent_bps.set(130)
        self.ent_bps.pack(anchor=tk.W, pady=2)
        
        # 5. Serum Cholesterol (chol)
        ttk.Label(scrollable_frame, text="Serum Cholesterol (mg/dl):", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.ent_chol = ttk.Spinbox(scrollable_frame, from_=100, to=600, width=15)
        self.ent_chol.set(240)
        self.ent_chol.pack(anchor=tk.W, pady=2)
        
        # 6. Fasting Blood Sugar (fbs)
        ttk.Label(scrollable_frame, text="Fasting Blood Sugar > 120 mg/dl:", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.cb_fbs = ttk.Combobox(scrollable_frame, values=["No (<= 120)", "Yes (> 120)"], state="readonly", width=15)
        self.cb_fbs.set("No (<= 120)")
        self.cb_fbs.pack(anchor=tk.W, pady=2)
        
        # 7. Resting ECG (restecg)
        ttk.Label(scrollable_frame, text="Resting ECG:", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.cb_ecg = ttk.Combobox(scrollable_frame, values=["0: Normal", "1: ST-T Wave Abnormality", "2: LV Hypertrophy"], state="readonly", width=25)
        self.cb_ecg.set("0: Normal")
        self.cb_ecg.pack(anchor=tk.W, pady=2)
        
        # 8. Max Heart Rate (thalach)
        ttk.Label(scrollable_frame, text="Max Heart Rate Achieved:", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.ent_hr = ttk.Spinbox(scrollable_frame, from_=50, to=220, width=15)
        self.ent_hr.set(150)
        self.ent_hr.pack(anchor=tk.W, pady=2)
        
        # 9. Exercise Induced Angina (exang)
        ttk.Label(scrollable_frame, text="Exercise Induced Angina:", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.cb_exang = ttk.Combobox(scrollable_frame, values=["No", "Yes"], state="readonly", width=15)
        self.cb_exang.set("No")
        self.cb_exang.pack(anchor=tk.W, pady=2)
        
        # 10. ST Depression (oldpeak)
        ttk.Label(scrollable_frame, text="ST Depression (oldpeak):", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.ent_oldpeak = ttk.Spinbox(scrollable_frame, from_=0.0, to=10.0, increment=0.1, width=15)
        self.ent_oldpeak.set(1.0)
        self.ent_oldpeak.pack(anchor=tk.W, pady=2)
        
        # 11. Slope of peak ST segment (slope)
        ttk.Label(scrollable_frame, text="Slope of ST Segment:", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.cb_slope = ttk.Combobox(scrollable_frame, values=["0: Upsloping", "1: Flat", "2: Downsloping"], state="readonly", width=25)
        self.cb_slope.set("1: Flat")
        self.cb_slope.pack(anchor=tk.W, pady=2)
        
        # 12. Vessels colored by fluoroscopy (ca)
        ttk.Label(scrollable_frame, text="Major Vessels (ca):", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.cb_ca = ttk.Combobox(scrollable_frame, values=["0", "1", "2", "3", "4"], state="readonly", width=15)
        self.cb_ca.set("0")
        self.cb_ca.pack(anchor=tk.W, pady=2)
        
        # 13. Thalassemia (thal)
        ttk.Label(scrollable_frame, text="Thalassemia (thal):", style="Panel.TLabel").pack(anchor=tk.W, pady=(5, 2))
        self.cb_thal = ttk.Combobox(scrollable_frame, values=["0: Null", "1: Normal", "2: Fixed Defect", "3: Reversible Defect"], state="readonly", width=25)
        self.cb_thal.set("2: Fixed Defect")
        self.cb_thal.pack(anchor=tk.W, pady=2)
        
        self.draw_divider(scrollable_frame)
        
        # Diagnosis Actions
        self.btn_predict = ttk.Button(scrollable_frame, text="Run Heart Diagnosis", command=self.predict_diagnosis, style="TButton")
        self.btn_predict.pack(fill=tk.X, pady=10)
        
        self.diag_res_lbl = ttk.Label(scrollable_frame, text="Run training first...", style="Header.TLabel", anchor=tk.CENTER, justify=tk.CENTER)
        self.diag_res_lbl.pack(fill=tk.X, pady=10)

    def draw_divider(self, parent):
        sep = ttk.Separator(parent, orient="horizontal")
        sep.pack(fill=tk.X, padx=15, pady=10)

    def auto_load_data(self):
        """Attempts to load the dataset automatically at launch."""
        self.status_lbl.configure(text="Status: Loading dataset...", foreground="#fbbf24")
        self.update()
        
        if not os.path.exists(self.dataset_path):
            url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
            try:
                urllib.request.urlretrieve(url, self.dataset_path)
            except Exception as e:
                self.status_lbl.configure(text="Status: Download Failed", foreground="#ef4444")
                messagebox.showerror("Error", f"Failed to download dataset: {e}")
                return
                
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.status_lbl.configure(text="Status: Dataset Loaded", foreground="#10b981")
            
            # Simple dataset overview
            n_samples = self.df.shape[0]
            n_features = self.df.shape[1] - 1
            n_pos = np.sum(self.df['target'] == 1)
            n_neg = np.sum(self.df['target'] == 0)
            
            stats_text = (
                f"Heart Disease Dataset:\n"
                f"• Samples: {n_samples}\n"
                f"• Features: {n_features}\n"
                f"• Heart Disease cases: {n_pos} ({n_pos/n_samples:.1%})\n"
                f"• Normal cases: {n_neg} ({n_neg/n_samples:.1%})\n\n"
                f"Click 'Train & Evaluate' to fit the KNN model."
            )
            self.stats_lbl_var.set(stats_text)
            self.train_and_evaluate()  # Train initial model
        except Exception as e:
            self.status_lbl.configure(text="Status: Load Failed", foreground="#ef4444")
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def load_dataset_action(self):
        # Force re-download / reload
        if os.path.exists(self.dataset_path):
            os.remove(self.dataset_path)
        self.auto_load_data()

    def train_and_evaluate(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No dataset loaded.")
            return
            
        self.status_lbl.configure(text="Status: Fitting model...", foreground="#fbbf24")
        self.update()
        
        # Get settings
        k = self.k_val.get()
        metric = self.metric_val.get().lower()
        scaling = self.scaling_val.get()
        ratio = self.test_ratio.get()
        
        # Split Data
        X = self.df.drop(columns=['target']).values
        y = self.df['target'].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        
        # Preprocessing Scaler
        if scaling == "StandardScaler":
            self.scaler = StandardScaler()
            X_tr = self.scaler.fit_transform(self.X_train)
            X_te = self.scaler.transform(self.X_test)
        elif scaling == "MinMaxScaler":
            self.scaler = MinMaxScaler()
            X_tr = self.scaler.fit_transform(self.X_train)
            X_te = self.scaler.transform(self.X_test)
        else:
            self.scaler = None
            X_tr = self.X_train
            X_te = self.X_test
            
        # Fit custom model
        self.model = KNNClassifier(k=k, metric=metric)
        self.model.fit(X_tr, self.y_train)
        
        # Predict & Evaluate
        train_preds = self.model.predict(X_tr)
        test_preds = self.model.predict(X_te)
        
        train_acc = accuracy_score(self.y_train, train_preds)
        test_acc = accuracy_score(self.y_test, test_preds)
        prec = precision_score(self.y_test, test_preds)
        rec = recall_score(self.y_test, test_preds)
        f1 = f1_score(self.y_test, test_preds)
        cm = confusion_matrix(self.y_test, test_preds)
        
        # Compare with sklearn
        from sklearn.neighbors import KNeighborsClassifier
        sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        sklearn_knn.fit(X_tr, self.y_train)
        sklearn_preds = sklearn_knn.predict(X_te)
        sklearn_acc = accuracy_score(self.y_test, sklearn_preds)
        
        # Build Stats Text
        stats_text = (
            f"=== MODEL PERFORMANCE ===\n"
            f"• Train Accuracy:      {train_acc:.2%}\n"
            f"• Test Accuracy:       {test_acc:.2%}\n"
            f"• Precision (Disease): {prec:.2%}\n"
            f"• Recall (Disease):    {rec:.2%}\n"
            f"• F1-Score:            {f1:.2%}\n\n"
            f"=== COMPARATIVE VALIDATION ===\n"
            f"• Sklearn Accuracy:    {sklearn_acc:.2%}\n"
            f"• Prediction Mismatch: {np.sum(test_preds != sklearn_preds)} samples"
        )
        self.stats_lbl_var.set(stats_text)
        
        # Plot Confusion Matrix
        self.plot_confusion_matrix(cm)
        self.status_lbl.configure(text="Status: Model Trained", foreground="#10b981")
        self.diag_res_lbl.configure(text="Ready to Diagnose Patient", foreground=self.text_color)

    def plot_confusion_matrix(self, cm):
        self.ax.clear()
        self.fig.patch.set_facecolor("#1f2937")
        self.ax.set_facecolor("#1f2937")
        
        im = self.ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        self.ax.set_title("Test Set Confusion Matrix", color="#f9fafb", fontsize=12, fontweight='bold', pad=10)
        
        classes = ['Normal', 'Disease']
        tick_marks = np.arange(len(classes))
        self.ax.set_xticks(tick_marks)
        self.ax.set_xticklabels(classes, color="#f9fafb")
        self.ax.set_yticks(tick_marks)
        self.ax.set_yticklabels(classes, color="#f9fafb")
        
        # Add labels
        self.ax.set_ylabel('True Label', color="#f9fafb", fontsize=10)
        self.ax.set_xlabel('Predicted Label', color="#f9fafb", fontsize=10)
        
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             verticalalignment="center",
                             color="white" if cm[i, j] > thresh else "black",
                             fontsize=12, fontweight='bold')
                             
        self.fig.tight_layout()
        self.canvas.draw()

    def run_tuning_plot(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No dataset loaded.")
            return
            
        self.status_lbl.configure(text="Status: Tuning hyperparameters...", foreground="#fbbf24")
        self.update()
        
        metric = self.metric_val.get().lower()
        scaling = self.scaling_val.get()
        ratio = self.test_ratio.get()
        
        # Split Data
        X = self.df.drop(columns=['target']).values
        y = self.df['target'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        
        if scaling == "StandardScaler":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        elif scaling == "MinMaxScaler":
            scaler = MinMaxScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr = X_train
            X_te = X_test
            
        # Grid Search
        k_values = list(range(1, 26))
        train_accs = []
        test_accs = []
        
        for k in k_values:
            knn = KNNClassifier(k=k, metric=metric)
            knn.fit(X_tr, y_train)
            train_accs.append(accuracy_score(y_train, knn.predict(X_tr)))
            test_accs.append(accuracy_score(y_test, knn.predict(X_te)))
            
        # Plot
        self.ax.clear()
        self.fig.patch.set_facecolor("#1f2937")
        self.ax.set_facecolor("#1f2937")
        
        self.ax.plot(k_values, train_accs, label='Train Accuracy', marker='o', color='#3b82f6', linewidth=2)
        self.ax.plot(k_values, test_accs, label='Test Accuracy', marker='s', color='#ef4444', linewidth=2)
        
        self.ax.set_title(f'Tuning: K vs Accuracy ({metric.capitalize()})', color="#f9fafb", fontsize=12, fontweight='bold', pad=10)
        self.ax.set_xlabel('K (Number of Neighbors)', color="#f9fafb")
        self.ax.set_ylabel('Accuracy', color="#f9fafb")
        self.ax.set_xticks(range(1, 26, 2))
        self.ax.tick_params(colors="#f9fafb")
        self.ax.grid(True, color="#374151", linestyle='--')
        
        # Legend with dark style
        leg = self.ax.legend(facecolor="#1f2937", edgecolor="#374151")
        for text in leg.get_texts():
            text.set_color("#f9fafb")
            
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_lbl.configure(text="Status: Tuning Plots Ready", foreground="#10b981")

    def predict_diagnosis(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Train the model first.")
            return
            
        try:
            # Parse entries
            age = float(self.ent_age.get())
            
            sex_str = self.cb_sex.get()
            sex = 1.0 if sex_str == "Male" else 0.0
            
            cp_str = self.cb_cp.get()
            cp = float(cp_str.split(":")[0])
            
            bps = float(self.ent_bps.get())
            chol = float(self.ent_chol.get())
            
            fbs_str = self.cb_fbs.get()
            fbs = 1.0 if "Yes" in fbs_str else 0.0
            
            ecg_str = self.cb_ecg.get()
            ecg = float(ecg_str.split(":")[0])
            
            hr = float(self.ent_hr.get())
            
            exang_str = self.cb_exang.get()
            exang = 1.0 if exang_str == "Yes" else 0.0
            
            oldpeak = float(self.ent_oldpeak.get())
            
            slope_str = self.cb_slope.get()
            slope = float(slope_str.split(":")[0])
            
            ca = float(self.cb_ca.get())
            
            thal_str = self.cb_thal.get()
            thal = float(thal_str.split(":")[0])
            
            # Pack features in order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
            features = np.array([[age, sex, cp, bps, chol, fbs, ecg, hr, exang, oldpeak, slope, ca, thal]])
            
            # Apply same scaling
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
                
            # Predict
            pred = self.model.predict(features_scaled)[0]
            probas = self.model.predict_proba(features_scaled)[0]
            
            # Display result
            # Index of target classes: self.model.classes_ contains [0, 1] usually
            # Class 0: Normal, Class 1: Disease
            disease_prob = probas[1] if len(probas) > 1 else (probas[0] if pred == 1 else 0.0)
            
            if pred == 1:
                self.diag_res_lbl.configure(
                    text=f"DIAGNOSIS: DISEASE RISK\nProbability: {disease_prob:.1%}\n(Consult a physician)",
                    foreground="#f87171" # light red
                )
            else:
                self.diag_res_lbl.configure(
                    text=f"DIAGNOSIS: NORMAL HEALTH\nDisease Risk: {disease_prob:.1%}\n(Healthy Profile)",
                    foreground="#34d399" # light green
                )
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input values: {e}")

if __name__ == "__main__":
    app = KNNDashboardApp()
    app.mainloop()
