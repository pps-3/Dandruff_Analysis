"""
DANDRUFF ANALYSIS - MODEL TRAINING & COMPARISON
This script trains multiple models and provides detailed comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
import warnings
import os
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# Import core classes
from main import DandruffDataset, FeatureEngineer, ClusterAnalyzer, RecommendationEngine

class ModelComparison:
    """Compare multiple ML models for dandruff severity prediction"""
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
        # Scale data
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
    def initialize_models(self):
        """Initialize all models to compare"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear'
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Naive Bayes': GaussianNB()
        }
        
        print(f"✓ Initialized {len(self.models)} models for comparison")
        
    def train_all_models(self):
        """Train all models and collect metrics"""
        print("\n" + "="*80)
        print("🤖 TRAINING ALL MODELS")
        print("="*80)
        
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='f1_macro')
            
            # Calculate metrics
            metrics = {
                'model': model,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision_macro': precision_score(self.y_test, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(self.y_test, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(self.y_test, y_pred, average='macro', zero_division=0),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            self.results[name] = metrics
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision_macro']:.4f}")
            print(f"  Recall:    {metrics['recall_macro']:.4f}")
            print(f"  F1-Score:  {metrics['f1_macro']:.4f}")
            print(f"  CV F1:     {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        print("\n✓ All models trained successfully!")
        
    def get_comparison_table(self):
        """Generate comparison table"""
        comparison_data = []
        
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision_macro'],
                'Recall': metrics['recall_macro'],
                'F1-Score': metrics['f1_macro'],
                'CV F1-Score': metrics['cv_mean'],
                'CV Std': metrics['cv_std']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        return df_comparison
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON TABLE")
        print("="*80)
        
        df = self.get_comparison_table()
        print(df.to_string(index=False))
        
        print("\n" + "="*80)
        print("🏆 BEST MODEL RANKING (by F1-Score)")
        print("="*80)
        
        for idx, row in df.iterrows():
            rank = df.index.get_loc(idx) + 1
            print(f"{rank}. {row['Model']:<25} F1-Score: {row['F1-Score']:.4f}")
    
    def plot_model_comparison(self, save_path='outputs/model_comparison.png'):
        """Visualize model comparison"""
        df = self.get_comparison_table()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. F1-Score comparison
        ax1 = axes[0, 0]
        colors = sns.color_palette("viridis", len(df))
        ax1.barh(df['Model'], df['F1-Score'], color=colors)
        ax1.set_xlabel('F1-Score', fontsize=12)
        ax1.set_title('Model Performance - F1-Score', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        for i, v in enumerate(df['F1-Score']):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # 2. Multiple metrics comparison
        ax2 = axes[0, 1]
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            ax2.bar(x + i*width, df[metric], width, label=metric)
        
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Cross-validation scores with error bars
        ax3 = axes[1, 0]
        ax3.barh(df['Model'], df['CV F1-Score'], xerr=df['CV Std'], 
                color='skyblue', alpha=0.7, capsize=5)
        ax3.set_xlabel('CV F1-Score', fontsize=12)
        ax3.set_title('Cross-Validation Performance (5-Fold)', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        
        # 4. Accuracy vs F1-Score scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['Accuracy'], df['F1-Score'], 
                            s=200, c=range(len(df)), cmap='viridis', alpha=0.7)
        
        for idx, row in df.iterrows():
            ax4.annotate(row['Model'], (row['Accuracy'], row['F1-Score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Accuracy', fontsize=12)
        ax4.set_ylabel('F1-Score', fontsize=12)
        ax4.set_title('Accuracy vs F1-Score', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Model comparison plot saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, save_path='outputs/confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        labels = ['Low', 'Medium', 'High']
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, metrics['predictions'], labels=labels)
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title(f'{name}\nF1: {metrics["f1_macro"]:.3f}', fontweight='bold')
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved: {save_path}")
        plt.close()
    
    def plot_classification_reports(self, save_path='outputs/classification_reports.png'):
        """Visualize detailed classification reports"""
        n_models = len(self.results)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        labels = ['Low', 'Medium', 'High']
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            report = classification_report(self.y_test, metrics['predictions'], 
                                         target_names=labels, output_dict=True)
            
            # Extract metrics for each class
            data = []
            for label in labels:
                data.append([
                    report[label]['precision'],
                    report[label]['recall'],
                    report[label]['f1-score']
                ])
            
            data = np.array(data).T
            
            ax = axes[idx]
            x = np.arange(len(labels))
            width = 0.25
            
            ax.bar(x - width, data[0], width, label='Precision', alpha=0.8)
            ax.bar(x, data[1], width, label='Recall', alpha=0.8)
            ax.bar(x + width, data[2], width, label='F1-Score', alpha=0.8)
            
            ax.set_xlabel('Severity Class')
            ax.set_ylabel('Score')
            ax.set_title(f'{name} - Per-Class Performance', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Classification reports saved: {save_path}")
        plt.close()
    
    def get_best_model(self):
        """Get the best performing model"""
        best_name = max(self.results.items(), key=lambda x: x[1]['f1_macro'])[0]
        best_model = self.results[best_name]['model']
        best_f1 = self.results[best_name]['f1_macro']
        
        print(f"\n🏆 Best Model: {best_name}")
        print(f"   F1-Score: {best_f1:.4f}")
        
        return best_name, best_model
    
    def save_best_model(self, path='saved_models'):
        """Save the best model"""
        best_name, best_model = self.get_best_model()
        
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, 'random_forest_model.pkl')
        joblib.dump(best_model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(path, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'best_model': best_name,
            'f1_score': self.results[best_name]['f1_macro'],
            'accuracy': self.results[best_name]['accuracy'],
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_names': self.feature_names
        }
        
        metadata_path = os.path.join(path, 'model_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        print(f"\n✓ Best model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
        print(f"✓ Metadata saved: {metadata_path}")
        
        return model_path


def run_complete_training(filepath='Dandruff_odatatset - Form Responses 1 (1).csv'):
    """Run complete training pipeline with model comparison"""
    
    print("\n" + "="*80)
    print("🚀 DANDRUFF ANALYSIS - COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    # 1. Load and prepare data
    print("\n📊 STEP 1: Loading and Preparing Data")
    print("-" * 80)
    
    dataset = DandruffDataset(filepath)
    df = dataset.load_data()
    
    if df is None:
        print("❌ Failed to load data")
        return None
    
    dataset.clean_column_names()
    dataset.detect_target()
    dataset.handle_missing_values()
    
    # 2. Feature Engineering
    print("\n🔧 STEP 2: Feature Engineering")
    print("-" * 80)
    
    engineer = FeatureEngineer(df)
    df = engineer.engineer_all_features()
    
    # 3. Prepare features
    print("\n📋 STEP 3: Preparing Features")
    print("-" * 80)
    
    feature_cols = ['HairCareIndex', 'DietQualityScore', 'StressProxy', 
                   'SleepScore', 'age', 'gender_encoded']
    
    X = df[feature_cols].values
    y = df[dataset.target_column].values
    y_binned = pd.cut(y, bins=[0, 2, 3, 6], labels=['Low', 'Medium', 'High'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binned, test_size=0.2, random_state=42, stratify=y_binned
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"✓ Features: {len(feature_cols)}")
    
    # 4. Model Comparison
    print("\n🤖 STEP 4: Training and Comparing Models")
    print("-" * 80)
    
    comparator = ModelComparison(X_train, X_test, y_train, y_test, feature_cols)
    comparator.initialize_models()
    comparator.train_all_models()
    
    # 5. Results and Visualizations
    print("\n📊 STEP 5: Generating Results and Visualizations")
    print("-" * 80)
    
    comparator.print_comparison_table()
    comparator.plot_model_comparison()
    comparator.plot_confusion_matrices()
    comparator.plot_classification_reports()
    
    # 6. Save best model
    print("\n💾 STEP 6: Saving Best Model")
    print("-" * 80)
    
    model_path = comparator.save_best_model()
    
    # 7. Clustering Analysis
    print("\n🎯 STEP 7: Clustering Analysis")
    print("-" * 80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clusterer = ClusterAnalyzer(n_clusters=4)
    cluster_labels = clusterer.fit_clusters(X_train_scaled)
    clusterer.elbow_analysis(X_train_scaled)
    profiles = clusterer.get_cluster_profiles(X_train_scaled, cluster_labels, feature_cols)
    
    # 8. Summary Statistics
    print("\n" + "="*80)
    print("📈 TRAINING SUMMARY")
    print("="*80)
    
    comparison_df = comparator.get_comparison_table()
    
    print(f"\n✓ Models Trained: {len(comparator.models)}")
    print(f"✓ Best Model: {comparator.get_best_model()[0]}")
    print(f"✓ Best F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
    print(f"✓ Best Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\n📁 Check 'outputs/' folder for visualizations")
    print("💾 Check 'saved_models/' folder for trained models")
    
    return {
        'comparator': comparator,
        'clusterer': clusterer,
        'comparison_table': comparison_df,
        'best_model': comparator.get_best_model()[0]
    }


if __name__ == "__main__":
    # Run complete training pipeline
    results = run_complete_training()
    
    if results:
        print("\n🎉 Ready for deployment!")
        print("\nNext steps:")
        print("1. Review model comparison in 'outputs/' folder")
        print("2. Run 'python app.py' to start the web application")