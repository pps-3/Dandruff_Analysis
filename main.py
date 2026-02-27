import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os
import joblib

# Try importing SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')


class DandruffDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.target_column = None
        
    def load_data(self):
        """Load CSV dataset"""
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"✓ Data loaded: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def clean_column_names(self):
        """Sanitize column names"""
        self.data.columns = (self.data.columns
                             .str.strip()
                             .str.lower()
                             .str.replace(' ', '_')
                             .str.replace('?', '')
                             .str.replace('(', '')
                             .str.replace(')', '')
                             .str.replace('.', ''))
        print(f"✓ Columns cleaned: {len(self.data.columns)} features")
    
    def detect_target(self):
        """Auto-detect severity column"""
        for col in self.data.columns:
            if 'severe' in col.lower() and 'dandruff' in col.lower():
                self.target_column = col
                print(f"✓ Target detected: {col}")
                return col
        return None
    
    def handle_missing_values(self):
        """Handle missing data"""
        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            num_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[num_cols] = self.data[num_cols].fillna(
                self.data[num_cols].median()
            )
            cat_cols = self.data.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if len(self.data[col].mode()) > 0:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            print(f"✓ Missing values handled: {missing_count} cells")
        else:
            print("✓ No missing values found")


class FeatureEngineer:
    def __init__(self, dataframe):
        self.df = dataframe
        self.label_encoders = {}
        
    def encode_categorical_features(self):
        """Encode all categorical variables"""
        if 'gender' in self.df.columns:
            le_gender = LabelEncoder()
            self.df['gender_encoded'] = le_gender.fit_transform(self.df['gender'])
            self.label_encoders['gender'] = le_gender
        
        print(f"✓ Categorical features encoded")
        
    def create_hair_care_index(self):
        """Composite score for hair care habits"""
        oil_col = [col for col in self.df.columns if 'apply_oil' in col][0]
        
        oil_score = self.df[oil_col].apply(lambda x: 
            1.0 if 'regularly' in str(x).lower() and 'times' in str(x).lower() else
            0.5 if 'occasionally' in str(x).lower() else
            0.25 if 'rarely' in str(x).lower() else
            0.0 if 'never' in str(x).lower() else 0.5
        )
        
        conditioner_col = [col for col in self.df.columns if 'conditioner' in col][0]
        
        conditioner_score = self.df[conditioner_col].apply(lambda x:
            1.0 if 'always' in str(x).lower() else
            0.5 if 'sometimes' in str(x).lower() else
            0.3 if 'rarely' in str(x).lower() else
            0.0 if 'never' in str(x).lower() else 0.5
        )
        
        hair_care_index = (oil_score + conditioner_score) / 2
        
        print(f"✓ HairCareIndex created (mean: {hair_care_index.mean():.3f})")
        return hair_care_index
    
    def create_diet_quality_score(self):
        """Composite score for dietary habits"""
        water_col = [col for col in self.df.columns if 'water' in col and 'drink' in col][0]
        
        water_score = self.df[water_col].apply(lambda x:
            0.2 if 'less than 1' in str(x).lower() else
            0.5 if '1' in str(x) and '2' in str(x) else
            0.8 if '2' in str(x) and '3' in str(x) else
            1.0 if 'more than 3' in str(x).lower() else 0.5
        )
        
        diet_col = [col for col in self.df.columns if 'describe_your_diet' in col][0]
        
        diet_score = self.df[diet_col].apply(lambda x:
            1.0 if 'balanced' in str(x).lower() else
            0.4 if 'irregular' in str(x).lower() else
            0.3 if 'oily' in str(x).lower() or 'spicy' in str(x).lower() else
            0.1 if 'processed' in str(x).lower() or 'junk' in str(x).lower() else 0.4
        )
        
        diet_quality = (water_score + diet_score) / 2
        
        print(f"✓ DietQualityScore created (mean: {diet_quality.mean():.3f})")
        return diet_quality
    
    def create_stress_proxy(self):
        """Create stress score"""
        stress_col = [col for col in self.df.columns if 'stress' in col and 'face' in col][0]
        
        stress_score = self.df[stress_col].apply(lambda x:
            1.0 if 'high' in str(x).lower() else
            0.6 if 'moderate' in str(x).lower() else
            0.3 if 'minimal' in str(x).lower() else
            0.0 if 'no stress' in str(x).lower() else 0.5
        )
        
        print(f"✓ StressProxy created (mean: {stress_score.mean():.3f})")
        return stress_score
    
    def create_sleep_score(self):
        """Create sleep quality score"""
        sleep_col = [col for col in self.df.columns if 'sleep' in col][0]
        
        sleep_score = self.df[sleep_col].apply(lambda x:
            1.0 if 'more than 8' in str(x).lower() else
            0.9 if '7' in str(x) and '8' in str(x) else
            0.6 if '5' in str(x) and '7' in str(x) else
            0.3 if 'less than 5' in str(x).lower() else 0.6
        )
        
        print(f"✓ SleepScore created (mean: {sleep_score.mean():.3f})")
        return sleep_score
    
    def engineer_all_features(self):
        """Create all engineered features"""
        self.encode_categorical_features()
        self.df['HairCareIndex'] = self.create_hair_care_index()
        self.df['DietQualityScore'] = self.create_diet_quality_score()
        self.df['StressProxy'] = self.create_stress_proxy()
        self.df['SleepScore'] = self.create_sleep_score()
        
        print(f"✓ All features engineered: {self.df.shape}")
        return self.df


class MLClassifier:
    def __init__(self):
        self.rf_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_rf_params = None
        self.best_lr_params = None
        self.feature_names = None
        
    def prepare_data(self, X, y, feature_names, test_size=0.2):
        """Split and scale data"""
        self.feature_names = feature_names
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")
    
    def tune_random_forest(self):
        """Hyperparameter tuning for Random Forest"""
        print("\n--- Tuning Random Forest Hyperparameters ---")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        }
        
        rf_base = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_rf_params = grid_search.best_params_
        
        print(f"\n✓ Best Random Forest Parameters:")
        for param, value in self.best_rf_params.items():
            print(f"  {param}: {value}")
        print(f"  Best CV F1-Score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_, grid_search.best_score_
    
    def tune_logistic_regression(self):
        """Hyperparameter tuning for Logistic Regression"""
        print("\n--- Tuning Logistic Regression Hyperparameters ---")
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced'],
            'max_iter': [1000]
        }
        
        lr_base = LogisticRegression(random_state=42, solver='liblinear')
        
        grid_search = GridSearchCV(
            estimator=lr_base,
            param_grid=param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_lr_params = grid_search.best_params_
        
        print(f"\n✓ Best Logistic Regression Parameters:")
        for param, value in self.best_lr_params.items():
            print(f"  {param}: {value}")
        print(f"  Best CV F1-Score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_, grid_search.best_score_
    
    def train_random_forest(self, use_tuning=True):
        """Train Random Forest classifier"""
        if use_tuning:
            self.rf_model, rf_score = self.tune_random_forest()
            print(f"✓ Random Forest trained with tuned hyperparameters")
        else:
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            self.rf_model.fit(self.X_train, self.y_train)
            
            cv_scores = cross_val_score(
                self.rf_model, self.X_train, self.y_train, 
                cv=5, scoring='f1_macro'
            )
            rf_score = cv_scores.mean()
            
            print(f"✓ Random Forest trained")
            print(f"  F1-Score (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return rf_score
    
    def train_logistic_regression(self, use_tuning=True):
        """Train Logistic Regression classifier"""
        if use_tuning:
            self.lr_model, lr_score = self.tune_logistic_regression()
            print(f"✓ Logistic Regression trained with tuned hyperparameters")
        else:
            self.lr_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear'
            )
            
            self.lr_model.fit(self.X_train, self.y_train)
            
            cv_scores = cross_val_score(
                self.lr_model, self.X_train, self.y_train,
                cv=5, scoring='f1_macro'
            )
            lr_score = cv_scores.mean()
            
            print(f"✓ Logistic Regression trained")
            print(f"  F1-Score (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return lr_score
    
    def evaluate_models(self):
        """Evaluate both models on test set"""
        results = {}
        
        rf_pred = self.rf_model.predict(self.X_test)
        print("\n=== RANDOM FOREST ===")
        print(classification_report(self.y_test, rf_pred, 
                                   target_names=['Low', 'Medium', 'High']))
        results['rf'] = classification_report(self.y_test, rf_pred, output_dict=True)
        
        lr_pred = self.lr_model.predict(self.X_test)
        print("\n=== LOGISTIC REGRESSION ===")
        print(classification_report(self.y_test, lr_pred,
                                   target_names=['Low', 'Medium', 'High']))
        results['lr'] = classification_report(self.y_test, lr_pred, output_dict=True)
        
        return results
    
    def get_feature_importance(self):
        """Get Random Forest feature importance"""
        importances = self.rf_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n=== TOP FEATURES ===")
        print(feature_importance)
        
        return feature_importance
    
    def explain_with_shap(self, sample_idx=0, class_idx=0):
        """SHAP explainability (optional)"""
        if not SHAP_AVAILABLE:
            print("⚠️  SHAP not available. Install with: pip install shap")
            return None
        
        print("\n=== SHAP EXPLAINABILITY ===")
        
        model = self.rf_model if self.rf_model is not None else self.lr_model
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)
        
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[class_idx][sample_idx],
                base_values=explainer.expected_value[class_idx],
                data=self.X_test[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        return shap_values
    
    def save_model(self, path="saved_models"):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        
        if self.rf_model is not None:
            joblib.dump(self.rf_model, os.path.join(path, "random_forest_model.pkl"))
            print("✓ Random Forest model saved")
        
        if self.lr_model is not None:
            joblib.dump(self.lr_model, os.path.join(path, "logistic_regression_model.pkl"))
            print("✓ Logistic Regression model saved")
        
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        print("✓ Scaler saved")


class ClusterAnalyzer:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = None
        self.silhouette = None
        
    def fit_clusters(self, X):
        """Fit K-Means clustering"""
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )
        
        cluster_labels = self.model.fit_predict(X)
        self.silhouette = silhouette_score(X, cluster_labels)
        
        print(f"✓ K-Means fitted (k={self.n_clusters})")
        print(f"  Silhouette Score: {self.silhouette:.3f}")
        
        return cluster_labels
    
    def elbow_analysis(self, X, k_range=range(2, 7)):
        """Perform elbow method analysis"""
        inertias = []
        silhouettes = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
        
        print("\n=== ELBOW ANALYSIS ===")
        for k, inertia, sil in zip(k_range, inertias, silhouettes):
            print(f"k={k}: Inertia={inertia:.2f}, Silhouette={sil:.3f}")
        
        return inertias, silhouettes
    
    def get_cluster_profiles(self, X, cluster_labels, feature_names):
        """Analyze cluster characteristics"""
        df = pd.DataFrame(X, columns=feature_names)
        df['Cluster'] = cluster_labels
        
        print("\n=== CLUSTER PROFILES ===")
        for cluster in range(self.n_clusters):
            cluster_data = df[df['Cluster'] == cluster]
            print(f"\nCluster {cluster} (n={len(cluster_data)}):")
            print(cluster_data.describe().loc['mean'])
        
        return df.groupby('Cluster').mean()


class RecommendationEngine:
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self):
        """Define recommendation rules"""
        return {
            'High': {
                'hair_care': [
                    "Switch to medicated anti-dandruff shampoo (ketoconazole 2%)",
                    "Apply coconut oil 2-3 times per week",
                    "Avoid harsh chemical treatments"
                ],
                'diet': [
                    "Increase water intake to 2-3 liters daily",
                    "Consume omega-3 rich foods (fish, walnuts)",
                    "Add zinc supplements (consult doctor)"
                ],
                'lifestyle': [
                    "Practice stress management (meditation, yoga)",
                    "Get 7-8 hours of sleep",
                    "Reduce scalp scratching"
                ]
            },
            'Medium': {
                'hair_care': [
                    "Use mild anti-dandruff shampoo 2-3 times weekly",
                    "Apply oil regularly before washing",
                    "Rinse hair thoroughly after shampooing"
                ],
                'diet': [
                    "Maintain adequate hydration (1.5-2L water)",
                    "Include fresh fruits and vegetables",
                    "Limit processed foods"
                ],
                'lifestyle': [
                    "Manage stress levels",
                    "Maintain regular sleep schedule",
                    "Avoid excessive heat styling"
                ]
            },
            'Low': {
                'hair_care': [
                    "Continue current hair care routine",
                    "Use gentle shampoo as needed",
                    "Regular oiling recommended"
                ],
                'diet': [
                    "Maintain balanced diet",
                    "Stay hydrated",
                    "Continue healthy eating habits"
                ],
                'lifestyle': [
                    "Keep up good habits",
                    "Monitor scalp condition",
                    "Avoid excessive stress"
                ]
            }
        }
    
    def generate_recommendations(self, severity, cluster_id):
        """Generate personalized recommendations"""
        base_recs = self.rules[severity]
        
        if cluster_id == 0:
            priority = "URGENT: Consult dermatologist if condition worsens"
        elif cluster_id == 1:
            priority = "FOCUS: Improve dietary habits and hydration"
        elif cluster_id == 2:
            priority = "MAINTAIN: Continue preventive care"
        else:
            priority = "FOCUS: Stress management critical"
        
        recommendations = {
            'priority': priority,
            'hair_care': base_recs['hair_care'],
            'diet': base_recs['diet'],
            'lifestyle': base_recs['lifestyle']
        }
        
        return recommendations