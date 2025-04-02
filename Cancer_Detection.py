import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class CancerDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_ranges = None

    def load_and_preprocess(self, filepath):
        """Load data and establish valid ranges"""
        df = pd.read_csv('E:\Cancer Detection\data\Cancer_Data.csv')
        work_df = df.copy().drop(['id', 'Unnamed: 32'], axis=1)
        work_df['diagnosis'] = work_df['diagnosis'].replace({"B": 0, "M": 1})
        
        # Calculate realistic ranges from data
        X = work_df.drop('diagnosis', axis=1)
        self.feature_ranges = {
            col: {
                'min': float(X[col].min()),
                'max': float(X[col].max()),
                'mean': float(X[col].mean())
            } 
            for col in X.columns
        }
        
        return work_df

    def train_model(self, work_df):
        """Train model with validated parameters"""
        X = work_df.drop('diagnosis', axis=1)
        y = work_df['diagnosis']
        self.feature_columns = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Updated with optimal parameters
        self.model = SVC(
            random_state=42,
            class_weight={0: 1, 1: 2},  # Adjusted class weights
            probability=True,
            C=0.5,  # More conservative regularization
            gamma='scale',
            kernel='rbf'
        )
        self.model.fit(X_scaled, y)
        return self

    def validate_input(self, features):
        """Check if inputs are within observed ranges"""
        errors = {}
        for col in self.feature_columns:
            val = features.get(col, 0)
            min_val = self.feature_ranges[col]['min']
            max_val = self.feature_ranges[col]['max']
            
            if not (min_val <= val <= max_val):
                errors[col] = {
                    'value': val,
                    'valid_min': min_val,
                    'valid_max': max_val
                }
        return errors

    def predict(self, features_dict):
        """Safe prediction with validation"""
        # Input validation
        errors = self.validate_input(features_dict)
        if errors:
            raise ValueError(f"Invalid inputs: {errors}")
        
        # Prepare input
        input_df = pd.DataFrame([features_dict], columns=self.feature_columns)
        input_scaled = self.scaler.transform(input_df)
        
        # Predict with calibrated probabilities
        prediction = self.model.predict(input_scaled)[0]
        proba = self.model.predict_proba(input_scaled)[0]
        
        diagnosis = "Cancer detected" if prediction == 1 else "Cancer is not detected"
        confidence = proba[1] if prediction == 1 else proba[0]
        
        return diagnosis, float(confidence)

# Singleton instance
detector = CancerDetector()