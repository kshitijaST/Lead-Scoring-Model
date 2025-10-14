import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MODEL_TYPE = 'random_forest'
    
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_MIN_SAMPLES_SPLIT = 20
    RF_MIN_SAMPLES_LEAF = 10
    
    SCORE_THRESHOLDS = {'hot': 80, 'warm': 60, 'cool': 40, 'cold': 0}

class DataSimulator:
    def __init__(self, random_state=Config.RANDOM_STATE):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_sample_data(self, n_samples=10000):
        print("📊 Generating sample lead data...")
        data = {
            'lead_id': range(1, n_samples + 1),
            'lead_source': np.random.choice(
                ['Website', 'Email', 'Social Media', 'Referral', 'Cold Call', 'Paid Search'], 
                n_samples, 
                p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
            ),
            'company_size': np.random.choice(
                ['Small', 'Medium', 'Large', 'Enterprise'], 
                n_samples, 
                p=[0.4, 0.3, 0.2, 0.1]
            ),
            'industry': np.random.choice(
                ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail'], 
                n_samples,
                p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
            ),
            'country': np.random.choice(
                ['USA', 'UK', 'Canada', 'Germany', 'Australia', 'Other'],
                n_samples,
                p=[0.5, 0.15, 0.1, 0.1, 0.05, 0.1]
            ),
            'website_visits': np.random.poisson(5, n_samples),
            'email_opens': np.random.poisson(3, n_samples),
            'time_on_site': np.random.exponential(300, n_samples),
            'pages_viewed': np.random.poisson(8, n_samples),
            'form_submissions': np.random.poisson(1, n_samples),
            'email_click_rate': np.random.beta(2, 5, n_samples),
            'days_since_created': np.random.exponential(30, n_samples),
            'has_demo_requested': np.random.binomial(1, 0.2, n_samples),
            'content_downloads': np.random.poisson(0.5, n_samples),
        }
        
        df = pd.DataFrame(data)
        df['converted'] = self._generate_target_variable(df)
        
        print(f"✅ Generated {len(df)} sample leads")
        print(f"📈 Conversion rate: {df['converted'].mean():.2%}")
        return df
    
    def _generate_target_variable(self, df):
        conversion_probability = (
            (df['website_visits'] * 0.1) +
            (df['email_opens'] * 0.05) +
            (df['pages_viewed'] * 0.08) +
            (df['form_submissions'] * 0.3) +
            (df['email_click_rate'] * 2) +
            (df['time_on_site'] * 0.001) +
            (df['has_demo_requested'] * 0.4) +
            (df['content_downloads'] * 0.2) -
            (df['days_since_created'] * 0.01)
        )
        
        industry_bonus = {
            'Technology': 0.3, 'Finance': 0.2, 'Healthcare': 0.1,
            'Manufacturing': 0.0, 'Education': -0.1, 'Retail': 0.0
        }
        conversion_probability += df['industry'].map(industry_bonus)
        
        source_bonus = {
            'Referral': 0.4, 'Website': 0.2, 'Paid Search': 0.1,
            'Email': 0.0, 'Social Media': -0.1, 'Cold Call': -0.2
        }
        conversion_probability += df['lead_source'].map(source_bonus)
        
        conversion_probability += np.random.normal(0, 0.3, len(df))
        return (conversion_probability > conversion_probability.median()).astype(int)
    
    def generate_new_leads(self, n_leads=10):
        df = self.generate_sample_data(n_leads)
        return df.drop('converted', axis=1)

class LeadScoringAI:
    def __init__(self, model_type=Config.MODEL_TYPE):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.data_simulator = DataSimulator()
    
    def preprocess_data(self, df, is_training=True):
        processed_df = df.copy()
        
        # Create engineered features
        processed_df['engagement_score'] = (
            processed_df['website_visits'] * 0.3 +
            processed_df['email_opens'] * 0.2 +
            processed_df['pages_viewed'] * 0.5 +
            processed_df['content_downloads'] * 0.5
        )
        
        processed_df['conversion_velocity'] = (
            processed_df['form_submissions'] / (processed_df['days_since_created'] + 1)
        )
        
        processed_df['digital_footprint'] = (
            processed_df['website_visits'] +
            processed_df['pages_viewed'] +
            processed_df['form_submissions'] * 3
        )
        
        # Handle categorical variables
        categorical_cols = ['lead_source', 'company_size', 'industry', 'country']
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
        
        if is_training:
            self.feature_columns = [col for col in processed_df.columns if col != 'converted']
        
        if not is_training and self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            processed_df = processed_df[self.feature_columns]
        
        return processed_df
    
    def plot_feature_importance(self, feature_names, top_n=15):
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = np.abs(self.model.coef_[0])
        
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance - Lead Scoring Model", fontsize=16, fontweight='bold')
        bars = plt.barh(range(min(top_n, len(indices))), 
                       importance[indices][:top_n][::-1],
                       color='skyblue', edgecolor='black')
        
        plt.yticks(range(min(top_n, len(indices))), 
                  [feature_names[i] for i in indices[:top_n]][::-1], fontsize=12)
        plt.xlabel('Importance Score', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_lead_score_distribution(self, scores):
        thresholds = Config.SCORE_THRESHOLDS
        
        plt.figure(figsize=(12, 6))
        colors = ['red', 'orange', 'lightblue', 'darkblue']
        bin_ranges = [thresholds['cold'], thresholds['cool'], thresholds['warm'], thresholds['hot'], 100]
        
        n, bins, patches = plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        
        for i, (patch, left_bin, right_bin) in enumerate(zip(patches, bins[:-1], bins[1:])):
            if right_bin <= thresholds['cool']:
                patch.set_facecolor(colors[3])
            elif right_bin <= thresholds['warm']:
                patch.set_facecolor(colors[2])
            elif right_bin <= thresholds['hot']:
                patch.set_facecolor(colors[1])
            else:
                patch.set_facecolor(colors[0])
        
        plt.title('Lead Score Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Lead Score', fontsize=12)
        plt.ylabel('Number of Leads', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        for threshold_name, threshold_value in thresholds.items():
            if threshold_value > 0:
                plt.axvline(x=threshold_value, color='black', linestyle='--', alpha=0.7)
                plt.text(threshold_value + 1, plt.ylim()[1] * 0.9, 
                        f'{threshold_name.title()}: {threshold_value}', 
                        rotation=90, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model_performance(self, y_true, y_pred, y_pred_proba):
        print("=" * 60)
        print("LEAD SCORING MODEL EVALUATION")
        print("=" * 60)
        
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Converted', 'Converted'],
                   yticklabels=['Not Converted', 'Converted'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return roc_auc
    
    def train(self, n_samples=10000, test_size=Config.TEST_SIZE):
        print("🚀 Training Lead Scoring Model...")
        
        df = self.data_simulator.generate_sample_data(n_samples)
        
        # FIXED: Preprocess data and separate features and target
        processed_data = self.preprocess_data(df)
        X = processed_data.drop('converted', axis=1)
        y = processed_data['converted']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=Config.RANDOM_STATE, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=Config.RF_N_ESTIMATORS,
                max_depth=Config.RF_MAX_DEPTH,
                min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
                random_state=Config.RANDOM_STATE
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        roc_auc = self.evaluate_model_performance(y_test, y_pred, y_pred_proba)
        
        if hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance(X.columns)
        
        print("✅ Model training completed!")
        return roc_auc
    
    def predict_scores(self, lead_data):
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
        
        if isinstance(lead_data, dict):
            lead_df = pd.DataFrame([lead_data])
        else:
            lead_df = lead_data.copy()
        
        lead_processed = self.preprocess_data(lead_df, is_training=False)
        lead_scaled = self.scaler.transform(lead_processed)
        conversion_probabilities = self.model.predict_proba(lead_scaled)[:, 1]
        lead_scores = (conversion_probabilities * 100).astype(int)
        
        return lead_scores
    
    def categorize_leads(self, scores):
        thresholds = Config.SCORE_THRESHOLDS
        categories = []
        for score in scores:
            if score >= thresholds['hot']:
                categories.append("Hot")
            elif score >= thresholds['warm']:
                categories.append("Warm")
            elif score >= thresholds['cool']:
                categories.append("Cool")
            else:
                categories.append("Cold")
        return categories
    
    def analyze_leads(self, lead_data):
        if isinstance(lead_data, dict):
            lead_data = pd.DataFrame([lead_data])
        
        scores = self.predict_scores(lead_data)
        categories = self.categorize_leads(scores)
        
        results = []
        for i, (score, category) in enumerate(zip(scores, categories)):
            result = {
                'lead_id': lead_data.iloc[i]['lead_id'] if 'lead_id' in lead_data.columns else i + 1,
                'score': score,
                'category': category,
                'priority': self._get_priority_level(category),
                'recommendation': self._get_recommendation(category, score)
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _get_priority_level(self, category):
        priority_map = {'Hot': 'High', 'Warm': 'Medium', 'Cool': 'Low', 'Cold': 'Very Low'}
        return priority_map.get(category, 'Unknown')
    
    def _get_recommendation(self, category, score):
        recommendations = {
            'Hot': f"🔥 Immediate follow-up! Call within 24 hours. Score: {score}",
            'Warm': f"✅ Schedule follow-up this week. Nurture with targeted content. Score: {score}",
            'Cool': f"📧 Add to email nurture sequence. Monitor engagement. Score: {score}",
            'Cold': f"❌ Low priority. Generic newsletter only. Re-engage later. Score: {score}"
        }
        return recommendations.get(category, "No recommendation available")
    
    def save_model(self, filepath='lead_scoring_model.pkl'):
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'config': {'thresholds': Config.SCORE_THRESHOLDS, 'model_type': Config.MODEL_TYPE}
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved successfully to: {filepath}")
    
    def load_model(self, filepath='lead_scoring_model.pkl'):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data['scaler']
        self.is_trained = True
        print(f"✅ Model loaded successfully from: {filepath}")

def main():
    print("🎯 LEAD SCORING AI - PREDICTIVE ANALYSIS")
    print("=" * 50)
    
    # Initialize and train model
    ai_model = LeadScoringAI()
    
    # Train the model
    print("\n1. TRAINING THE AI MODEL...")
    ai_model.train(n_samples=5000)
    
    # Create sample leads for prediction
    print("\n2. GENERATING SAMPLE LEADS FOR PREDICTION...")
    sample_leads = [
        {
            'lead_id': 1001,
            'lead_source': 'Website',
            'company_size': 'Medium',
            'industry': 'Technology',
            'country': 'USA',
            'website_visits': 15,
            'email_opens': 12,
            'time_on_site': 600,
            'pages_viewed': 20,
            'form_submissions': 3,
            'email_click_rate': 0.6,
            'days_since_created': 5,
            'has_demo_requested': 1,
            'content_downloads': 2
        },
        {
            'lead_id': 1002,
            'lead_source': 'Referral',
            'company_size': 'Enterprise',
            'industry': 'Finance',
            'country': 'USA',
            'website_visits': 8,
            'email_opens': 6,
            'time_on_site': 400,
            'pages_viewed': 12,
            'form_submissions': 2,
            'email_click_rate': 0.4,
            'days_since_created': 10,
            'has_demo_requested': 1,
            'content_downloads': 1
        },
        {
            'lead_id': 1003,
            'lead_source': 'Cold Call',
            'company_size': 'Small',
            'industry': 'Retail',
            'country': 'Other',
            'website_visits': 2,
            'email_opens': 1,
            'time_on_site': 60,
            'pages_viewed': 3,
            'form_submissions': 0,
            'email_click_rate': 0.1,
            'days_since_created': 45,
            'has_demo_requested': 0,
            'content_downloads': 0
        },
        {
            'lead_id': 1004,
            'lead_source': 'Email',
            'company_size': 'Large',
            'industry': 'Healthcare',
            'country': 'UK',
            'website_visits': 6,
            'email_opens': 8,
            'time_on_site': 250,
            'pages_viewed': 8,
            'form_submissions': 1,
            'email_click_rate': 0.3,
            'days_since_created': 15,
            'has_demo_requested': 0,
            'content_downloads': 1
        }
    ]
    
    # Analyze the leads
    print("\n3. ANALYZING LEADS...")
    results = ai_model.analyze_leads(sample_leads)
    
    # Display results
    print("\n4. LEAD SCORING RESULTS:")
    print("=" * 80)
    for _, result in results.iterrows():
        print(f"Lead ID: {result['lead_id']}")
        print(f"  Score: {result['score']}/100")
        print(f"  Category: {result['category']}")
        print(f"  Priority: {result['priority']}")
        print(f"  Action: {result['recommendation']}")
        print("-" * 50)
    
    # Show score distribution
    print("\n5. VISUALIZING LEAD SCORES...")
    all_scores = ai_model.predict_scores(sample_leads)
    ai_model.plot_lead_score_distribution(all_scores)
    
    # Save the model
    print("\n6. SAVING THE MODEL...")
    ai_model.save_model('lead_scoring_model.pkl')
    
    # Summary statistics
    print("\n7. SUMMARY:")
    print("=" * 30)
    category_counts = results['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category} Leads: {count}")
    
    avg_score = results['score'].mean()
    print(f"  Average Score: {avg_score:.1f}/100")
    
    print("\n✅ LEAD SCORING AI COMPLETED SUCCESSFULLY!")

def demo_new_leads():
    """Demo function to show how to score new leads"""
    print("\n" + "="*60)
    print("DEMO: SCORING NEW LEADS")
    print("="*60)
    
    # Load the trained model
    ai_model = LeadScoringAI()
    ai_model.load_model('lead_scoring_model.pkl')
    
    # New lead data
    new_lead = {
        'lead_id': 2001,
        'lead_source': 'Website',
        'company_size': 'Medium', 
        'industry': 'Technology',
        'country': 'USA',
        'website_visits': 10,
        'email_opens': 8,
        'time_on_site': 450,
        'pages_viewed': 15,
        'form_submissions': 2,
        'email_click_rate': 0.5,
        'days_since_created': 3,
        'has_demo_requested': 1,
        'content_downloads': 2
    }
    
    # Score the new lead
    result = ai_model.analyze_leads(new_lead)
    
    print("NEW LEAD ANALYSIS:")
    print(f"Lead ID: {result.iloc[0]['lead_id']}")
    print(f"Score: {result.iloc[0]['score']}/100")
    print(f"Category: {result.iloc[0]['category']}") 
    print(f"Priority: {result.iloc[0]['priority']}")
    print(f"Recommendation: {result.iloc[0]['recommendation']}")

if __name__ == "__main__":
    # Run the main example
    main()
    
    # Demo scoring new leads
    demo_new_leads()