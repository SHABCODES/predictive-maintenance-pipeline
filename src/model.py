import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import CONFIG

def train_model(df):
    
    # Engine-wise split
    engine_ids = df['engine_id'].unique()
    split_idx = int(len(engine_ids) * (1 - CONFIG["test_split"]))
    
    train_ids = engine_ids[:split_idx]
    test_ids = engine_ids[split_idx:]
    
    train_df = df[df['engine_id'].isin(train_ids)]
    test_df = df[df['engine_id'].isin(test_ids)]
    
    X_train = train_df.drop(['engine_id','cycle','max_cycle','RUL','failure'], axis=1)
    y_train = train_df['failure']
    
    X_test = test_df.drop(['engine_id','cycle','max_cycle','RUL','failure'], axis=1)
    y_test = test_df['failure']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    logging.info("Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'predictive_maintenance_model.pkl')
    logging.info("Model saved as predictive_maintenance_model.pkl")
