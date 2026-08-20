import joblib
import json
import numpy as np

def export_model():
    model_path = "data/phase4/models/yeamin_style_model.pkl"
    print("Loading Phase 5 Model...")
    model = joblib.load(model_path)
    
    classifier = model.steps[-1][1]
    
    # HistGradientBoostingClassifier stores trees in _predictors
    # shape: (n_iters, n_classes) or (n_iters, 1) for binary
    predictors = classifier._predictors
    
    trees_data = []
    
    for i in range(len(predictors)):
        tree = predictors[i][0] # binary classification
        nodes = tree.nodes
        
        # nodes is a structured array. fields: value, feature_idx, num_threshold, has_missing_sl,
        # left, right, is_leaf
        
        node_list = []
        for n in nodes:
            is_leaf = bool(n['is_leaf'])
            node_list.append({
                'is_leaf': is_leaf,
                'value': float(n['value']),
                'feature_idx': int(n['feature_idx']),
                'num_threshold': float(n['num_threshold']),
                'left': int(n['left']),
                'right': int(n['right'])
            })
            
        trees_data.append(node_list)
        
    export_dict = {
        'baseline_prediction': float(classifier._baseline_prediction[0, 0]),
        'trees': trees_data
    }
    
    with open("../yeamin-bot-web/public/models/yeamin_style_model.json", "w") as f:
        json.dump(export_dict, f)
        
    print("Successfully exported trees to yeamin_style_model.json")

if __name__ == '__main__':
    export_model()
