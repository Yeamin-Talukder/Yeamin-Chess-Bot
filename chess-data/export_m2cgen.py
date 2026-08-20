import joblib
import m2cgen as m2c

def export_model():
    model_path = "data/phase4/models/yeamin_style_model.pkl"
    print("Loading Phase 5 Model...")
    model = joblib.load(model_path)
    
    # Extract classifier
    classifier = model.steps[-1][1]
    
    print("Exporting to JavaScript using m2cgen...")
    try:
        code = m2c.export_to_javascript(classifier)
        
        # Add ES module export
        code = "export " + code
        
        with open("../yeamin-bot-web/src/ml/yeamin_model_raw.js", "w") as f:
            f.write(code)
            
        print("Successfully exported model to yeamin_model_raw.js")
    except Exception as e:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        print(f"m2cgen Export Error: {repr(e)}")

if __name__ == '__main__':
    export_model()
