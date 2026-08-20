import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import numpy as np

def export_model():
    model_path = "data/phase4/models/yeamin_style_model.pkl"
    model = joblib.load(model_path)
    
    # We need to define the exact input schema.
    # From pipeline.py, we have 24 num_cols and 6 cat_cols.
    num_cols = [
        'move_number', 'w_pawns', 'b_pawns', 'w_knights', 'b_knights',
        'w_bishops', 'b_bishops', 'w_rooks', 'b_rooks', 'w_queens', 'b_queens',
        'w_material', 'b_material', 'material_balance', 'legal_move_count', 
        'stockfish_rank', 'candidate_eval', 'eval_drop', 'is_mate',
        'cand_is_capture', 'cand_is_castling', 'cand_is_promotion', 
        'cand_is_check', 'cand_is_pawn_move'
    ]
    
    cat_cols = [
        'side_to_move', 'time_control', 'eco', 'opening', 'game_phase', 'cand_moving_piece'
    ]
    
    # However, the scikit-learn pipeline expects a pandas DataFrame where columns have specific names!
    # skl2onnx has an option to convert a pipeline that expects a DataFrame by providing initial_types.
    # Wait, skl2onnx supports initial_types as a list of tuples (column_name, TensorType).
    initial_types = []
    for c in num_cols:
        initial_types.append((c, FloatTensorType([None, 1])))
    for c in cat_cols:
        initial_types.append((c, StringTensorType([None, 1])))
        
    print("Converting model to ONNX...")
    # There are known issues with HistGradientBoosting and skl2onnx depending on version.
    # Let's see if it works.
    onx = convert_sklearn(model, initial_types=initial_types, target_opset=12)
    with open("yeamin_style_model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    print("Successfully exported to yeamin_style_model.onnx")

if __name__ == '__main__':
    export_model()
