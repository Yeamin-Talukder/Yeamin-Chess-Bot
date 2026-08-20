import type { PositionFeatures, CandidateFeatures } from './FeatureExtractor';

export class YeaminModel {
    private isReady: boolean = false;
    private trees: any[] = [];
    private baselinePrediction: number = 0;

    // Feature ordering strictly aligned with pipeline.py num_cols + cat_cols
    private numCols = [
        'move_number', 'w_pawns', 'b_pawns', 'w_knights', 'b_knights',
        'w_bishops', 'b_bishops', 'w_rooks', 'b_rooks', 'w_queens', 'b_queens',
        'w_material', 'b_material', 'material_balance', 'legal_move_count',
        'stockfish_rank', 'candidate_eval', 'eval_drop', 'is_mate',
        'cand_is_capture', 'cand_is_castling', 'cand_is_promotion',
        'cand_is_check', 'cand_is_pawn_move',
        // cat_cols
        'game_phase', 'side_to_move', 'time_control', 'eco', 'opening', 'cand_moving_piece'
    ];

    public async initialize(modelPath: string = '/models/yeamin_style_model.json') {
        try {
            const response = await fetch(modelPath);
            const data = await response.json();
            this.trees = data.trees;
            this.baselinePrediction = data.baseline_prediction;
            this.isReady = true;
            console.log("Yeamin Style Model (JSON) loaded successfully.");
        } catch (e) {
            console.error("Failed to load Yeamin Style Model:", e);
            throw e;
        }
    }

    public isInitialized(): boolean {
        return this.isReady;
    }

    public async predict(position: PositionFeatures, candidate: CandidateFeatures, stockfishRank: number, candidateEval: number, bestEval: number): Promise<number> {
        if (!this.isReady) throw new Error("Model not initialized");

        // 1. Build combined feature row
        const row: Record<string, any> = {
            ...position,
            ...candidate,
            stockfish_rank: stockfishRank,
            candidate_eval: candidateEval,
            eval_drop: bestEval - candidateEval,
            is_mate: candidateEval > 9000 || candidateEval < -9000 ? 1 : 0,
        };

        // 2. Format into Array
        const input = this.numCols.map(col => {
            const val = row[col];
            return typeof val === 'number' ? val : 0; // Simple imputation
        });

        // 3. Evaluate trees
        let score = this.baselinePrediction;
        
        for (const tree of this.trees) {
            let node = tree[0]; // Start at root
            while (!node.is_leaf) {
                if (input[node.feature_idx] <= node.num_threshold) {
                    node = tree[node.left];
                } else {
                    node = tree[node.right];
                }
            }
            score += node.value; // Accumulate
        }
        
        // HistGradientBoosting outputs log-odds for binary classification
        return 1.0 / (1.0 + Math.exp(-score));
    }
}
