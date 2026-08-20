import type { StockfishService } from '../engine/StockfishService';
import type { YeaminModel } from '../ml/ModelLoader';
import { FeatureExtractor } from '../ml/FeatureExtractor';

export interface BotSettings {
    styleStrength: number; // 0.0 to 1.0
    maxCplDrop: number;    // e.g. 100
    depth: number;
    multiPV: number;
    skillLevel?: number;   // 0 to 20
}

export interface BotDecision {
    bestMoveUci: string;
    engineEval: number;
    styleScore: number;
    finalScore: number;
    candidates: any[];
}

export class YeaminBot {
    private engine: StockfishService;
    private model: YeaminModel;
    
    constructor(engine: StockfishService, model: YeaminModel) {
        this.engine = engine;
        this.model = model;
    }

    public async predictMove(fen: string, settings: BotSettings): Promise<BotDecision> {
        // 1. Get engine candidates
        const candidates = await this.engine.analyze({
            fen,
            depth: settings.depth,
            multiPV: settings.multiPV,
            skillLevel: settings.skillLevel
        });

        if (candidates.length === 0) {
            throw new Error("No candidates returned from Stockfish");
        }

        const bestEval = candidates[0].evaluation;
        const positionFeatures = FeatureExtractor.extractPositionFeatures(fen);
        
        const scoredCandidates = [];

        // 2. Score candidates with ML model
        for (const cand of candidates) {
            const candFeatures = FeatureExtractor.extractCandidateFeatures(fen, cand.move);
            
            // Calculate CPL drop (positive value)
            const cplDrop = bestEval - cand.evaluation;
            
            let styleProb = 0;
            if (cplDrop <= settings.maxCplDrop) {
                // Only invoke ML if move is acceptable
                styleProb = await this.model.predict(
                    positionFeatures,
                    candFeatures,
                    cand.rank,
                    cand.evaluation,
                    bestEval
                );
            }
            
            // --- Hardcoded Opening Repertoire Bonus ---
            const fenParts = fen.split(' ');
            const fullMove = parseInt(fenParts[5] || '1', 10);
            const sideToMove = fenParts[1]; // 'w' or 'b'
            
            let isRepertoireMove = false;
            if (fullMove <= 5) {
                if (sideToMove === 'w' && ['e2e4', 'f2f4', 'g1f3'].includes(cand.move)) {
                    isRepertoireMove = true;
                } else if (sideToMove === 'b' && ['g7g6', 'd7d6', 'g8f6'].includes(cand.move)) {
                    isRepertoireMove = true;
                }
            }

            // Phase 5 scoring formula: 
            // Score = (1 - strength) * EngineRankScore + strength * StyleProb
            // EngineRankScore usually normalizes rank 1 to 1.0, rank 2 to something lower...
            // Or simpler: sorting criteria.
            const engineScore = 1.0 / cand.rank; // Simple reciprocal
            let finalScore = (1.0 - settings.styleStrength) * engineScore + (settings.styleStrength * styleProb);

            // Apply massive bonus if it's a preferred repertoire move
            if (isRepertoireMove) {
                finalScore += 1000;
            }

            scoredCandidates.push({
                ...cand,
                cplDrop,
                styleProb,
                engineScore,
                finalScore,
                isAcceptable: (cplDrop <= settings.maxCplDrop) || isRepertoireMove // Always allow repertoire moves
            });
        }

        // 3. Sort by final score
        scoredCandidates.sort((a, b) => b.finalScore - a.finalScore);
        
        // Ensure at least one acceptable move, else fallback to engine best
        const bestCandidate = scoredCandidates.find(c => c.isAcceptable) || scoredCandidates[0];

        return {
            bestMoveUci: bestCandidate.move,
            engineEval: bestCandidate.evaluation,
            styleScore: bestCandidate.styleProb,
            finalScore: bestCandidate.finalScore,
            candidates: scoredCandidates
        };
    }
}
