import { createStockfishWorker } from './StockfishWorker';

export interface AnalysisOptions {
    fen: string;
    multiPV?: number;
    depth?: number;
    moveTime?: number;
    skillLevel?: number;
}

export interface CandidateMove {
    move: string;
    evaluation: number;
    depth: number;
    rank: number;
}

export class StockfishService {
    private worker: Worker;
    private resolveReady!: () => void;
    private readyPromise: Promise<void>;
    private analysisResolve: ((candidates: CandidateMove[]) => void) | null = null;
    
    private currentCandidates: Map<string, CandidateMove> = new Map();
    private multiPV: number = 1;
    private isAnalyzing: boolean = false;

    constructor() {
        this.worker = createStockfishWorker();
        this.readyPromise = new Promise((resolve) => {
            this.resolveReady = resolve;
        });

        this.worker.onmessage = this.handleMessage.bind(this);
        this.worker.postMessage('uci');
    }

    public async waitForReady(): Promise<void> {
        return this.readyPromise;
    }

    private handleMessage(event: MessageEvent) {
        const line: string = event.data;
        // console.log("SF ->", line);

        if (line === 'uciok') {
            this.worker.postMessage('isready');
        } else if (line === 'readyok') {
            this.resolveReady();
        } else if (line.startsWith('info depth') && this.isAnalyzing) {
            this.parseInfo(line);
        } else if (line.startsWith('bestmove') && this.isAnalyzing) {
            this.isAnalyzing = false;
            if (this.analysisResolve) {
                // sort candidates by rank
                const results = Array.from(this.currentCandidates.values()).sort((a, b) => a.rank - b.rank);
                this.analysisResolve(results);
                this.analysisResolve = null;
            }
        }
    }

    private parseInfo(line: string) {
        // e.g., info depth 10 seldepth 14 multipv 1 score cp 12 nodes 15124 nps 756200 time 20 pv e2e4 e7e5
        const depthMatch = line.match(/depth (\d+)/);
        const multiPvMatch = line.match(/multipv (\d+)/);
        const scoreCpMatch = line.match(/score cp (-?\d+)/);
        const scoreMateMatch = line.match(/score mate (-?\d+)/);
        const pvMatch = line.match(/ pv ([a-h1-8qrbn]+)/);

        if (depthMatch && multiPvMatch && pvMatch && (scoreCpMatch || scoreMateMatch)) {
            const depth = parseInt(depthMatch[1], 10);
            const rank = parseInt(multiPvMatch[1], 10);
            const move = pvMatch[1];
            
            let evaluation = 0;
            if (scoreMateMatch) {
                const mateIn = parseInt(scoreMateMatch[1], 10);
                evaluation = mateIn > 0 ? 10000 - mateIn : -10000 - mateIn;
            } else if (scoreCpMatch) {
                evaluation = parseInt(scoreCpMatch[1], 10);
            }

            this.currentCandidates.set(move, { move, evaluation, depth, rank });
        }
    }

    public async analyze(options: AnalysisOptions): Promise<CandidateMove[]> {
        await this.waitForReady();

        return new Promise((resolve) => {
            this.analysisResolve = resolve;
            this.isAnalyzing = true;
            this.currentCandidates.clear();
            
            this.multiPV = options.multiPV || 1;
            this.worker.postMessage(`setoption name MultiPV value ${this.multiPV}`);
            
            if (options.skillLevel !== undefined) {
                this.worker.postMessage(`setoption name Skill Level value ${options.skillLevel}`);
            } else {
                this.worker.postMessage(`setoption name Skill Level value 20`); // default max
            }
            
            this.worker.postMessage(`position fen ${options.fen}`);
            
            let goCommand = 'go';
            if (options.depth) goCommand += ` depth ${options.depth}`;
            if (options.moveTime) goCommand += ` movetime ${options.moveTime}`;
            
            if (!options.depth && !options.moveTime) {
                goCommand += ' depth 15'; // fallback
            }

            this.worker.postMessage(goCommand);
        });
    }

    public stop() {
        if (this.isAnalyzing) {
            this.worker.postMessage('stop');
        }
    }

    public quit() {
        this.worker.postMessage('quit');
        this.worker.terminate();
    }
}
