import { useState, useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import { StockfishService } from '../../engine/StockfishService';
import { YeaminModel } from '../../ml/ModelLoader';
import { YeaminBot } from '../../bot/YeaminBot';

interface LogEntry {
    type: 'info' | 'user' | 'bot' | 'error' | 'system';
    text: string;
}

export function TextTestPage() {
    const [logs, setLogs] = useState<LogEntry[]>([
        { type: 'system', text: '=== Yeamin Bot Text Tester ===' },
        { type: 'system', text: 'Initializing Stockfish + ML model...' },
    ]);
    const [input, setInput] = useState('');
    const [ready, setReady] = useState(false);
    const [loading, setLoading] = useState(false);

    const chessRef = useRef(new Chess());
    const botRef = useRef<YeaminBot | null>(null);
    const logEndRef = useRef<HTMLDivElement>(null);

    const addLog = (type: LogEntry['type'], text: string) => {
        setLogs(prev => [...prev, { type, text }]);
    };

    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    useEffect(() => {
        async function init() {
            try {
                addLog('info', 'Starting Stockfish worker...');
                const engine = new StockfishService();
                await engine.waitForReady();
                addLog('info', '✅ Stockfish ready.');

                addLog('info', 'Loading Yeamin ML model...');
                const model = new YeaminModel();
                await model.initialize();
                addLog('info', '✅ ML model loaded.');

                botRef.current = new YeaminBot(engine, model);
                setReady(true);
                addLog('system', '✅ Bot ready! Type a move (e.g. e4, Nf3, d4) and press Enter.');
                addLog('info', `Starting position FEN: ${chessRef.current.fen()}`);
            } catch (e: any) {
                addLog('error', `❌ Init failed: ${e?.message ?? e}`);
                console.error(e);
            }
        }
        init();
    }, []);

    const handleMove = async () => {
        const moveText = input.trim();
        if (!moveText || !ready || loading) return;
        setInput('');

        const chess = chessRef.current;

        // Special commands
        if (moveText.toLowerCase() === 'reset') {
            chessRef.current = new Chess();
            addLog('system', '♟ Board reset to starting position.');
            addLog('info', `FEN: ${chessRef.current.fen()}`);
            return;
        }
        if (moveText.toLowerCase() === 'fen') {
            addLog('info', `Current FEN: ${chess.fen()}`);
            return;
        }
        if (moveText.toLowerCase() === 'moves') {
            const moves = chess.moves();
            addLog('info', `Legal moves: ${moves.join(', ')}`);
            return;
        }

        // Try to apply the player's move
        addLog('user', `You: ${moveText}`);
        let result;
        try {
            result = chess.move(moveText);
        } catch {
            result = null;
        }

        if (!result) {
            const legal = chess.moves();
            addLog('error', `❌ Illegal move: "${moveText}". Legal: ${legal.slice(0, 10).join(', ')}...`);
            return;
        }

        addLog('info', `✅ Move accepted: ${result.san} | FEN: ${chess.fen()}`);

        if (chess.isGameOver()) {
            addLog('system', `Game over: ${chess.isCheckmate() ? 'Checkmate!' : 'Draw!'}`);
            return;
        }

        // Bot replies
        setLoading(true);
        addLog('bot', 'Yeamin is thinking...');
        try {
            const decision = await botRef.current!.predictMove(chess.fen(), {
                styleStrength: 0.8,
                maxCplDrop: 100,
                depth: 10,
                multiPV: 5,
            });

            const botMove = chess.move(decision.bestMoveUci);
            if (botMove) {
                addLog('bot', `🤖 Yeamin plays: ${botMove.san}`);
                addLog('info', `Style score: ${decision.styleScore.toFixed(3)} | Eval: ${decision.engineEval}`);
                addLog('info', `FEN: ${chess.fen()}`);
            } else {
                addLog('error', `❌ Bot returned invalid move: ${decision.bestMoveUci}`);
            }

            if (chess.isGameOver()) {
                addLog('system', `Game over: ${chess.isCheckmate() ? 'Checkmate!' : 'Draw!'}`);
            }
        } catch (e: any) {
            addLog('error', `❌ Bot error: ${e?.message ?? e}`);
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const colorMap: Record<LogEntry['type'], string> = {
        system: '#F1C40F',
        info:   '#7f8c8d',
        user:   '#2ECC71',
        bot:    '#3498db',
        error:  '#C0392B',
    };

    return (
        <div style={{
            padding: '24px',
            height: '100vh',
            boxSizing: 'border-box',
            display: 'flex',
            flexDirection: 'column',
            fontFamily: 'monospace',
            background: '#0d1117',
            color: '#e6edf3',
        }}>
            <h2 style={{ color: '#2ECC71', marginBottom: '16px', fontFamily: 'Inter, sans-serif' }}>
                🤖 Yeamin Bot — Text Tester
            </h2>

            {/* Log window */}
            <div style={{
                flex: 1,
                overflowY: 'auto',
                background: '#161b22',
                border: '1px solid #30363d',
                borderRadius: '8px',
                padding: '16px',
                marginBottom: '16px',
                fontSize: '13px',
                lineHeight: '1.8',
            }}>
                {logs.map((log, i) => (
                    <div key={i} style={{ color: colorMap[log.type] }}>
                        {log.text}
                    </div>
                ))}
                <div ref={logEndRef} />
            </div>

            {/* Input */}
            <div style={{ display: 'flex', gap: '10px' }}>
                <input
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleMove()}
                    placeholder={ready ? 'Type a move (e4, Nf3...) or: reset, fen, moves' : 'Initializing...'}
                    disabled={!ready || loading}
                    style={{
                        flex: 1,
                        padding: '12px 16px',
                        background: '#21262d',
                        border: '1px solid #30363d',
                        borderRadius: '8px',
                        color: '#e6edf3',
                        fontSize: '14px',
                        fontFamily: 'monospace',
                        outline: 'none',
                    }}
                />
                <button
                    onClick={handleMove}
                    disabled={!ready || loading || !input.trim()}
                    style={{
                        padding: '12px 24px',
                        background: ready && !loading ? '#2ECC71' : '#2C3E50',
                        color: '#0d1117',
                        border: 'none',
                        borderRadius: '8px',
                        fontWeight: 'bold',
                        cursor: ready && !loading ? 'pointer' : 'not-allowed',
                        fontSize: '14px',
                    }}
                >
                    {loading ? '...' : 'Send'}
                </button>
            </div>

            <p style={{ color: '#484f58', fontSize: '12px', marginTop: '10px', fontFamily: 'monospace' }}>
                Commands: <span style={{ color: '#8b949e' }}>e4</span> • <span style={{ color: '#8b949e' }}>reset</span> • <span style={{ color: '#8b949e' }}>fen</span> • <span style={{ color: '#8b949e' }}>moves</span>
            </p>
        </div>
    );
}
