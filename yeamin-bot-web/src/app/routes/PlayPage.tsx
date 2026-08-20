import { useState, useEffect, useRef, useCallback } from 'react';
import { Chess } from 'chess.js';
import type { Api } from '@lichess-org/chessground/api';
import type { Key, Dests, Color as CgColor } from '@lichess-org/chessground/types';
import { ChessgroundBoard } from '../../components/chess/ChessgroundBoard';
import { StockfishService } from '../../engine/StockfishService';
import { YeaminModel } from '../../ml/ModelLoader';
import { YeaminBot } from '../../bot/YeaminBot';
import './PlayPage.css';

// ─── Types ───────────────────────────────────────────────────────────────────

interface GameSettings {
    playerColor: 'white' | 'black';
    styleStrength: number;
    depth: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const DEPTH_TIME: Record<number, string> = {
    5: '< 1s', 6: '~1s', 7: '~2s', 8: '~3s', 9: '~4s', 10: '~5s',
    11: '~8s', 12: '~12s', 13: '~18s', 14: '~25s', 15: '~35s',
    16: '~50s', 17: '~70s', 18: '~90s',
};

const STYLE_PRESETS = [
    { label: 'Pure Engine', value: 0.0, icon: '⚙️', desc: 'Plays the objectively best Stockfish moves' },
    { label: 'Mostly Engine', value: 0.3, icon: '🧮', desc: 'Engine-leaning with a touch of style' },
    { label: 'Balanced', value: 0.5, icon: '⚖️', desc: 'Equal mix of engine strength and personal style' },
    { label: 'Yeamin Style', value: 0.8, icon: '🤖', desc: 'Plays just like Yeamin from his real games' },
    { label: 'Pure Yeamin', value: 1.0, icon: '🧠', desc: "Always picks Yeamin's exact move regardless of evaluation" },
];



function getThinkTime(depth: number, style: number): string {
    if (style >= 1.0) return '< 1s (instant pick)';
    return DEPTH_TIME[depth] ?? '~?s';
}

// ─── Chess helpers ────────────────────────────────────────────────────────────

function getLegalDests(chess: Chess): Dests {
    const dests: Dests = new Map();
    for (const move of chess.moves({ verbose: true }) as any[]) {
        const src = move.from as Key;
        if (!dests.has(src)) dests.set(src, []);
        dests.get(src)!.push(move.to as Key);
    }
    return dests;
}

function cgColor(chess: Chess): CgColor {
    return chess.turn() === 'w' ? 'white' : 'black';
}

// Piece values for material count
const PIECE_VALUES: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9 };

function getMaterial(chess: Chess) {
    const board = chess.board();
    let wMat = 0, bMat = 0;
    const wCaptured: string[] = [], bCaptured: string[] = [];
    const startCount: Record<string, number> = { p: 8, n: 2, b: 2, r: 2, q: 1 };
    const wCount: Record<string, number> = { p: 0, n: 0, b: 0, r: 0, q: 0 };
    const bCount: Record<string, number> = { p: 0, n: 0, b: 0, r: 0, q: 0 };

    for (const row of board) {
        for (const sq of row) {
            if (!sq || sq.type === 'k') continue;
            if (sq.color === 'w') { wCount[sq.type] = (wCount[sq.type] || 0) + 1; wMat += PIECE_VALUES[sq.type]; }
            else { bCount[sq.type] = (bCount[sq.type] || 0) + 1; bMat += PIECE_VALUES[sq.type]; }
        }
    }
    // Captured pieces = start - current on board
    for (const [type, start] of Object.entries(startCount)) {
        const sym = type === 'p' ? '♟' : type === 'n' ? '♞' : type === 'b' ? '♝' : type === 'r' ? '♜' : '♛';
        const wSym = type === 'p' ? '♙' : type === 'n' ? '♘' : type === 'b' ? '♗' : type === 'r' ? '♖' : '♕';
        const wLost = start - (wCount[type] || 0);
        const bLost = start - (bCount[type] || 0);
        for (let i = 0; i < bLost; i++) wCaptured.push(wSym); // white captured black's pieces shown as black symbols
        for (let i = 0; i < wLost; i++) bCaptured.push(sym);
    }
    return { wMat, bMat, wCaptured, bCaptured, diff: wMat - bMat };
}

// ─── Setup Screen ─────────────────────────────────────────────────────────────

const DEPTH_SPECTRUM = [
    { label: 'Bullet', depth: 5, time: '< 1s', desc: 'Instant', icon: '⚡', color: '#e74c3c' },
    { label: 'Blitz', depth: 10, time: '~5s', desc: 'Fast', icon: '🔥', color: '#F1C40F' },
    { label: 'Rapid', depth: 14, time: '~25s', desc: 'Balanced', icon: '♟', color: '#3498db' },
    { label: 'Classical', depth: 18, time: '~90s', desc: 'Deep', icon: '🏛', color: '#9b59b6' },
];

function SetupScreen({ onStart }: { onStart: (s: GameSettings) => void }) {
    const [playerColor, setPlayerColor] = useState<'white' | 'black'>('white');
    const [styleIdx, setStyleIdx] = useState(3);
    const [depth, setDepth] = useState(10);
    const [hoverStyle, setHoverStyle] = useState<number | null>(null);
    const [mounted, setMounted] = useState(false);

    useEffect(() => { const t = setTimeout(() => setMounted(true), 50); return () => clearTimeout(t); }, []);

    const preset = STYLE_PRESETS[styleIdx];
    const previewStyle = hoverStyle !== null ? STYLE_PRESETS[hoverStyle] : preset;
    const thinkTime = getThinkTime(depth, preset.value);
    const depthPct = ((depth - 5) / (18 - 5)) * 100;

    // Find which spectrum zone we're in
    const activeZone = DEPTH_SPECTRUM.reduce((best, zone) =>
        Math.abs(zone.depth - depth) < Math.abs(best.depth - depth) ? zone : best
    );

    return (
        <div className={`ss-root ${mounted ? 'ss-mounted' : ''}`}>
            {/* Ambient background */}
            <div className="ss-ambient">
                <div className="ss-orb ss-orb-1" />
                <div className="ss-orb ss-orb-2" />
                <div className="ss-orb ss-orb-3" />
            </div>

            <div className="ss-container">
                {/* ── Left panel: Board preview + Identity ── */}
                <div className="ss-left">
                    <div className="ss-brand">
                        <div className="ss-brand-icon">♟</div>
                        <div>
                            <div className="ss-brand-name">Yeamin Chess Bot</div>
                            <div className="ss-brand-sub">Personal AI · 10,994 games trained</div>
                        </div>
                    </div>

                    {/* Mini board preview */}
                    <div className="ss-mini-board-wrap">
                        <div className="ss-mini-board-glow" style={{ background: playerColor === 'white' ? 'rgba(236,240,241,0.15)' : 'rgba(44,62,80,0.4)' }} />
                        <div className="ss-mini-board">
                            {['♜♞♝♛♚♝♞♜', '♟♟♟♟♟♟♟♟', '        ', '        ', '    ♙   ', '      ♘ ', '♙♙♙♙ ♙♙♙', '♖♘♗♕♔♗  '].map((row, ri) =>
                                row.split('').map((ch, fi) => (
                                    <div key={`${ri}-${fi}`} className={`ss-sq ${(ri + fi) % 2 === 0 ? 'ss-sq-l' : 'ss-sq-d'}`}>
                                        {ch.trim() && (
                                            <span className={`ss-pc ${ri < 2 ? 'ss-pc-black' : 'ss-pc-white'}`}>{ch}</span>
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Side selection */}
                    <div className="ss-side-section">
                        <p className="ss-label">You play as</p>
                        <div className="ss-side-toggle">
                            <button
                                className={`ss-side-opt ${playerColor === 'white' ? 'ss-side-active' : ''}`}
                                onClick={() => setPlayerColor('white')}
                            >
                                <span className="ss-side-piece">♙</span>
                                <span>White</span>
                                <span className="ss-side-hint">First move</span>
                            </button>
                            <div className="ss-side-vs">VS</div>
                            <button
                                className={`ss-side-opt ss-side-dark ${playerColor === 'black' ? 'ss-side-active ss-side-active-dark' : ''}`}
                                onClick={() => setPlayerColor('black')}
                            >
                                <span className="ss-side-piece ss-side-piece-dark">♟</span>
                                <span>Black</span>
                                <span className="ss-side-hint">Second move</span>
                            </button>
                        </div>
                    </div>

                    {/* Stats */}
                    <div className="ss-stats">
                        {[
                            { v: '42.3%', l: 'Imitation Rate' },
                            { v: '321K', l: 'Positions' },
                            { v: 'Top-1', l: 'ML Accuracy' },
                        ].map((s, i) => (
                            <div key={i} className="ss-stat">
                                <span className="ss-stat-v">{s.v}</span>
                                <span className="ss-stat-l">{s.l}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* ── Right panel: Settings ── */}
                <div className="ss-right">
                    {/* ── Personality ── */}
                    <div className="ss-section">
                        <div className="ss-section-hd">
                            <span className="ss-section-title">Bot Personality</span>
                            <span className="ss-section-badge">{previewStyle.icon} {previewStyle.label}</span>
                        </div>

                        {/* Personality spectrum track */}
                        <div className="ss-personality-track">
                            {STYLE_PRESETS.map((p, i) => (
                                <button
                                    key={i}
                                    className={`ss-pers-node ${styleIdx === i ? 'ss-pers-active' : ''}`}
                                    onClick={() => setStyleIdx(i)}
                                    onMouseEnter={() => setHoverStyle(i)}
                                    onMouseLeave={() => setHoverStyle(null)}
                                >
                                    <div className="ss-pers-icon-wrap">
                                        <span className="ss-pers-icon">{p.icon}</span>
                                        {styleIdx === i && <div className="ss-pers-ring" />}
                                    </div>
                                    <span className="ss-pers-label">{p.label}</span>
                                    <span className="ss-pers-pct">{Math.round(p.value * 100)}%</span>
                                </button>
                            ))}
                            {/* Connecting line */}
                            <div className="ss-pers-line">
                                <div
                                    className="ss-pers-fill"
                                    style={{ width: `${(styleIdx / (STYLE_PRESETS.length - 1)) * 100}%` }}
                                />
                            </div>
                        </div>

                        {/* Description */}
                        <div className="ss-pers-desc">
                            <div className="ss-pers-desc-icon">{previewStyle.icon}</div>
                            <p>{previewStyle.desc}</p>
                        </div>
                    </div>

                    {/* ── Depth & Time ── */}
                    <div className="ss-section">
                        <div className="ss-section-hd">
                            <span className="ss-section-title">Search Depth & Time</span>
                            <div className="ss-time-badge" style={{ background: `${activeZone.color}18`, borderColor: `${activeZone.color}40`, color: activeZone.color }}>
                                {activeZone.icon} {thinkTime} per move
                            </div>
                        </div>

                        {/* Depth mode cards */}
                        <div className="ss-depth-cards">
                            {DEPTH_SPECTRUM.map(d => (
                                <button
                                    key={d.depth}
                                    className={`ss-depth-card ${depth === d.depth ? 'ss-depth-card-active' : ''}`}
                                    style={{ '--dc': d.color } as React.CSSProperties}
                                    onClick={() => setDepth(d.depth)}
                                >
                                    <span className="ss-dc-icon">{d.icon}</span>
                                    <span className="ss-dc-name">{d.label}</span>
                                    <span className="ss-dc-time">{d.time}</span>
                                    <span className="ss-dc-desc">{d.desc}</span>
                                    {depth === d.depth && <div className="ss-dc-active-bar" style={{ background: d.color }} />}
                                </button>
                            ))}
                        </div>

                        {/* Custom slider */}
                        <div className="ss-slider-wrap">
                            <div className="ss-slider-labels">
                                <span>Fast (5)</span>
                                <span style={{ color: activeZone.color, fontWeight: 700 }}>Depth {depth}</span>
                                <span>Deep (18)</span>
                            </div>
                            <div className="ss-slider-track-wrap">
                                <div className="ss-slider-track">
                                    <div
                                        className="ss-slider-fill"
                                        style={{ width: `${depthPct}%`, background: `linear-gradient(90deg, #e74c3c, ${activeZone.color})` }}
                                    />
                                    <div className="ss-slider-thumb" style={{ left: `${depthPct}%`, borderColor: activeZone.color, boxShadow: `0 0 12px ${activeZone.color}60` }} />
                                </div>
                                <input
                                    type="range" min={5} max={18} step={1} value={depth}
                                    onChange={e => setDepth(+e.target.value)}
                                    className="ss-slider-input"
                                />
                            </div>

                            {/* Time spectrum bar */}
                            <div className="ss-time-spectrum">
                                <div className="ss-ts-bar">
                                    <div className="ss-ts-fill" style={{ width: `${depthPct}%` }} />
                                </div>
                                <div className="ss-ts-labels">
                                    <span>⚡ Instant</span>
                                    <span>🔥 Fast</span>
                                    <span>♟ Balanced</span>
                                    <span>🏛 Deep</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ── Start ── */}
                    <button
                        className="ss-start-btn"
                        onClick={() => onStart({ playerColor, styleStrength: preset.value, depth })}
                    >
                        <div className="ss-start-content">
                            <span className="ss-start-piece">{playerColor === 'white' ? '♙' : '♟'}</span>
                            <div className="ss-start-text">
                                <span className="ss-start-main">Start Game as {playerColor === 'white' ? 'White' : 'Black'}</span>
                                <span className="ss-start-sub">{preset.label} · Depth {depth} · {thinkTime}/move</span>
                            </div>
                            <span className="ss-start-arrow">→</span>
                        </div>
                        <div className="ss-start-shimmer" />
                    </button>
                </div>
            </div>
        </div>
    );
}

// ─── Captured Pieces Row ─────────────────────────────────────────────────────

function CapturedRow({ pieces, advantage }: { pieces: string[]; advantage: number }) {
    return (
        <div className="captured-row">
            <span className="captured-pieces">{pieces.join('')}</span>
            {advantage > 0 && <span className="material-adv">+{advantage}</span>}
        </div>
    );
}

// ─── Main Game ────────────────────────────────────────────────────────────────

function Game({ settings, onResign }: { settings: GameSettings; onResign: () => void }) {
    const chessRef = useRef(new Chess());
    const cgRef = useRef<Api | null>(null);
    const botRef = useRef<YeaminBot | null>(null);
    const botBusy = useRef(false);

    const [status, setStatus] = useState('Loading bot...');
    const [moveHistory, setMoveHistory] = useState<string[]>([]);
    const [, setBotReady] = useState(false);
    const [thinkTimer, setThinkTimer] = useState<number | null>(null);
    const [material, setMaterial] = useState(getMaterial(new Chess()));
    const [isFlipped, setIsFlipped] = useState(false);
    const [gameOver, setGameOver] = useState(false);
    const thinkStart = useRef<number>(0);
    const historyEndRef = useRef<HTMLDivElement>(null);

    const { playerColor, styleStrength, depth } = settings;
    const botColor: CgColor = playerColor === 'white' ? 'black' : 'white';

    const playerLabel = playerColor === 'white' ? 'You' : 'You';
    const botLabel = 'Yeamin Bot';

    const effectiveOrientation: CgColor = isFlipped
        ? (playerColor === 'white' ? 'black' : 'white')
        : (playerColor as CgColor);

    // ── Bot init ──────────────────────────────────────────────────────────────
    useEffect(() => {
        (async () => {
            try {
                setStatus('Loading Stockfish engine...');
                const engine = new StockfishService();
                await engine.waitForReady();
                setStatus('Loading Yeamin ML model...');
                const model = new YeaminModel();
                await model.initialize();
                botRef.current = new YeaminBot(engine, model);
                setBotReady(true);

                if (playerColor === 'black') {
                    setTimeout(() => doBotMove(), 600);
                } else {
                    setStatus('Your turn');
                }
            } catch (e: any) {
                setStatus('❌ Failed to load bot');
                console.error(e);
            }
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        historyEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [moveHistory]);

    // ── Live think timer ──────────────────────────────────────────────────────
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (thinkTimer !== null) {
            interval = setInterval(() => {
                setThinkTimer(Math.floor((Date.now() - thinkStart.current) / 1000));
            }, 500);
        }
        return () => clearInterval(interval);
    }, [thinkTimer]);

    // ── Sync board state ──────────────────────────────────────────────────────
    const syncBoard = useCallback(() => {
        const chess = chessRef.current;
        const cg = cgRef.current;
        if (!cg) return;

        const isOver = chess.isGameOver();
        const currentTurn = cgColor(chess);

        cg.set({
            fen: chess.fen(),
            turnColor: currentTurn,
            movable: {
                color: isOver ? undefined : (playerColor as CgColor),
                dests: isOver ? new Map() : getLegalDests(chess),
                free: false,
            },
            check: chess.inCheck() ? currentTurn : undefined,
        });

        setMaterial(getMaterial(chess));
    }, [playerColor]);

    // ── Bot move ──────────────────────────────────────────────────────────────
    const doBotMove = useCallback(async () => {
        const chess = chessRef.current;
        if (!botRef.current || botBusy.current || chess.isGameOver()) return;

        botBusy.current = true;
        thinkStart.current = Date.now();
        setThinkTimer(0);
        setStatus('Yeamin is thinking...');

        try {
            const decision = await botRef.current.predictMove(chess.fen(), {
                styleStrength, maxCplDrop: 500, depth, multiPV: 15, skillLevel: 5
            });

            const from = decision.bestMoveUci.slice(0, 2) as Key;
            const to = decision.bestMoveUci.slice(2, 4) as Key;
            const promo = decision.bestMoveUci[4] as string | undefined;

            const result = chess.move({ from, to, promotion: promo });
            if (result) {
                cgRef.current?.move(from, to);
                setMoveHistory(h => [...h, result.san]);
            }

            // Clear busy flag BEFORE syncing board, because if there's a premove, 
            // syncBoard will immediately and synchronously fire onUserMove.
            botBusy.current = false;
            syncBoard();
            setThinkTimer(null);

            if (chess.isCheckmate()) { setStatus(`Checkmate! ${botColor === playerColor ? 'You win! 🏆' : 'Yeamin wins! 🤖'}`); setGameOver(true); }
            else if (chess.isDraw()) { setStatus('Draw! 🤝'); setGameOver(true); }
            else if (chess.isGameOver()) { setStatus('Game over'); setGameOver(true); }
            else setStatus('Your turn');
        } catch (e: any) {
            console.error('Bot error:', e);
            setStatus('Bot error — your turn');
            setThinkTimer(null);
        } finally {
            botBusy.current = false;
        }
    }, [styleStrength, depth, botColor, playerColor, syncBoard]);

    // ── Player move ───────────────────────────────────────────────────────────
    const onUserMove = useCallback((from: Key, to: Key) => {
        if (botBusy.current) return;
        const chess = chessRef.current;
        const piece = chess.get(from as any);
        const isPromo = piece?.type === 'p' && (to[1] === '8' || to[1] === '1');

        let result: any;
        try { result = chess.move({ from, to, promotion: isPromo ? 'q' : undefined }); }
        catch { result = null; }

        if (!result) { syncBoard(); return; }

        setMoveHistory(h => [...h, result.san]);

        if (chess.isCheckmate()) { syncBoard(); setStatus('Checkmate! You win! 🏆'); setGameOver(true); return; }
        if (chess.isDraw()) { syncBoard(); setStatus('Draw! 🤝'); setGameOver(true); return; }
        if (chess.isGameOver()) { syncBoard(); setStatus('Game over'); setGameOver(true); return; }

        syncBoard();
        setTimeout(doBotMove, 300);
    }, [syncBoard, doBotMove]);

    // ── Board mount ───────────────────────────────────────────────────────────
    const onBoardMount = useCallback((api: Api) => {
        cgRef.current = api;
        syncBoard();
    }, [syncBoard]);

    const cgConfig = {
        orientation: effectiveOrientation,
        coordinates: true,
        movable: {
            color: playerColor as CgColor,
            free: false,
            dests: getLegalDests(chessRef.current),
            showDests: true,
            events: { after: onUserMove },
        },
        highlight: { lastMove: true, check: true },
        animation: { enabled: true, duration: 200 },
        draggable: { enabled: true, showGhost: true },
        selectable: { enabled: true },
        premovable: { 
            enabled: true, 
            showDests: true, 
            castle: true 
        },
    };

    const thinking = thinkTimer !== null;

    // Which player is on top vs bottom (depends on orientation)
    const topColor = effectiveOrientation === 'white' ? botColor : (playerColor as CgColor);
    const bottomColor = effectiveOrientation === 'white' ? (playerColor as CgColor) : botColor;
    const topIsBot = topColor === botColor;
    const topName = topIsBot ? botLabel : playerLabel;
    const bottomName = topIsBot ? playerLabel : botLabel;
    const topCaptures = topIsBot
        ? (playerColor === 'white' ? material.bCaptured : material.wCaptured)
        : (playerColor === 'white' ? material.wCaptured : material.bCaptured);
    const bottomCaptures = topIsBot
        ? (playerColor === 'white' ? material.wCaptured : material.bCaptured)
        : (playerColor === 'white' ? material.bCaptured : material.wCaptured);
    const topAdv = topColor === 'white' ? Math.max(0, material.diff) : Math.max(0, -material.diff);
    const bottomAdv = bottomColor === 'white' ? Math.max(0, material.diff) : Math.max(0, -material.diff);

    const moveCount = chessRef.current.moveNumber();

    return (
        <div className="game-root">
            {/* ── Top bar ── */}
            <header className="game-topbar">
                <button className="back-btn" onClick={onResign}>
                    ← Back
                </button>
                <div className="topbar-center">
                    <span className="topbar-logo">♟ Yeamin Chess Bot</span>
                    <span className="topbar-move-count">Move {moveCount}</span>
                </div>
                <div className="topbar-actions">
                    <button
                        className="icon-btn"
                        title="Flip board"
                        onClick={() => {
                            setIsFlipped(f => !f);
                            cgRef.current?.toggleOrientation();
                        }}
                    >
                        ⇅
                    </button>
                    <button className="icon-btn resign-icon-btn" title="Resign" onClick={onResign}>
                        🏳
                    </button>
                </div>
            </header>

            {/* ── Main content ── */}
            <main className="game-main">
                {/* ── Board column ── */}
                <div className="game-board-col">
                    {/* Top player */}
                    <div className="player-strip top-player">
                        <div className="player-info">
                            <span className="player-avatar-icon">{topIsBot ? '🤖' : '👤'}</span>
                            <div className="player-details">
                                <span className="player-name-text">{topName}</span>
                                {topIsBot && <span className="player-sub-text">Style {Math.round(styleStrength * 100)}% · D{depth}</span>}
                            </div>
                        </div>
                        <CapturedRow pieces={topCaptures} advantage={topAdv} />
                        {thinking && topIsBot && (
                            <div className="think-indicator">
                                <span className="think-dots">
                                    <span /><span /><span />
                                </span>
                                <span className="think-secs">{thinkTimer}s</span>
                            </div>
                        )}
                    </div>

                    {/* Board */}
                    <div className="board-shell">
                        <ChessgroundBoard config={cgConfig} onMount={onBoardMount} />
                    </div>

                    {/* Bottom player */}
                    <div className="player-strip bottom-player">
                        <div className="player-info">
                            <span className="player-avatar-icon">{topIsBot ? '👤' : '🤖'}</span>
                            <div className="player-details">
                                <span className="player-name-text">{bottomName}</span>
                                {!topIsBot && <span className="player-sub-text">Style {Math.round(styleStrength * 100)}% · D{depth}</span>}
                            </div>
                        </div>
                        <CapturedRow pieces={bottomCaptures} advantage={bottomAdv} />
                    </div>
                </div>

                {/* ── Sidebar ── */}
                <aside className="game-sidebar">
                    {/* Status */}
                    <div className={`game-status-card ${thinking ? 'gs-thinking' : gameOver ? 'gs-over' : 'gs-player'}`}>
                        <div className="gs-dot" />
                        <span className="gs-text">{status}</span>
                    </div>

                    {/* Move history */}
                    <div className="history-panel">
                        <div className="history-header">
                            <h3 className="history-heading">Moves</h3>
                            <span className="history-count">{Math.ceil(moveHistory.length / 2)} / ∞</span>
                        </div>
                        <div className="history-body">
                            {moveHistory.length === 0
                                ? <p className="history-placeholder">No moves yet — make your first move!</p>
                                : (
                                    <div className="move-list">
                                        {Array.from({ length: Math.ceil(moveHistory.length / 2) }, (_, i) => (
                                            <div key={i} className="move-row">
                                                <span className="move-num">{i + 1}</span>
                                                <span className={`move-cell ${playerColor === 'white' ? 'move-you' : 'move-bot-cell'}`}>
                                                    {moveHistory[i * 2]}
                                                </span>
                                                <span className={`move-cell ${playerColor === 'white' ? 'move-bot-cell' : 'move-you'}`}>
                                                    {moveHistory[i * 2 + 1] ?? ''}
                                                </span>
                                            </div>
                                        ))}
                                        <div ref={historyEndRef} />
                                    </div>
                                )
                            }
                        </div>
                    </div>

                    {/* Game info */}
                    <div className="game-info-panel">
                        <div className="info-row">
                            <span className="info-label">You play</span>
                            <span className="info-value">{playerColor === 'white' ? 'White ♙' : 'Black ♟'}</span>
                        </div>
                        <div className="info-row">
                            <span className="info-label">Bot style</span>
                            <span className="info-value" style={{ color: '#2ECC71' }}>
                                {STYLE_PRESETS.find(p => p.value === styleStrength)?.label ?? `${Math.round(styleStrength * 100)}%`}
                            </span>
                        </div>
                        <div className="info-row">
                            <span className="info-label">Search depth</span>
                            <span className="info-value">{depth} ply</span>
                        </div>
                        <div className="info-row">
                            <span className="info-label">Est. think time</span>
                            <span className="info-value" style={{ color: '#F1C40F' }}>{getThinkTime(depth, styleStrength)}</span>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="sidebar-action-row">
                        <button className="sa-btn sa-flip" onClick={() => { setIsFlipped(f => !f); cgRef.current?.toggleOrientation(); }}>
                            ⇅ Flip
                        </button>
                        <button className="sa-btn sa-resign" onClick={onResign}>
                            🏳 New Game
                        </button>
                    </div>
                </aside>
            </main>
        </div>
    );
}

// ─── Root ─────────────────────────────────────────────────────────────────────

export function PlayPage() {
    const [settings, setSettings] = useState<GameSettings | null>(null);
    const [gameKey, setGameKey] = useState(0);

    if (!settings) return <SetupScreen onStart={s => setSettings(s)} />;
    return <Game key={gameKey} settings={settings} onResign={() => { setSettings(null); setGameKey(k => k + 1); }} />;
}
