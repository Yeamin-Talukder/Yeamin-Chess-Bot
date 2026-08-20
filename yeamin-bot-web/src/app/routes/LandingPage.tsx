import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

const STATS = [
    { value: '10,994', label: 'Real Games', icon: '🎮' },
    { value: '321,339', label: 'Positions Analyzed', icon: '📊' },
    { value: '42.3%', label: 'Move Imitation', icon: '🎯' },
    { value: 'Top-1', label: 'Model Accuracy', icon: '🏆' },
];

const FEATURES = [
    {
        icon: '🧠',
        title: 'Trained on Real Games',
        desc: 'Analyzed 10,994 Chess.com games played by Yeamin to learn his unique patterns, openings, and tactical tendencies.',
        color: '#2ECC71',
    },
    {
        icon: '⚡',
        title: 'Stockfish + ML Fusion',
        desc: 'Combines Stockfish engine analysis at your chosen depth with a personal ML model that imitates Yeamin\'s decision-making.',
        color: '#3498db',
    },
    {
        icon: '🎮',
        title: 'Adjustable Personality',
        desc: 'Slide from Pure Engine to Pure Yeamin. Pick how strongly the bot plays like Yeamin versus playing the objectively best move.',
        color: '#F1C40F',
    },
    {
        icon: '♟',
        title: 'Lichess Board Engine',
        desc: 'Uses the same board engine as lichess.org — smooth piece animations, legal move highlighting, drag & drop.',
        color: '#9b59b6',
    },
];

const TECH = ['Stockfish 16', 'HistGradient Boosting', 'React + Vite', 'Chess.js', 'Chessground'];

// Decorative mini-board
const BOARD_PATTERN = [
    ['♜','♞','♝','♛','♚','♝','♞','♜'],
    ['♟','♟','♟','♟','♟','♟','♟','♟'],
    ['','','','','','','',''],
    ['','','','','','','',''],
    ['','','','','♙','','',''],
    ['','','','','','♘','',''],
    ['♙','♙','♙','♙','','♙','♙','♙'],
    ['♖','♘','♗','♕','♔','♗','','♖'],
];

function DecoBoard() {
    return (
        <div className="deco-board">
            {BOARD_PATTERN.map((row, ri) =>
                row.map((piece, fi) => (
                    <div
                        key={`${ri}-${fi}`}
                        className={`deco-sq ${(ri + fi) % 2 === 0 ? 'sq-light' : 'sq-dark'}`}
                    >
                        {piece && <span className={`deco-piece ${ri < 2 ? 'piece-black' : 'piece-white'}`}>{piece}</span>}
                    </div>
                ))
            )}
        </div>
    );
}

export function LandingPage() {
    const navigate = useNavigate();

    return (
        <div className="landing-root">
            {/* ── Hero ── */}
            <section className="hero-section">
                <div className="hero-content">
                    <div className="hero-badge">
                        <span className="badge-dot" />
                        AI Chess Bot · Powered by Stockfish + ML
                    </div>
                    <h1 className="hero-title">
                        Play Against<br />
                        <span className="hero-name-highlight">Yeamin's</span><br />
                        Chess Style
                    </h1>
                    <p className="hero-desc">
                        A personal chess AI trained from 10,994 real games. It doesn't just play
                        good moves — it plays <em>Yeamin's</em> moves.
                    </p>
                    <div className="hero-ctas">
                        <button className="cta-primary" onClick={() => navigate('/play')}>
                            <span>Play Now</span>
                            <span className="cta-arrow">♟</span>
                        </button>
                        <button className="cta-secondary" onClick={() => navigate('/test')}>
                            Text Tester
                        </button>
                    </div>
                </div>

                <div className="hero-visual">
                    <div className="hero-board-glow" />
                    <DecoBoard />
                </div>
            </section>

            {/* ── Stats ── */}
            <section className="stats-section">
                {STATS.map((s, i) => (
                    <div key={i} className="stat-card">
                        <span className="stat-icon">{s.icon}</span>
                        <span className="stat-value">{s.value}</span>
                        <span className="stat-label">{s.label}</span>
                    </div>
                ))}
            </section>

            {/* ── Features ── */}
            <section className="features-section">
                <div className="section-header">
                    <h2 className="section-title">How It Works</h2>
                    <p className="section-sub">Built with real game data and modern machine learning</p>
                </div>
                <div className="features-grid">
                    {FEATURES.map((f, i) => (
                        <div key={i} className="feature-card" style={{ '--accent': f.color } as React.CSSProperties}>
                            <div className="feature-icon-wrap">
                                <span className="feature-icon">{f.icon}</span>
                            </div>
                            <h3 className="feature-title">{f.title}</h3>
                            <p className="feature-desc">{f.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* ── Model performance ── */}
            <section className="perf-section">
                <div className="perf-content">
                    <h2 className="section-title">Model Performance</h2>
                    <p className="section-sub">Compared against raw Stockfish top-1 on the same test positions</p>
                    <div className="perf-grid">
                        {[
                            { label: 'Model Top-1 Accuracy', value: 42.3, color: '#2ECC71', vs: 'vs Stockfish 36.9%' },
                            { label: 'Model Top-3 Coverage', value: 66.1, color: '#3498db', vs: 'vs Stockfish 63.3%' },
                            { label: 'Style Imitation', value: 56.9, color: '#F1C40F', vs: 'agree with my real moves' },
                            { label: 'Candidate Coverage', value: 74.4, color: '#9b59b6', vs: 'my move in candidate list' },
                        ].map((m, i) => (
                            <div key={i} className="perf-card">
                                <div className="perf-label">{m.label}</div>
                                <div className="perf-bar-wrap">
                                    <div className="perf-bar" style={{ width: `${m.value}%`, background: m.color }} />
                                </div>
                                <div className="perf-row">
                                    <span className="perf-value" style={{ color: m.color }}>{m.value}%</span>
                                    <span className="perf-vs">{m.vs}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Tech ── */}
            <section className="tech-section">
                <p className="tech-label">Built with</p>
                <div className="tech-chips">
                    {TECH.map((t, i) => (
                        <span key={i} className="tech-chip">{t}</span>
                    ))}
                </div>
            </section>

            {/* ── Footer CTA ── */}
            <section className="footer-cta">
                <h2 className="footer-title">Ready to Play?</h2>
                <p className="footer-sub">Can you beat Yeamin's chess brain?</p>
                <button className="cta-primary large-cta" onClick={() => navigate('/play')}>
                    Start a Game ♟
                </button>
            </section>
        </div>
    );
}
