// stockfish.js natively expects to be run in a Worker context
// and handles onmessage/postMessage by itself.

export function createStockfishWorker(): Worker {
    return new Worker('/stockfish/stockfish.js');
}
