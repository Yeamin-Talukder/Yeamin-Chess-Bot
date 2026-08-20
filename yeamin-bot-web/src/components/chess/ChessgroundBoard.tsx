import { useEffect, useRef } from 'react';
import { Chessground } from '@lichess-org/chessground';
import type { Api } from '@lichess-org/chessground/api';
import type { Config } from '@lichess-org/chessground/config';

interface ChessgroundBoardProps {
    config: Config;
    onMount?: (api: Api) => void;
}

/**
 * A thin React wrapper around the Chessground library.
 * `config` is applied on every render via `cg.set()`.
 * `onMount` fires once when the board is ready.
 */
export function ChessgroundBoard({ config, onMount }: ChessgroundBoardProps) {
    const boardRef = useRef<HTMLDivElement>(null);
    const cgRef = useRef<Api | null>(null);

    // Mount once
    useEffect(() => {
        if (!boardRef.current) return;

        const cg = Chessground(boardRef.current, config);
        cgRef.current = cg;
        onMount?.(cg);

        return () => {
            cg.destroy();
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
        <div
            ref={boardRef}
            style={{ width: '100%', height: '100%' }}
        />
    );
}
