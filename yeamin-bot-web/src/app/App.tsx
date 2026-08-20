
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { PlayPage } from './routes/PlayPage';
import { LandingPage } from './routes/LandingPage';
import { TextTestPage } from './routes/TextTestPage';

export function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/play" element={<PlayPage />} />
          <Route path="/test" element={<TextTestPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
