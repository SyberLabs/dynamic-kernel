import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import ControlHub from './pages/ControlHub';
import BaseSystem from './pages/BaseSystem';
import MallPort from './pages/MallPort';
import ComparePort from './pages/ComparePort';
import NeuralPort from './pages/NeuralPort';
import './index.css';

// A simple back button to appear on the sub-apps
const NavLayer = () => {
  const loc = useLocation();
  if (loc.pathname === '/') return null;
  
  return (
    <Link to="/" style={{
      position: 'absolute', top: 16, right: 16, zIndex: 1000,
      background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
      color: '#fff', textDecoration: 'none', padding: '6px 12px',
      borderRadius: 6, fontSize: 12, fontWeight: 500, fontFamily: 'Inter'
    }}>
      ← Hub
    </Link>
  );
};

export default function App() {
  return (
    <Router>
      <NavLayer />
      <Routes>
        <Route path="/" element={<ControlHub />} />
        <Route path="/base" element={<BaseSystem />} />
        <Route path="/mall" element={<MallPort />} />
        <Route path="/compare" element={<ComparePort />} />
        <Route path="/neural" element={<NeuralPort />} />
      </Routes>
    </Router>
  );
}
