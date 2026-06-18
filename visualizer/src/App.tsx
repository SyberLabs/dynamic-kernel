import { BrowserRouter as Router, Link, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import BaseSystem from './pages/BaseSystem';
import ComparePort from './pages/ComparePort';
import ControlHub from './pages/ControlHub';
import DemoPort from './pages/DemoPort';
import MethodPort from './pages/MethodPort';
import MallPort from './pages/MallPort';
import NeuralPort from './pages/NeuralPort';
import './index.css';

const NavLayer = () => {
  const loc = useLocation();
  if (loc.pathname === '/') return null;

  return (
    <Link
      to="/"
      style={{
        position: 'absolute',
        top: 16,
        right: 16,
        zIndex: 1000,
        background: 'rgba(255,255,255,0.05)',
        border: '1px solid rgba(255,255,255,0.1)',
        color: '#fff',
        textDecoration: 'none',
        padding: '6px 12px',
        borderRadius: 6,
        fontSize: 12,
        fontWeight: 500,
        fontFamily: 'Inter',
      }}
    >
      Hub
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
        <Route path="/demo" element={<DemoPort />} />
        <Route path="demo" element={<DemoPort />} />
        <Route path="/methodology" element={<MethodPort />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}
