import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import About from './components/About';
import ClassificationForm from './components/ClassificationForm';
import UploadDataset from './components/UploadDataset';
import ModelsPage from './components/ModelsPage';
import { checkHealth } from './services/api';

function App() {
  const [serverStatus, setServerStatus] = useState('checking');
  const [availableDomains, setAvailableDomains] = useState(['base']);
  const [selectedDomain, setSelectedDomain] = useState('base');

  useEffect(() => {
    checkHealth().then(data => {
      setServerStatus(data.status === 'healthy' || data.status === 'degraded' ? 'online' : 'offline');
      if (data.loaded_domains && data.loaded_domains.length > 0) {
        setAvailableDomains(data.loaded_domains);
      }
    }).catch(() => setServerStatus('offline'));
  }, []);

  const handleUploadSuccess = (newDomain) => {
    if (!availableDomains.includes(newDomain)) {
      setAvailableDomains(prev => [...prev, newDomain]);
    }
    setSelectedDomain(newDomain);
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 flex flex-col font-sans">
        <Navbar serverStatus={serverStatus} />

        <main className="flex-grow flex flex-col p-4 sm:p-8">
          <Routes>
            <Route path="/" element={
              <div className="w-full max-w-4xl mx-auto flex flex-col">
                <div className="text-center mb-10 mt-8">
                  <h1 className="text-4xl sm:text-5xl font-extrabold text-gray-900 tracking-tight mb-4">
                    Headline Text <span className="text-primary">Classifier</span>
                  </h1>
                  <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                    Leverage the power of our advanced neural network to classify your text in seconds.
                  </p>
                </div>
                
                <ClassificationForm 
                  selectedDomain={selectedDomain}
                  onDomainChange={setSelectedDomain}
                  availableDomains={availableDomains}
                />
              </div>
            } />

            <Route path="/train" element={
              <div className="w-full max-w-4xl mx-auto mt-4">
                <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-100 mb-8 text-center bg-gradient-to-br from-indigo-50 to-white">
                    <h1 className="text-3xl font-extrabold text-gray-900 mb-3">Custom Domain Training</h1>
                    <p className="text-gray-600 max-w-2xl mx-auto">Upload datasets to dynamically generate and load custom classification models without restarting the server.</p>
                </div>
                {serverStatus === 'online' ? (
                  <UploadDataset onUploadSuccess={handleUploadSuccess} />
                ) : (
                  <div className="bg-red-50 text-red-600 p-6 rounded-xl text-center border border-red-100 flex flex-col items-center">
                    <svg className="w-12 h-12 mb-3 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                    <p className="font-semibold text-lg">Server Connection Required</p>
                    <p className="text-sm mt-1">Please ensure the backend API is running to upload datasets.</p>
                  </div>
                )}
              </div>
            } />

            <Route path="/models" element={
              <ModelsPage 
                availableDomains={availableDomains} 
                serverStatus={serverStatus} 
              />
            } />

            <Route path="/about" element={<About />} />
          </Routes>
        </main>

        <footer className="bg-white border-t border-gray-200 mt-auto">
          <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <p className="text-center text-gray-500 text-sm">
              © {new Date().getFullYear()} Advanced Text Classification System. Built with React & Flask.
            </p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
