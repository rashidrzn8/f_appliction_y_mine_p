import React from 'react';

const ModelsPage = ({ availableDomains, serverStatus }) => {
  return (
    <div className="w-full max-w-4xl mx-auto mt-4">
      <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-100 mb-8 text-center bg-gradient-to-br from-indigo-50 to-white">
        <h1 className="text-3xl font-extrabold text-gray-900 mb-3">Models & API Endpoints</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Explore the available text classification models and learn how to integrate them into your own applications using our REST API.
        </p>
      </div>

      {serverStatus === 'online' ? (
        <div className="space-y-6">
          {availableDomains.map((domain) => (
            <div key={domain} className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow duration-300">
              <div className="px-6 py-4 border-b border-gray-100 bg-gray-50 flex justify-between items-center">
                <div className="flex items-center space-x-3">
                  <span className="bg-primary/10 text-primary p-2 rounded-lg">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path>
                    </svg>
                  </span>
                  <h2 className="text-xl font-bold text-gray-900 capitalize">{domain === 'base' ? 'Base (Default)' : domain} Model</h2>
                </div>
                <span className="px-3 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded-full border border-green-200">Available</span>
              </div>
              
              <div className="p-6">
                <div className="mb-5">
                  <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Endpoint URL</h3>
                  <div className="flex items-center space-x-3 bg-gray-50 p-3 rounded-xl border border-gray-200">
                    <span className="px-2.5 py-1 bg-indigo-100 text-indigo-700 text-xs font-bold rounded-md border border-indigo-200 shadow-sm">POST</span>
                    <code className="text-gray-800 font-mono text-sm break-all font-medium">http://localhost:5002/process</code>
                  </div>
                </div>

                <div>
                  <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Example JSON Payload</h3>
                  <div className="bg-gray-900 rounded-xl p-4 shadow-inner border border-gray-800 overflow-x-auto">
                    <pre className="text-gray-300 font-mono text-sm leading-relaxed">
{`{
  "input": "Your text to classify goes here...",
  "domain": "${domain}"
}`}
                    </pre>
                  </div>
                </div>
                
                <div className="mt-5">
                  <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Example Response</h3>
                  <div className="bg-gray-900 rounded-xl p-4 shadow-inner border border-gray-800 overflow-x-auto">
                    <pre className="text-green-400 font-mono text-sm leading-relaxed">
{`{
  "message": "Predicted Class",
  "details": {
    "class": "Predicted Class",
    "confidence": 0.95
  }
}`}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-red-50 text-red-600 p-6 rounded-xl text-center border border-red-100 flex flex-col items-center">
          <svg className="w-12 h-12 mb-3 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
          <p className="font-semibold text-lg">Server Connection Required</p>
          <p className="text-sm mt-1">Please ensure the backend API is running to view active models.</p>
        </div>
      )}
    </div>
  );
};

export default ModelsPage;
