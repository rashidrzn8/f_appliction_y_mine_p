import { useState } from 'react';
import { uploadDataset } from '../services/api';

const UploadDataset = ({ onUploadSuccess }) => {
    const [file, setFile] = useState(null);
    const [domainName, setDomainName] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!file || !domainName.trim()) {
            setError("Both file and domain name are required.");
            return;
        }

        setLoading(true);
        setError('');
        setSuccessMessage('');

        try {
            const data = await uploadDataset(file, domainName.trim());
            setSuccessMessage(`Successfully trained model for domain: ${data.domain || domainName}`);
            setFile(null);
            setDomainName('');
            
            // Notify parent to refresh domains if needed
            if (onUploadSuccess) {
                onUploadSuccess(data.domain || domainName);
            }
        } catch (err) {
            setError(err.message || 'Failed to upload and train dataset');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto mt-8 p-6 bg-white rounded-xl shadow-lg border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-2 text-center">Train New Domain</h2>
            <p className="text-sm text-gray-500 text-center mb-6">Upload a CSV file containing 'text' and 'label' columns to train a custom model.</p>
            
            <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                    <label htmlFor="domainName" className="block text-sm font-medium text-gray-700 mb-2">
                        Domain Name
                    </label>
                    <input
                        type="text"
                        id="domainName"
                        value={domainName}
                        onChange={(e) => setDomainName(e.target.value)}
                        placeholder="e.g. health, tech, finance"
                        className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-primary focus:border-transparent transition-all text-gray-700 placeholder-gray-400"
                        disabled={loading}
                    />
                </div>

                <div>
                    <label htmlFor="datasetFile" className="block text-sm font-medium text-gray-700 mb-2">
                        Dataset (CSV files only)
                    </label>
                    <div className="flex items-center justify-center w-full">
                        <label htmlFor="datasetFile" className={`flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}>
                            <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                <svg className="w-8 h-8 mb-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                                    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                                </svg>
                                <p className="mb-2 text-sm text-gray-500 font-semibold">{file ? file.name : 'Click to upload or drag and drop'}</p>
                                <p className="text-xs text-gray-500">CSV dataset (Max 5MB recommended)</p>
                            </div>
                            <input 
                                id="datasetFile" 
                                type="file" 
                                accept=".csv" 
                                className="hidden" 
                                onChange={handleFileChange}
                                disabled={loading}
                            />
                        </label>
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={loading || !file || !domainName.trim()}
                    className={`w-full py-3 px-6 text-white font-semibold rounded-lg shadow-md transition-all transform hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500
                        ${loading || !file || !domainName.trim()
                            ? 'bg-gray-400 cursor-not-allowed' 
                            : 'bg-green-600 hover:bg-green-700'
                        }`}
                >
                    {loading ? (
                        <span className="flex items-center justify-center">
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Training Custom Model... This may take a few minutes.
                        </span>
                    ) : 'Upload and Train'}
                </button>
            </form>

            {error && (
                <div className="mt-4 p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 flex items-start text-sm">
                    <svg className="w-5 h-5 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span>{error}</span>
                </div>
            )}

            {successMessage && (
                <div className="mt-4 p-4 rounded-lg bg-green-50 border border-green-200 text-green-700 flex items-start text-sm">
                    <svg className="w-5 h-5 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    <span>{successMessage}</span>
                </div>
            )}
        </div>
    );
};

export default UploadDataset;
