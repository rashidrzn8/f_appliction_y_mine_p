const About = () => {
    return (
        <div className="max-w-4xl mx-auto p-4 sm:p-8 animate-fade-in">
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
                <div className="bg-gradient-to-r from-primary to-indigo-600 px-8 py-12 text-white text-center">
                    <h1 className="text-4xl font-extrabold mb-4">About the System</h1>
                    <p className="text-lg opacity-90 max-w-2xl mx-auto">
                        A dynamic, AI-powered text classification architecture built to adapt to your specific domain needs.
                    </p>
                </div>
                
                <div className="p-8 sm:p-12">
                    <div className="prose prose-indigo max-w-none text-gray-600">
                        <h2 className="text-2xl font-bold text-gray-900 mb-4 border-b pb-2">How It Works</h2>
                        <p className="mb-6 leading-relaxed">
                            This application uses a standard deep-learning model powered by TensorFlow and Keras to classify text inputs. What makes this system unique is its <strong>Dynamic Domain capability</strong>. Instead of being locked into a single set of categories, you can feed the system custom datasets and instantly tailor the classification logic to specialized areas such as Healthcare, Finance, or Support Tickets.
                        </p>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 my-10">
                            <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
                                <div className="w-12 h-12 bg-indigo-100 text-indigo-600 rounded-lg flex items-center justify-center mb-4">
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                                </div>
                                <h3 className="text-xl font-bold text-gray-900 mb-2">Real-time Inference</h3>
                                <p className="text-sm">Pass text to the standard pre-trained model to categorize general information instantly via the frontend.</p>
                            </div>
                            
                            <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
                                <div className="w-12 h-12 bg-green-100 text-green-600 rounded-lg flex items-center justify-center mb-4">
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                                </div>
                                <h3 className="text-xl font-bold text-gray-900 mb-2">Custom Training</h3>
                                <p className="text-sm">Upload a CSV mapped with `text` and `label` columns. The server automatically cleans, tokenizes, and trains a specialized neural network on your data.</p>
                            </div>
                        </div>

                        <h2 className="text-2xl font-bold text-gray-900 mb-4 border-b pb-2">Technology Stack</h2>
                        <ul className="space-y-3 mb-6">
                            <li className="flex items-start">
                                <svg className="w-5 h-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" fill="fill" stroke="currentColor" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path></svg>
                                <span><strong>Frontend:</strong> React, Vite, TailwindCSS, React Router</span>
                            </li>
                            <li className="flex items-start">
                                <svg className="w-5 h-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" fill="fill" stroke="currentColor" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path></svg>
                                <span><strong>Backend:</strong> Python, Flask, Pandas</span>
                            </li>
                            <li className="flex items-start">
                                <svg className="w-5 h-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" fill="fill" stroke="currentColor" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path></svg>
                                <span><strong>Machine Learning:</strong> TensorFlow, Keras, Scikit-Learn</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default About;
