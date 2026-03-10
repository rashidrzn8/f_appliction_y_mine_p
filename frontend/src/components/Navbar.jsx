import { Link, useLocation } from 'react-router-dom';

const Navbar = ({ serverStatus }) => {
  const location = useLocation();

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Models & API', path: '/models' },
    { name: 'Train Domain', path: '/train' },
    { name: 'About', path: '/about' },
  ];

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-primary to-indigo-600">
              AI Classifier
            </Link>
          </div>

          <div className="flex items-center space-x-8">
            <div className="hidden sm:flex space-x-6">
              {navLinks.map((link) => (
                <Link
                  key={link.name}
                  to={link.path}
                  className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors ${
                    location.pathname === link.path
                      ? 'border-primary text-gray-900'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {link.name}
                </Link>
              ))}
            </div>

            <div className="flex items-center space-x-2 text-sm bg-gray-50 px-3 py-1.5 rounded-full border border-gray-200">
              <span className={`h-2.5 w-2.5 rounded-full ${serverStatus === 'online' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]'}`}></span>
              <span className="text-gray-600 font-medium hidden xs:inline-block">
                {serverStatus === 'checking' ? 'Checking...' : (serverStatus === 'online' ? 'Online' : 'Offline')}
              </span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Mobile menu (simplified) */}
      <div className="sm:hidden border-t border-gray-100 bg-gray-50 flex justify-around py-2">
        {navLinks.map((link) => (
          <Link
            key={link.name}
            to={link.path}
            className={`text-xs font-medium px-3 py-2 rounded-md ${
              location.pathname === link.path ? 'bg-primary text-white shadow-sm' : 'text-gray-600'
            }`}
          >
            {link.name}
          </Link>
        ))}
      </div>
    </nav>
  );
};

export default Navbar;
