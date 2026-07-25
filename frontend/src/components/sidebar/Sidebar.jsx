import React, { useRef, useState, useEffect } from 'react';
import { FileText, Sun, Moon, LogOut, UploadCloud, X, User } from 'lucide-react';
import { useAppContext } from '../../context/AppContext';
import DocumentSelector from './DocumentSelector';

const Sidebar = ({ 
  sidebarOpen, 
  setSidebarOpen, 
  isUploading, 
  handleFileUpload, 
  userDocuments, 
  selectedDocument, 
  setSelectedDocument, 
  handleDeleteDocument 
}) => {
  const { isDarkMode, toggleDarkMode, logout, username } = useAppContext();
  const fileInputRef = useRef(null);
  
  // 1. STATE FOR SMART PROGRESS BAR
  const [progress, setProgress] = useState(0);

  // 2. SIMULATE SMOOTH PROGRESS TO KEEP USER ENGAGED
  useEffect(() => {
    let interval;
    if (isUploading) {
      setProgress(0); // Reset on start
      interval = setInterval(() => {
        setProgress((prev) => {
          // Slows down heavily after 85% to wait for backend processing
          if (prev >= 85) return prev + 0.5; 
          // Jumps quickly initially
          const increment = Math.random() * 15; 
          return Math.min(prev + increment, 85);
        });
      }, 400);
    } else {
      // Complete the circle instantly when upload is actually finished
      setProgress(100);
      const timeout = setTimeout(() => setProgress(0), 500); 
      return () => clearTimeout(timeout);
    }
    return () => clearInterval(interval);
  }, [isUploading]);

  // SVG parameters for the circular progress
  const radius = 8;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <div className={`fixed inset-y-0 left-0 z-40 w-72 md:w-80 bg-zinc-50 dark:bg-[#18181b] border-r border-zinc-200 dark:border-zinc-800/50 flex flex-col transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
      
      {/* Header */}
      <div className="p-5 flex justify-between items-center">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-zinc-900 dark:bg-white rounded-lg flex items-center justify-center shadow-sm">
            <FileText className="w-4 h-4 text-white dark:text-zinc-900" />
          </div>
          <h1 className="text-[17px] font-semibold text-zinc-800 dark:text-zinc-100 tracking-tight">
            FinAI<span className="text-zinc-400 dark:text-zinc-500 font-normal"></span>
          </h1>
        </div>
        
        <div className="flex gap-1">
          <button onClick={toggleDarkMode} className="p-2 rounded-md hover:bg-zinc-200/50 dark:hover:bg-zinc-800 text-zinc-500 transition-colors">
            {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          
          <button 
            onClick={() => setSidebarOpen(false)} 
            className="p-2 rounded-md hover:bg-zinc-200/50 dark:hover:bg-zinc-800 text-zinc-500 transition-colors lg:hidden"
          >
            <X className="w-4 h-4" />
          </button>

          <button onClick={logout} className="p-2 rounded-md hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-900/20 dark:hover:text-red-400 text-zinc-500 transition-colors" title="Logout">
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Area */}
      <div className="px-5 py-4 flex-1 overflow-y-auto custom-scrollbar">
        <h2 className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-widest mb-4">Workspace</h2>
        
        {/* 3. SLEEK UPLOAD BUTTON WITH SVG CIRCULAR PROGRESS */}
        <button 
          onClick={() => fileInputRef.current.click()} 
          disabled={isUploading} 
          className="w-full h-[46px] flex items-center justify-center gap-2 bg-white dark:bg-zinc-800/50 text-zinc-700 dark:text-zinc-300 border border-dashed border-zinc-300 dark:border-zinc-700 rounded-xl hover:bg-zinc-100 hover:border-zinc-400 dark:hover:bg-zinc-800 dark:hover:border-zinc-500 disabled:opacity-80 transition-all duration-300"
        >
          {isUploading ? (
            <div className="flex items-center justify-center animate-in fade-in zoom-in duration-300">
              <svg className="w-5 h-5 transform -rotate-90" viewBox="0 0 20 20">
                {/* Background Track */}
                <circle
                  className="text-zinc-200 dark:text-zinc-700"
                  strokeWidth="2"
                  stroke="currentColor"
                  fill="transparent"
                  r={radius}
                  cx="10"
                  cy="10"
                />
                {/* Animated Progress Indicator */}
                <circle
                  className="text-zinc-800 dark:text-zinc-300 transition-all duration-300 ease-out"
                  strokeWidth="2"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  stroke="currentColor"
                  fill="transparent"
                  r={radius}
                  cx="10"
                  cy="10"
                />
              </svg>
            </div>
          ) : (
            <>
              <UploadCloud className="w-4 h-4 text-zinc-500 shrink-0" />
              <span className="text-sm font-medium">Upload Document</span>
            </>
          )}
        </button>
        <input type="file" accept=".pdf, .docx, .txt" ref={fileInputRef} onChange={handleFileUpload} className="hidden" />

        <div className="mt-8">
          <DocumentSelector 
            userDocuments={userDocuments} 
            selectedDocument={selectedDocument} 
            setSelectedDocument={setSelectedDocument} 
            handleDeleteDocument={handleDeleteDocument} 
          />
        </div>
      </div>

      {/* Bottom Profile Section */}
      <div className="mt-auto p-5 border-t border-zinc-200 dark:border-zinc-800/50 flex items-center gap-3">
        {/* Avatar Circle generating initials */}
        <div className="w-9 h-9 rounded-full bg-blue-600 flex items-center justify-center font-bold text-white shadow-sm shrink-0">
          {username ? username.charAt(0).toUpperCase() : <User className="w-4 h-4" />}
        </div>
        
        {/* Username Display */}
        <div className="flex flex-col min-w-0">
          <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate capitalize">
            {username || 'Loading...'}
          </span>
          <span className="text-xs text-zinc-500 dark:text-zinc-500 truncate">
            Free Plan
          </span>
        </div>
      </div>

    </div>
  );
};

export default Sidebar;