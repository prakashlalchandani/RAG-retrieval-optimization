import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, Sparkles } from 'lucide-react';

const ChatWindow = ({ messages, isLoading }) => {
  const messagesEndRef = useRef(null);
  
  // Get username from local storage
  const username = localStorage.getItem('username') || 'User'; 

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-8 flex flex-col scroll-smooth custom-scrollbar">
      
      {/* PREMIUM WELCOME SCREEN */}
      {messages.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center animate-in fade-in zoom-in-95 duration-500">
          <div className="w-14 h-14 bg-zinc-100 dark:bg-zinc-800/50 rounded-2xl flex items-center justify-center mb-6 shadow-sm border border-zinc-200 dark:border-zinc-700/50">
            <Sparkles className="w-7 h-7 text-zinc-400 dark:text-zinc-500" />
          </div>
          <h2 className="text-3xl md:text-4xl font-semibold text-zinc-800 dark:text-zinc-100 mb-3 tracking-tight text-center">
            Hi, <span className="capitalize text-transparent bg-clip-text bg-gradient-to-r from-zinc-500 to-zinc-800 dark:from-zinc-400 dark:to-white">{username}</span>
          </h2>
          <p className="text-zinc-500 dark:text-zinc-400 text-lg text-center max-w-md">
            Ready to audit your documents? Upload an agreement to get started.
          </p>
        </div>
      )}

      {/* CHAT BUBBLES */}
      {messages.length > 0 && (
        <div className="space-y-8 w-full">
          {messages.map((msg, index) => (
            <div key={index} className={`flex gap-4 md:gap-6 w-full max-w-4xl mx-auto ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              
              {msg.role === 'ai' && (
                <div className="w-8 h-8 md:w-10 md:h-10 rounded-full bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 flex items-center justify-center shrink-0">
                  <Bot className="w-5 h-5 text-zinc-600 dark:text-zinc-300" />
                </div>
              )}

              <div className={`text-[15px] leading-relaxed transition-colors duration-300 ${
                msg.role === 'user' 
                  ? 'max-w-[75%] bg-zinc-800 text-white dark:bg-zinc-200 dark:text-zinc-900 px-5 py-3.5 rounded-3xl rounded-tr-sm shadow-sm' 
                  : msg.role === 'system' 
                    ? (msg.isError ? 'bg-red-50 text-red-600 dark:bg-red-900/10 dark:text-red-400 border border-red-100 dark:border-red-900/30 px-4 py-2 rounded-xl w-full text-center text-sm' : 'text-zinc-500 dark:text-zinc-400 w-full text-center text-xs font-medium tracking-wide uppercase') 
                    : 'max-w-[85%] text-zinc-800 dark:text-zinc-200 pt-1.5'
              }`}>
                {msg.role === 'ai' ? (
                  <div className="prose prose-zinc dark:prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-zinc-900 prose-pre:text-zinc-100 prose-pre:border prose-pre:border-zinc-800 prose-pre:rounded-xl">
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                ) : (
                  msg.text
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      
      {/* ⏳ PREMIUM SKELETON LOADER */}
      {isLoading && (
        <div className="flex gap-4 md:gap-6 w-full max-w-4xl mx-auto justify-start mt-8 animate-in fade-in duration-300">
           {/* Pulsing Avatar */}
           <div className="w-8 h-8 md:w-10 md:h-10 rounded-full bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 flex items-center justify-center shrink-0 animate-pulse">
              <Bot className="w-5 h-5 text-zinc-400 dark:text-zinc-500" />
            </div>
            
          {/* Shimmering Text Blocks */}
          <div className="pt-2 flex flex-col gap-3 w-full max-w-[60%]">
            <div className="h-4 bg-zinc-200 dark:bg-zinc-800 rounded-md w-full animate-pulse"></div>
            <div className="h-4 bg-zinc-200 dark:bg-zinc-800 rounded-md w-[85%] animate-pulse"></div>
            <div className="h-4 bg-zinc-200 dark:bg-zinc-800 rounded-md w-[60%] animate-pulse"></div>
          </div>
        </div>
      )}
      
      <div ref={messagesEndRef} className="h-4 shrink-0" />
    </div>
  );
};

export default ChatWindow;