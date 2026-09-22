'use client';

import { useState, useEffect, useRef } from 'react';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: any[];
};

export default function SimulatorPage() {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [language, setLanguage] = useState('en-IN');
  const [isThinking, setIsThinking] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket('ws://localhost:8001/ws/voice');

    ws.onopen = () => {
      console.log('Connected to voice gateway');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received:', data);

      if (data.type === 'ready') {
        console.log('Session ready:', data.session_id);
      } else if (data.type === 'thinking') {
        setIsThinking(true);
      } else if (data.type === 'response') {
        setIsThinking(false);
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.text,
          timestamp: new Date().toISOString(),
          citations: data.citations
        }]);
      } else if (data.type === 'error') {
        console.error('Error:', data.message);
        setIsThinking(false);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };

    ws.onclose = () => {
      console.log('Disconnected');
      setConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current || !connected) return;

    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    }]);

    // Send to server
    wsRef.current.send(JSON.stringify({
      type: 'text',
      text: input,
      language: language
    }));

    setInput('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-lg">
        <div className="container mx-auto">
          <h1 className="text-2xl font-bold">SVKM NMIMS Voice Assistant</h1>
          <p className="text-sm opacity-90">Multilingual Admissions Support</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto p-4 max-w-4xl">
        {/* Status */}
        <div className="mb-4 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-gray-600">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="px-3 py-1 border rounded text-sm"
          >
            <option value="en-IN">English (India)</option>
            <option value="hi-IN">हिंदी (Hindi)</option>
            <option value="mr-IN">मराठी (Marathi)</option>
          </select>
        </div>

        {/* Messages */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-4 h-[500px] overflow-y-auto">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-20">
              <p className="text-lg mb-2">👋 Welcome!</p>
              <p>Ask me about courses, admissions, fees, or facilities.</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}
            >
              <div
                className={`inline-block max-w-[80%] px-4 py-2 rounded-lg ${
                  msg.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                <p>{msg.content}</p>
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-300 text-xs opacity-75">
                    <p className="font-semibold">Sources:</p>
                    {msg.citations.map((c, i) => (
                      <p key={i}>• {c.text.substring(0, 50)}...</p>
                    ))}
                  </div>
                )}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {new Date(msg.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}

          {isThinking && (
            <div className="text-left mb-4">
              <div className="inline-block bg-gray-100 px-4 py-2 rounded-lg">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="bg-white rounded-lg shadow-lg p-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your question..."
              className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={!connected}
            />
            <button
              onClick={sendMessage}
              disabled={!connected || !input.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white p-4 text-center text-sm">
        <p>SVKM NMIMS Global University, Dhule</p>
        <p className="opacity-75">Voice Assistant v1.0.0</p>
      </footer>
    </div>
  );
}
