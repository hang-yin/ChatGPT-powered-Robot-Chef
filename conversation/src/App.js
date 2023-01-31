import './App.css';
import React, { useState, useEffect } from 'react';
import "./ChatInterface.css";

function App() {

  useEffect(() => {
    document.title = "Kitchen Assistant Bot";
  }, []);

  const [messages, setMessages] = useState([
    { text: 'Hello, how can I help you today?', sender: 'bot' },
  ]);

  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    setMessages([...messages, { text: inputValue, sender: 'user' }]);
    setInputValue('');
  };

  return (
    <div className="chat-interface">
      <div className="right-side">
        <div className="messages">
          {messages.map((message, i) => (
            <div
              key={i}
              className={`message ${message.sender === 'bot' ? 'received bot-message' : 'sent'}`}
            >
              {message.text}
            </div>
          ))}
        </div>
        <form onSubmit={handleSubmit} className="input-container">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="input"
          />
          <button type="submit" className="submit-button">
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
