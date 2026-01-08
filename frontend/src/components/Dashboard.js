import React, { useState } from 'react';

const Dashboard = () => {
  const [query, setQuery] = useState('');
  const [projectPath, setProjectPath] = useState('');
  const [sessions, setSessions] = useState([]);
  const [isRunning, setIsRunning] = useState(false);

  const handleRunAgent = async () => {
    if (!query.trim()) return;

    setIsRunning(true);
    try {
      const response = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          project_path: projectPath || null,
          accuracy_threshold: 0.85
        })
      });
      const data = await response.json();
      setSessions(prev => [...prev, { id: data.session_id, query, status: 'started' }]);
      setQuery('');
    } catch (error) {
      console.error('Failed to start agent:', error);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'system-ui, sans-serif' }}>
      <h1>MLOps Agent Dashboard</h1>

      <div style={{ marginBottom: '20px' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your ML pipeline query..."
          style={{ width: '400px', padding: '10px', marginRight: '10px' }}
        />
        <input
          type="text"
          value={projectPath}
          onChange={(e) => setProjectPath(e.target.value)}
          placeholder="Project path (optional)"
          style={{ width: '250px', padding: '10px', marginRight: '10px' }}
        />
        <button
          onClick={handleRunAgent}
          disabled={isRunning || !query.trim()}
          style={{ padding: '10px 20px' }}
        >
          {isRunning ? 'Starting...' : 'Run Agent'}
        </button>
      </div>

      <h2>Sessions</h2>
      {sessions.length === 0 ? (
        <p>No sessions yet. Run the agent to get started.</p>
      ) : (
        <ul>
          {sessions.map((session) => (
            <li key={session.id}>
              <strong>{session.query}</strong> - Status: {session.status}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Dashboard;
