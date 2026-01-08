const API_BASE = 'http://localhost:8000';

export const api = {
  async runAgent(query, projectPath = null, accuracyThreshold = 0.85) {
    const response = await fetch(`${API_BASE}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        project_path: projectPath,
        accuracy_threshold: accuracyThreshold
      })
    });
    return response.json();
  },

  async getStatus(sessionId) {
    const response = await fetch(`${API_BASE}/status/${sessionId}`);
    return response.json();
  },

  async getSessions(limit = 20) {
    const response = await fetch(`${API_BASE}/sessions?limit=${limit}`);
    return response.json();
  },

  async getHealth() {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
  },

  connectWebSocket(sessionId, onMessage) {
    const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    ws.onmessage = (event) => onMessage(JSON.parse(event.data));
    return ws;
  }
};
