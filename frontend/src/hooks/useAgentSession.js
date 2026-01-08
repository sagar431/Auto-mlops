import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';

export const useAgentSession = (sessionId) => {
  const [status, setStatus] = useState(null);
  const [events, setEvents] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!sessionId) return;

    const ws = api.connectWebSocket(sessionId, (event) => {
      setEvents(prev => [...prev, event]);
      if (event.type === 'status') {
        setStatus(event.data.status);
      }
    });

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);

    return () => ws.close();
  }, [sessionId]);

  const refreshStatus = useCallback(async () => {
    if (!sessionId) return;
    const data = await api.getStatus(sessionId);
    setStatus(data.status);
    return data;
  }, [sessionId]);

  return { status, events, isConnected, refreshStatus };
};
