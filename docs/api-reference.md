# API Reference

The Auto-MLOps API provides RESTful endpoints for running ML pipelines and managing sessions.

## Base URL

```
http://localhost:8000
```

## Authentication

All endpoints (except `/health`) require authentication via API key.

```bash
# Header authentication
curl -H "X-API-Key: your-api-key" http://localhost:8000/run
```

## Endpoints

### Health Check

```
GET /health
```

Check if the API server is running.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Run Agent

```
POST /run
```

Start a new agent session.

**Request Body:**
```json
{
  "query": "Set up MLOps pipeline for my project",
  "project_path": "/path/to/project",
  "accuracy_threshold": 0.85
}
```

**Response:**
```json
{
  "session_id": "abc123",
  "status": "running",
  "message": "Agent session started"
}
```

### Get Session Status

```
GET /status/{session_id}
```

Get the current status of an agent session.

**Response:**
```json
{
  "session_id": "abc123",
  "status": "completed",
  "query": "Set up MLOps pipeline",
  "result": "Pipeline created successfully...",
  "steps_completed": 5,
  "steps_total": 5,
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z",
  "error": null
}
```

### List Sessions

```
GET /sessions
```

List past agent sessions.

**Query Parameters:**
- `limit` (int): Maximum number of sessions (default: 10)
- `offset` (int): Pagination offset (default: 0)

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "abc123",
      "query": "Set up MLOps pipeline",
      "status": "completed",
      "started_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 42
}
```

### List Tools

```
GET /tools
```

List available MCP tools.

**Response:**
```json
{
  "tools": [
    {
      "name": "create_hydra_config",
      "description": "Create Hydra configuration file",
      "category": "hydra"
    }
  ]
}
```

### Get Metrics

```
GET /metrics
```

Get system and agent metrics.

**Response:**
```json
{
  "system": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "disk_percent": 60.1
  },
  "agent": {
    "total_sessions": 100,
    "success_rate": 0.95,
    "avg_execution_time": 120.5
  },
  "pipeline": {
    "total_pipelines": 50,
    "tool_usage": {
      "create_hydra_config": 45,
      "init_mlflow_experiment": 42
    }
  }
}
```

## WebSocket

### Real-time Events

```
WS /ws/{session_id}
```

Connect to receive real-time events for a session.

**Events:**
```json
{"type": "status", "data": {"status": "running", "message": "Starting..."}}
{"type": "phase", "data": {"phase": "perception", "message": "Analyzing query..."}}
{"type": "step_start", "data": {"step_id": "1", "tool": "create_hydra_config"}}
{"type": "step_complete", "data": {"step_id": "1", "success": true}}
{"type": "complete", "data": {"status": "completed", "result": "..."}}
```

### Metrics Stream

```
WS /ws/metrics
```

Connect to receive real-time system metrics (updates every 5 seconds).

## Admin Endpoints

### Create User

```
POST /admin/users
```

Create a new user account.

**Request Body:**
```json
{
  "username": "john",
  "email": "john@example.com",
  "password": "secret123",
  "is_admin": false
}
```

### Create API Key

```
POST /admin/keys
```

Create a new API key.

**Request Body:**
```json
{
  "name": "My API Key",
  "user_id": "user-123",
  "expires_in_days": 30
}
```

**Response:**
```json
{
  "key_id": "key-abc",
  "raw_key": "mlops_xxxxx",
  "name": "My API Key",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### List Users

```
GET /admin/users
```

List all users (admin only).

### Revoke API Key

```
DELETE /admin/keys/{key_id}
```

Revoke an API key.

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here",
  "error_code": "INVALID_REQUEST"
}
```

**Common Status Codes:**
- `400` - Bad Request
- `401` - Unauthorized (missing or invalid API key)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

## Rate Limits

- Default: 100 requests per minute
- Configurable via `RATE_LIMIT_REQUESTS` environment variable
