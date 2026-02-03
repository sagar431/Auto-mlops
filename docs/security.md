# Security

This document covers the security features and best practices for Auto-MLOps.

## Authentication

### API Key Authentication

All API endpoints (except `/health`) require authentication via API key.

#### Creating API Keys

```bash
# Via CLI
mlops-agent admin create-key --name "Production Key"

# Output:
# API Key: mlops_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# IMPORTANT: Save this key now! It won't be shown again.
```

#### Using API Keys

```bash
# In HTTP requests
curl -H "X-API-Key: mlops_xxxxx" http://localhost:8000/run

# In Python SDK
from sdk import MLOpsClient
client = MLOpsClient(api_key="mlops_xxxxx")

# Via environment variable
export MLOPS_API_KEY=mlops_xxxxx
mlops-agent "Set up pipeline"
```

### Key Management

```bash
# List all keys
mlops-agent admin list-keys

# Revoke a key
mlops-agent admin revoke-key --key-id <key-id>

# Create key with expiration
mlops-agent admin create-key --name "Temp Key" --expires-in-days 30
```

## Authorization

### Role-Based Access Control (RBAC)

Two roles are supported:

| Role | Permissions |
|------|-------------|
| **User** | Run queries, view own sessions, read metrics |
| **Admin** | All user permissions + manage users/keys |

#### Creating Users

```bash
# Create regular user
mlops-agent admin create-user --username john --email john@example.com

# Create admin user
mlops-agent admin create-user --username admin --email admin@example.com --admin
```

## Rate Limiting

Requests are rate-limited to prevent abuse:

- **Default**: 100 requests per minute per API key
- **Configurable** via environment variable

```bash
# Set custom rate limit
export RATE_LIMIT_REQUESTS=200  # requests per minute
```

When rate limited, you'll receive:
```json
{
  "detail": "Rate limit exceeded. Try again in 30 seconds.",
  "retry_after": 30
}
```

## CORS Configuration

By default, CORS is restricted. Configure allowed origins:

```bash
# .env
CORS_ALLOWED_ORIGINS=https://your-frontend.com,https://app.example.com
```

For development:
```bash
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

## Secrets Management

### Environment Variables

Store sensitive values in environment variables:

```bash
# .env (never commit this file!)
GOOGLE_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

### HashiCorp Vault (Optional)

For production, use Vault for secrets:

```bash
# .env
VAULT_ADDR=https://vault.example.com
VAULT_TOKEN=hvs.xxxxx
```

The application will automatically read secrets from Vault if configured.

## Database Security

### Connection Security

Always use SSL for PostgreSQL:

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@host/db?ssl=require
```

### Password Hashing

User passwords are hashed using bcrypt with a work factor of 12.

### API Key Storage

API keys are stored as SHA-256 hashes. The raw key is only shown once at creation time.

## Best Practices

### 1. Use Strong API Keys

API keys are generated with 256 bits of entropy. Never:
- Share keys in code repositories
- Use the same key for multiple environments
- Store keys in plain text

### 2. Rotate Keys Regularly

```bash
# Create new key
mlops-agent admin create-key --name "Production Key v2"

# Update your applications to use new key

# Revoke old key
mlops-agent admin revoke-key --key-id <old-key-id>
```

### 3. Use Separate Keys per Environment

```bash
mlops-agent admin create-key --name "Development"
mlops-agent admin create-key --name "Staging"
mlops-agent admin create-key --name "Production"
```

### 4. Monitor Key Usage

Check the API logs for unusual activity:

```bash
# View recent logs
curl -H "X-API-Key: xxx" http://localhost:8000/logs?level=warning
```

### 5. Set Key Expiration

For temporary access:

```bash
mlops-agent admin create-key --name "Contractor Access" --expires-in-days 30
```

## Security Headers

The API server sets these security headers:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000` (when using HTTPS)

## Reporting Security Issues

If you discover a security vulnerability, please email security@example.com instead of opening a public issue.
