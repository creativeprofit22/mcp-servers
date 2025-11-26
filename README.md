# MCP Servers

Custom MCP (Model Context Protocol) servers for Agent-Girl.

## Available Servers

| Server | Description | Tokens | Tools |
|--------|-------------|--------|-------|
| [kie-seedream](./kie-seedream) | KIE.ai Seedream v4 image generation + Cloudinary | ~2,635 | 7 |

## Installation

Each server has its own `requirements.txt`. Install dependencies:

```bash
cd kie-seedream
pip install -r requirements.txt
```

## Configuration

Servers are configured in Agent-Girl's `mcpServers.ts`. See each server's README for required environment variables.

## Adding New Servers

1. Create a new directory with the server name
2. Add `src/server.py` (or equivalent)
3. Add `requirements.txt` with dependencies
4. Add `README.md` with documentation
5. Add `METRICS.md` with token cost analysis
