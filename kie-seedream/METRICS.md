# KIE MCP Server - Leanness Metrics

## Token Cost Analysis

When this MCP server is loaded into an LLM context, the tool schemas consume tokens.

### Per-Tool Token Cost

| Tool | Tokens | Purpose |
|------|--------|---------|
| `seedream_generate` | ~720 | Text-to-image generation |
| `seedream_edit` | ~600 | Image editing with references |
| `seedream_check_status` | ~150 | Poll task status |
| `cloudinary_upload` | ~440 | Upload images |
| `cloudinary_list` | ~225 | List images |
| `cloudinary_get_url` | ~350 | Get transformed URLs |
| `cloudinary_delete` | ~150 | Delete images |
| **Total** | **~2,635** | All 7 tools |

### Context Cost

| Component | Tokens |
|-----------|--------|
| Tool schemas | 2,635 |
| System prompt hints | ~50 |
| MCP framework overhead | ~100 |
| **Total per session** | **~2,785** |

This cost is paid **once per new session**, not per message.

## Code Metrics

### Size

| Metric | Value |
|--------|-------|
| Total lines | 1,081 |
| Meaningful lines | 826 |
| File size | 38.7 KB |
| Dependencies | 3 |

### Dependencies

```
mcp>=1.0.0        # MCP SDK
httpx>=0.25.0     # Async HTTP
cloudinary>=1.36.0 # Image storage
```

### Code Breakdown

| Component | Lines | Purpose |
|-----------|-------|---------|
| KieApiClient | 224 | KIE.ai API integration |
| CloudinaryClient | 173 | Image storage |
| Tool definitions | 247 | MCP tool schemas |
| Tool handlers | 195 | Tool execution logic |
| Utilities | 95 | Helpers, extraction |
| Config/Main | 66 | Setup |

## Efficiency Comparison

### vs Full MCP Servers

Typical MCP servers: 5,000-10,000+ tokens
This server: **2,635 tokens** (50-75% smaller)

### Token-per-Tool Efficiency

Average: **376 tokens/tool**
Most efficient: `cloudinary_delete` (150 tokens)
Most feature-rich: `seedream_generate` (720 tokens)

## Optimization Notes

### Why Seedream Tools Are Larger

The `seedream_generate` and `seedream_edit` tools have detailed schemas because:
- 9 `image_size` options (enums)
- 3 `image_resolution` options
- Multiple optional parameters with descriptions
- Rich descriptions for better LLM understanding

This is intentional - better schemas = better tool usage by the LLM.

### Potential Optimizations

1. **Remove unused tools** - If you don't need Cloudinary, ~1,165 tokens saved
2. **Shorter descriptions** - Could save ~200 tokens (not recommended)
3. **Tool caching** - When Claude API supports it, ~95% cost reduction for repeat sessions

## Repository Size

| State | Size |
|-------|------|
| Source only | 45.7 KB |
| With caches | 39 MB |
| After cleanup | ~50 KB |

The `.gitignore` prevents cache bloat from being committed.
