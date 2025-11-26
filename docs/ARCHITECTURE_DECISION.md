# Architecture Decision: Combined vs Separate MCP Servers

## Question

Should we have one MCP server with multiple tools, or one MCP server per tool/service?

## Current Setup: Combined (1 Server, 7 Tools)

```
kie-seedream/
├── seedream_generate      (720 tokens)
├── seedream_edit          (600 tokens)
├── seedream_check_status  (150 tokens)
├── cloudinary_upload      (440 tokens)
├── cloudinary_list        (225 tokens)
├── cloudinary_get_url     (350 tokens)
├── cloudinary_delete      (150 tokens)
Total: ~2,635 tokens (always loaded together)
```

## Alternative: Separate MCP Servers

```
seedream/           (~1,470 tokens)
├── generate
├── edit
├── check_status

cloudinary/         (~1,165 tokens)
├── upload
├── list
├── get_url
├── delete
```

## Trade-off Analysis

| Approach | Pros | Cons |
|----------|------|------|
| **Combined** | Simpler config, single process, shared dependencies | Always pay ~2,635 tokens even if you only need one service |
| **Separate** | Load only what you need, potential ~1,165 token savings | More processes, more config, duplicated MCP boilerplate |

## Token Savings

If you only need Seedream (no Cloudinary):
- **Combined**: 2,635 tokens (pay for both)
- **Separate**: 1,470 tokens (44% savings)

If you only need Cloudinary (no Seedream):
- **Combined**: 2,635 tokens (pay for both)
- **Separate**: 1,165 tokens (56% savings)

## When Combined Makes Sense

- Typical workflow is: generate image → upload to cloud → manage
- Services are tightly coupled
- Simplicity is prioritized over token optimization

## When Separate Makes Sense

- You often use only one service at a time
- Token budget is critical
- You want granular control over what's loaded
- Different modes need different tools (e.g., Designah needs Seedream, Coder doesn't)

## Recommendation

**Consider splitting if**:
- Agent-Girl modes could benefit from selective loading
- You want to offer Cloudinary as a general-purpose image storage tool (not just for Seedream output)

**Keep combined if**:
- The generate → upload workflow is your primary use case
- Config simplicity is more valuable than ~1,000 token savings

## Request for Input

What's the typical usage pattern?
1. Generate images AND upload to Cloudinary together?
2. Sometimes just generate (save locally)?
3. Sometimes just manage Cloudinary (without generating)?

The answer determines the optimal architecture.
