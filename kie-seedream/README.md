# KIE Seedream MCP Server

Minimal MCP server for [KIE.ai](https://kie.ai) Seedream v4 image generation + Cloudinary storage.

## Stats

| Metric | Value |
|--------|-------|
| Source code | 826 lines |
| Dependencies | 3 |
| Tools | 7 |
| Token cost | ~2,635 tokens |

See [METRICS.md](METRICS.md) for detailed analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `KIE_API_KEY` | Yes | KIE.ai API key |
| `CLOUDINARY_CLOUD_NAME` | No | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | No | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | No | Cloudinary API secret |
| `KIE_IMAGE_DIR` | No | Local save directory |

## Tools

### Image Generation

**`seedream_generate`** - Text-to-image
```json
{
  "prompt": "A cyberpunk cityscape at dusk",
  "image_size": "landscape_16_9",
  "image_resolution": "2K",
  "max_images": 1
}
```

**`seedream_edit`** - Edit with reference images (up to 10)
```json
{
  "prompt": "Transform to wear a red dress",
  "image_urls": ["https://example.com/ref.jpg"],
  "image_size": "portrait_3_2"
}
```

**`seedream_check_status`** - Poll pending tasks

### Cloudinary

- `cloudinary_upload` - Upload from URL or file
- `cloudinary_list` - List images by folder/tag
- `cloudinary_get_url` - Get transformed URLs
- `cloudinary_delete` - Delete images

## Parameters

**image_size** (aspect ratio):
- `square`, `square_hd`
- `portrait_4_3`, `portrait_3_2`, `portrait_9_16`
- `landscape_4_3`, `landscape_3_2`, `landscape_16_9`, `landscape_21_9`

**image_resolution** (quality):
- `1K`, `2K`, `4K` (uppercase required)

## API

- Endpoint: `https://api.kie.ai/api/v1/jobs/createTask`
- Auth: `Authorization: Bearer {API_KEY}`
- Pricing: ~$0.0175/image (3.5 credits)
