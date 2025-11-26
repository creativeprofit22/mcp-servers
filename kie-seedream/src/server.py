#!/usr/bin/env python3
"""
KIE.ai Seedream + Cloudinary MCP Server

This MCP server provides tools for:
- Seedream v4 text-to-image generation
- Seedream v4 image editing (with reference images)
- Cloudinary image upload and management

Environment Variables Required:
- KIE_API_KEY: Your KIE.ai API key
- CLOUDINARY_CLOUD_NAME: Cloudinary cloud name
- CLOUDINARY_API_KEY: Cloudinary API key
- CLOUDINARY_API_SECRET: Cloudinary API secret
- KIE_IMAGE_DIR: Local directory for saving images (optional)
"""

import os
import sys
import json
import time
import base64
import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import httpx

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, CallToolResult
except ImportError:
    print("Error: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Cloudinary imports
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False
    print("Warning: Cloudinary not installed. Run: pip install cloudinary", file=sys.stderr)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("kie-mcp-server")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    kie_api_key: str
    kie_base_url: str = "https://api.kie.ai/api/v1"
    cloudinary_cloud_name: Optional[str] = None
    cloudinary_api_key: Optional[str] = None
    cloudinary_api_secret: Optional[str] = None
    image_dir: str = "./generated-images"
    timeout: int = 120  # seconds
    poll_interval: int = 2  # seconds for checking task status

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("KIE_API_KEY")
        if not api_key:
            raise ValueError("KIE_API_KEY environment variable is required")

        return cls(
            kie_api_key=api_key,
            kie_base_url=os.environ.get("KIE_BASE_URL", "https://api.kie.ai/api/v1"),
            cloudinary_cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
            cloudinary_api_key=os.environ.get("CLOUDINARY_API_KEY"),
            cloudinary_api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
            image_dir=os.environ.get("KIE_IMAGE_DIR", "./generated-images"),
            timeout=int(os.environ.get("KIE_TIMEOUT", "120")),
            poll_interval=int(os.environ.get("KIE_POLL_INTERVAL", "2")),
        )


# =============================================================================
# KIE.ai API Client
# =============================================================================

class KieApiClient:
    """Client for KIE.ai Seedream API"""

    # Seedream models
    MODEL_TEXT_TO_IMAGE = "bytedance/seedream-v4-text-to-image"
    MODEL_EDIT = "bytedance/seedream-v4-edit"

    # Image size options
    IMAGE_SIZES = [
        "square", "square_hd",
        "portrait_4_3", "portrait_3_2", "portrait_9_16",
        "landscape_4_3", "landscape_3_2", "landscape_16_9", "landscape_21_9"
    ]

    # Resolution options (uppercase required by KIE.ai API)
    RESOLUTIONS = ["1K", "2K", "4K"]

    def __init__(self, config: Config):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.kie_api_key}",
            "Content-Type": "application/json"
        }

    async def create_task(
        self,
        model: str,
        prompt: str,
        image_size: str = "landscape_16_9",
        image_resolution: str = "2K",
        max_images: int = 1,
        seed: Optional[int] = None,
        image_urls: Optional[list[str]] = None,
        callback_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create an image generation task"""

        # Build the input object with all generation parameters
        input_params = {
            "prompt": prompt,
            "image_size": image_size,
            "image_resolution": image_resolution,
            "max_images": max_images,
        }

        if seed is not None:
            input_params["seed"] = seed

        if image_urls:
            input_params["image_urls"] = image_urls

        # Build the payload with model at root and params in input object
        payload = {
            "model": model,
            "input": input_params,
        }

        if callback_url:
            payload["callBackUrl"] = callback_url

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.config.kie_base_url}/jobs/createTask",
                headers=self.headers,
                json=payload
            )

            if response.status_code == 401:
                raise Exception("Authentication failed. Check your KIE_API_KEY.")
            elif response.status_code == 402:
                raise Exception("Insufficient credits. Please top up your KIE.ai account.")
            elif response.status_code == 429:
                raise Exception("Rate limited. Please wait and try again.")
            elif response.status_code >= 400:
                raise Exception(f"API error {response.status_code}: {response.text}")

            response_data = response.json()
            if response_data is None:
                raise Exception(f"Empty response body from API (status: {response.status_code})")
            return response_data

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a task"""

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.get(
                f"{self.config.kie_base_url}/jobs/recordInfo",
                headers=self.headers,
                params={"taskId": task_id}
            )

            if response.status_code >= 400:
                raise Exception(f"API error {response.status_code}: {response.text}")

            response_data = response.json()
            if response_data is None:
                raise Exception(f"Empty response body from API (status: {response.status_code})")
            return response_data

    async def wait_for_completion(
        self,
        task_id: str,
        max_wait: int = 120
    ) -> dict[str, Any]:
        """Wait for a task to complete and return results"""

        start_time = time.time()

        while True:
            if time.time() - start_time > max_wait:
                raise TimeoutError(f"Task {task_id} did not complete within {max_wait} seconds")

            result = await self.get_task_status(task_id)

            # Check status at top level or in data object (KIE.ai format)
            status = result.get("status", "").lower()
            if "data" in result and isinstance(result["data"], dict):
                status = result["data"].get("status", status).lower()

            logger.info(f"Task {task_id} status: {status}")

            if status in ["completed", "success", "done", "finished"]:
                return result
            elif status in ["failed", "error", "cancelled"]:
                error_msg = result.get("error", result.get("message", "Unknown error"))
                if "data" in result and isinstance(result["data"], dict):
                    error_msg = result["data"].get("error", error_msg)
                raise Exception(f"Task failed: {error_msg}")

            # Still processing, wait and poll again
            await asyncio.sleep(self.config.poll_interval)

    async def generate_image(
        self,
        prompt: str,
        image_size: str = "landscape_16_9",
        image_resolution: str = "2K",
        max_images: int = 1,
        seed: Optional[int] = None,
        wait_for_result: bool = True,
    ) -> dict[str, Any]:
        """Generate image using Seedream v4 text-to-image"""

        logger.info(f"Generating image with prompt: {prompt[:100]}...")

        task_result = await self.create_task(
            model=self.MODEL_TEXT_TO_IMAGE,
            prompt=prompt,
            image_size=image_size,
            image_resolution=image_resolution,
            max_images=max_images,
            seed=seed,
        )

        # Extract taskId - KIE.ai returns it in data.taskId
        task_id = task_result.get("taskId") or task_result.get("task_id") or task_result.get("id")
        if not task_id and "data" in task_result and isinstance(task_result["data"], dict):
            task_id = task_result["data"].get("taskId") or task_result["data"].get("task_id")

        if not task_id:
            # Some APIs return the result directly
            if "images" in task_result or "image_url" in task_result:
                return task_result
            raise Exception(f"No task ID in response: {task_result}")

        logger.info(f"Task created: {task_id}")

        if wait_for_result:
            return await self.wait_for_completion(task_id)

        return {"taskId": task_id, "status": "pending"}

    async def edit_image(
        self,
        prompt: str,
        image_urls: list[str],
        image_size: str = "landscape_16_9",
        image_resolution: str = "2K",
        max_images: int = 1,
        seed: Optional[int] = None,
        wait_for_result: bool = True,
    ) -> dict[str, Any]:
        """Edit images using Seedream v4 edit (supports multiple reference images)"""

        if not image_urls:
            raise ValueError("At least one image URL is required for editing")

        if len(image_urls) > 10:
            raise ValueError("Maximum 10 reference images allowed")

        logger.info(f"Editing {len(image_urls)} images with prompt: {prompt[:100]}...")

        task_result = await self.create_task(
            model=self.MODEL_EDIT,
            prompt=prompt,
            image_size=image_size,
            image_resolution=image_resolution,
            max_images=max_images,
            seed=seed,
            image_urls=image_urls,
        )

        # Extract taskId - KIE.ai returns it in data.taskId
        task_id = task_result.get("taskId") or task_result.get("task_id") or task_result.get("id")
        if not task_id and "data" in task_result and isinstance(task_result["data"], dict):
            task_id = task_result["data"].get("taskId") or task_result["data"].get("task_id")

        if not task_id:
            if "images" in task_result or "image_url" in task_result:
                return task_result
            raise Exception(f"No task ID in response: {task_result}")

        logger.info(f"Edit task created: {task_id}")

        if wait_for_result:
            return await self.wait_for_completion(task_id)

        return {"taskId": task_id, "status": "pending"}


# =============================================================================
# Cloudinary Client
# =============================================================================

class CloudinaryClient:
    """Client for Cloudinary image management"""

    def __init__(self, config: Config):
        self.config = config
        self.configured = False

        if CLOUDINARY_AVAILABLE and all([
            config.cloudinary_cloud_name,
            config.cloudinary_api_key,
            config.cloudinary_api_secret
        ]):
            cloudinary.config(
                cloud_name=config.cloudinary_cloud_name,
                api_key=config.cloudinary_api_key,
                api_secret=config.cloudinary_api_secret,
                secure=True
            )
            self.configured = True
            logger.info("Cloudinary configured successfully")
        else:
            logger.warning("Cloudinary not configured - missing credentials")

    def upload_from_url(
        self,
        image_url: str,
        folder: str = "agent-girl",
        public_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Upload an image from URL to Cloudinary"""

        if not self.configured:
            raise Exception("Cloudinary not configured. Check credentials.")

        options = {
            "folder": folder,
            "resource_type": "image",
        }

        if public_id:
            options["public_id"] = public_id

        if tags:
            options["tags"] = tags

        result = cloudinary.uploader.upload(image_url, **options)

        return {
            "public_id": result.get("public_id"),
            "url": result.get("secure_url"),
            "width": result.get("width"),
            "height": result.get("height"),
            "format": result.get("format"),
            "bytes": result.get("bytes"),
        }

    def upload_from_file(
        self,
        file_path: str,
        folder: str = "agent-girl",
        public_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Upload a local file to Cloudinary"""

        if not self.configured:
            raise Exception("Cloudinary not configured. Check credentials.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        options = {
            "folder": folder,
            "resource_type": "image",
        }

        if public_id:
            options["public_id"] = public_id

        if tags:
            options["tags"] = tags

        result = cloudinary.uploader.upload(file_path, **options)

        return {
            "public_id": result.get("public_id"),
            "url": result.get("secure_url"),
            "width": result.get("width"),
            "height": result.get("height"),
            "format": result.get("format"),
            "bytes": result.get("bytes"),
        }

    def list_images(
        self,
        folder: Optional[str] = None,
        max_results: int = 30,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """List images in Cloudinary"""

        if not self.configured:
            raise Exception("Cloudinary not configured. Check credentials.")

        options = {
            "resource_type": "image",
            "max_results": max_results,
        }

        if folder:
            options["prefix"] = folder

        if tags:
            # Search by tag
            result = cloudinary.api.resources_by_tag(tags[0], **options)
        else:
            result = cloudinary.api.resources(**options)

        return [
            {
                "public_id": r.get("public_id"),
                "url": r.get("secure_url"),
                "format": r.get("format"),
                "width": r.get("width"),
                "height": r.get("height"),
                "created_at": r.get("created_at"),
            }
            for r in result.get("resources", [])
        ]

    def get_image_url(
        self,
        public_id: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        crop: str = "fill",
        format: str = "auto",
    ) -> str:
        """Get a transformed image URL"""

        if not self.configured:
            raise Exception("Cloudinary not configured. Check credentials.")

        transformations = []

        if width or height:
            transform = {"crop": crop}
            if width:
                transform["width"] = width
            if height:
                transform["height"] = height
            transformations.append(transform)

        url = cloudinary.CloudinaryImage(public_id).build_url(
            transformation=transformations if transformations else None,
            fetch_format=format,
            secure=True,
        )

        return url

    def delete_image(self, public_id: str) -> bool:
        """Delete an image from Cloudinary"""

        if not self.configured:
            raise Exception("Cloudinary not configured. Check credentials.")

        result = cloudinary.uploader.destroy(public_id)
        return result.get("result") == "ok"


# =============================================================================
# Image Utilities
# =============================================================================

async def download_image(url: str, save_path: str) -> str:
    """Download an image from URL and save locally"""

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(response.content)

    return save_path


def generate_filename(prompt: str, extension: str = "png") -> str:
    """Generate a unique filename based on prompt and timestamp"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])

    return f"{timestamp}_{safe_prompt}_{prompt_hash}.{extension}"


# =============================================================================
# MCP Server Implementation
# =============================================================================

# Initialize server
server = Server("kie-seedream-cloudinary")

# Global clients (initialized on startup)
config: Optional[Config] = None
kie_client: Optional[KieApiClient] = None
cloudinary_client: Optional[CloudinaryClient] = None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""

    tools = [
        Tool(
            name="seedream_generate",
            description="""Generate images using Seedream v4 text-to-image.

Parameters:
- prompt (required): Detailed text description for image generation (max 5000 chars)
- image_size: Aspect ratio - square, square_hd, portrait_4_3, portrait_3_2, portrait_9_16, landscape_4_3, landscape_3_2, landscape_16_9, landscape_21_9
- image_resolution: Quality - 1k, 2k, or 4k
- max_images: Number of images to generate (1-6)
- seed: Random seed for reproducibility
- save_locally: Whether to save images to local directory
- upload_to_cloudinary: Whether to upload to Cloudinary

Returns image URLs and optionally Cloudinary URLs.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description for image generation",
                        "maxLength": 5000
                    },
                    "image_size": {
                        "type": "string",
                        "enum": KieApiClient.IMAGE_SIZES,
                        "default": "landscape_16_9"
                    },
                    "image_resolution": {
                        "type": "string",
                        "enum": KieApiClient.RESOLUTIONS,
                        "default": "2K"
                    },
                    "max_images": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 6,
                        "default": 1
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility"
                    },
                    "save_locally": {
                        "type": "boolean",
                        "default": True,
                        "description": "Save images to local directory"
                    },
                    "upload_to_cloudinary": {
                        "type": "boolean",
                        "default": False,
                        "description": "Upload generated images to Cloudinary"
                    },
                    "cloudinary_folder": {
                        "type": "string",
                        "default": "agent-girl/generated",
                        "description": "Cloudinary folder for uploads"
                    },
                    "cloudinary_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to apply to uploaded images"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="seedream_edit",
            description="""Edit images using Seedream v4 with reference images.

Supports up to 10 reference images for composition, style transfer, or derivative editing.

Parameters:
- prompt (required): Edit instructions
- image_urls (required): List of reference image URLs (1-10 images)
- image_size, image_resolution, max_images, seed: Same as generate
- save_locally, upload_to_cloudinary: Output options

Use cases:
- Character consistency across multiple images
- Style transfer from reference
- Combining elements from multiple sources
- Scene editing with reference""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Edit instructions",
                        "maxLength": 5000
                    },
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Reference image URLs (1-10)",
                        "minItems": 1,
                        "maxItems": 10
                    },
                    "image_size": {
                        "type": "string",
                        "enum": KieApiClient.IMAGE_SIZES,
                        "default": "landscape_16_9"
                    },
                    "image_resolution": {
                        "type": "string",
                        "enum": KieApiClient.RESOLUTIONS,
                        "default": "2K"
                    },
                    "max_images": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 6,
                        "default": 1
                    },
                    "seed": {"type": "integer"},
                    "save_locally": {"type": "boolean", "default": True},
                    "upload_to_cloudinary": {"type": "boolean", "default": False},
                    "cloudinary_folder": {"type": "string", "default": "agent-girl/edited"},
                    "cloudinary_tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["prompt", "image_urls"]
            }
        ),
        Tool(
            name="seedream_check_status",
            description="Check the status of a pending Seedream generation task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to check"
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="cloudinary_upload",
            description="""Upload an image to Cloudinary.

Supports uploading from:
- URL (image_url parameter)
- Local file path (file_path parameter)

Returns the Cloudinary URL and metadata.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "URL of image to upload"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Local file path to upload"
                    },
                    "folder": {
                        "type": "string",
                        "default": "agent-girl",
                        "description": "Cloudinary folder"
                    },
                    "public_id": {
                        "type": "string",
                        "description": "Custom public ID (optional)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to apply"
                    }
                }
            }
        ),
        Tool(
            name="cloudinary_list",
            description="List images stored in Cloudinary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Folder to list (optional)"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag (optional)"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 30,
                        "maximum": 100
                    }
                }
            }
        ),
        Tool(
            name="cloudinary_get_url",
            description="""Get a Cloudinary image URL with optional transformations.

Supports resizing, cropping, and format conversion.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "public_id": {
                        "type": "string",
                        "description": "Cloudinary public ID of the image"
                    },
                    "width": {"type": "integer", "description": "Target width"},
                    "height": {"type": "integer", "description": "Target height"},
                    "crop": {
                        "type": "string",
                        "enum": ["fill", "fit", "scale", "crop", "thumb"],
                        "default": "fill"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["auto", "png", "jpg", "webp"],
                        "default": "auto"
                    }
                },
                "required": ["public_id"]
            }
        ),
        Tool(
            name="cloudinary_delete",
            description="Delete an image from Cloudinary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "public_id": {
                        "type": "string",
                        "description": "Cloudinary public ID to delete"
                    }
                },
                "required": ["public_id"]
            }
        ),
    ]

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent] | CallToolResult:
    """Handle tool calls"""

    global config, kie_client, cloudinary_client

    try:
        # Seedream Generate
        if name == "seedream_generate":
            result = await kie_client.generate_image(
                prompt=arguments["prompt"],
                image_size=arguments.get("image_size", "landscape_16_9"),
                image_resolution=arguments.get("image_resolution", "2K"),
                max_images=arguments.get("max_images", 1),
                seed=arguments.get("seed"),
            )

            # Extract image URLs from result
            images = extract_images_from_result(result)

            output = {
                "status": "success",
                "images": images,
                "task_id": result.get("taskId"),
            }

            # Save locally if requested
            if arguments.get("save_locally", True) and images:
                local_paths = []
                for i, img in enumerate(images):
                    if img.get("url"):
                        filename = generate_filename(arguments["prompt"])
                        save_path = os.path.join(config.image_dir, filename)
                        await download_image(img["url"], save_path)
                        local_paths.append(save_path)
                        logger.info(f"Saved image to: {save_path}")
                output["local_paths"] = local_paths

            # Upload to Cloudinary if requested
            if arguments.get("upload_to_cloudinary", False) and cloudinary_client.configured:
                cloudinary_urls = []
                for img in images:
                    if img.get("url"):
                        upload_result = cloudinary_client.upload_from_url(
                            image_url=img["url"],
                            folder=arguments.get("cloudinary_folder", "agent-girl/generated"),
                            tags=arguments.get("cloudinary_tags"),
                        )
                        cloudinary_urls.append(upload_result)
                        logger.info(f"Uploaded to Cloudinary: {upload_result['url']}")
                output["cloudinary"] = cloudinary_urls

            return [TextContent(type="text", text=json.dumps(output, indent=2))]

        # Seedream Edit
        elif name == "seedream_edit":
            result = await kie_client.edit_image(
                prompt=arguments["prompt"],
                image_urls=arguments["image_urls"],
                image_size=arguments.get("image_size", "landscape_16_9"),
                image_resolution=arguments.get("image_resolution", "2K"),
                max_images=arguments.get("max_images", 1),
                seed=arguments.get("seed"),
            )

            images = extract_images_from_result(result)

            output = {
                "status": "success",
                "images": images,
                "task_id": result.get("taskId"),
                "reference_images": arguments["image_urls"],
            }

            # Save locally
            if arguments.get("save_locally", True) and images:
                local_paths = []
                for img in images:
                    if img.get("url"):
                        filename = generate_filename(f"edit_{arguments['prompt']}")
                        save_path = os.path.join(config.image_dir, filename)
                        await download_image(img["url"], save_path)
                        local_paths.append(save_path)
                output["local_paths"] = local_paths

            # Upload to Cloudinary
            if arguments.get("upload_to_cloudinary", False) and cloudinary_client.configured:
                cloudinary_urls = []
                for img in images:
                    if img.get("url"):
                        upload_result = cloudinary_client.upload_from_url(
                            image_url=img["url"],
                            folder=arguments.get("cloudinary_folder", "agent-girl/edited"),
                            tags=arguments.get("cloudinary_tags"),
                        )
                        cloudinary_urls.append(upload_result)
                output["cloudinary"] = cloudinary_urls

            return [TextContent(type="text", text=json.dumps(output, indent=2))]

        # Check Status
        elif name == "seedream_check_status":
            result = await kie_client.get_task_status(arguments["task_id"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # Cloudinary Upload
        elif name == "cloudinary_upload":
            if not cloudinary_client.configured:
                return [TextContent(type="text", text=json.dumps({
                    "error": "Cloudinary not configured. Check credentials."
                }))]

            if arguments.get("image_url"):
                result = cloudinary_client.upload_from_url(
                    image_url=arguments["image_url"],
                    folder=arguments.get("folder", "agent-girl"),
                    public_id=arguments.get("public_id"),
                    tags=arguments.get("tags"),
                )
            elif arguments.get("file_path"):
                result = cloudinary_client.upload_from_file(
                    file_path=arguments["file_path"],
                    folder=arguments.get("folder", "agent-girl"),
                    public_id=arguments.get("public_id"),
                    tags=arguments.get("tags"),
                )
            else:
                return [TextContent(type="text", text=json.dumps({
                    "error": "Either image_url or file_path is required"
                }))]

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # Cloudinary List
        elif name == "cloudinary_list":
            if not cloudinary_client.configured:
                return [TextContent(type="text", text=json.dumps({
                    "error": "Cloudinary not configured"
                }))]

            tags = [arguments["tag"]] if arguments.get("tag") else None
            result = cloudinary_client.list_images(
                folder=arguments.get("folder"),
                max_results=arguments.get("max_results", 30),
                tags=tags,
            )

            return [TextContent(type="text", text=json.dumps({
                "count": len(result),
                "images": result
            }, indent=2))]

        # Cloudinary Get URL
        elif name == "cloudinary_get_url":
            if not cloudinary_client.configured:
                return [TextContent(type="text", text=json.dumps({
                    "error": "Cloudinary not configured"
                }))]

            url = cloudinary_client.get_image_url(
                public_id=arguments["public_id"],
                width=arguments.get("width"),
                height=arguments.get("height"),
                crop=arguments.get("crop", "fill"),
                format=arguments.get("format", "auto"),
            )

            return [TextContent(type="text", text=json.dumps({"url": url}, indent=2))]

        # Cloudinary Delete
        elif name == "cloudinary_delete":
            if not cloudinary_client.configured:
                return [TextContent(type="text", text=json.dumps({
                    "error": "Cloudinary not configured"
                }))]

            success = cloudinary_client.delete_image(arguments["public_id"])

            return [TextContent(type="text", text=json.dumps({
                "deleted": success,
                "public_id": arguments["public_id"]
            }, indent=2))]

        else:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Unknown tool: {name}"
            }))]

    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Error in {name}: {str(e)}")]
        )


def extract_images_from_result(result: dict) -> list[dict]:
    """Extract image URLs from various API response formats"""

    images = []

    # KIE.ai format: data.resultJson contains JSON string with resultUrls array
    if "data" in result and isinstance(result["data"], dict):
        data = result["data"]

        # Handle resultJson (JSON string that needs parsing)
        if "resultJson" in data:
            try:
                result_json = json.loads(data["resultJson"])
                if "resultUrls" in result_json:
                    for url in result_json["resultUrls"]:
                        if isinstance(url, str):
                            images.append({"url": url.strip()})
            except (json.JSONDecodeError, TypeError):
                pass

        # Also check for direct resultUrls in data
        if "resultUrls" in data:
            for url in data["resultUrls"]:
                if isinstance(url, str):
                    images.append({"url": url.strip()})

    # Direct images array
    if "images" in result:
        for img in result["images"]:
            if isinstance(img, str):
                images.append({"url": img})
            elif isinstance(img, dict):
                images.append({
                    "url": img.get("url") or img.get("image_url") or img.get("data"),
                    "width": img.get("width"),
                    "height": img.get("height"),
                })

    # Single image_url
    elif "image_url" in result:
        images.append({"url": result["image_url"]})

    # Result array
    elif "result" in result:
        if isinstance(result["result"], list):
            for item in result["result"]:
                if isinstance(item, str):
                    images.append({"url": item})
                elif isinstance(item, dict):
                    images.append({"url": item.get("url") or item.get("image_url")})
        elif isinstance(result["result"], str):
            images.append({"url": result["result"]})

    # Output field
    elif "output" in result:
        if isinstance(result["output"], list):
            for item in result["output"]:
                if isinstance(item, str):
                    images.append({"url": item})
                elif isinstance(item, dict):
                    images.append({"url": item.get("url")})
                # Skip items that are neither string nor dict (e.g., None)
        elif isinstance(result["output"], str):
            images.append({"url": result["output"]})

    return [img for img in images if img.get("url")]


async def main():
    """Main entry point"""

    global config, kie_client, cloudinary_client

    try:
        # Load configuration
        config = Config.from_env()
        logger.info("Configuration loaded")

        # Initialize clients
        kie_client = KieApiClient(config)
        cloudinary_client = CloudinaryClient(config)

        # Create image directory
        os.makedirs(config.image_dir, exist_ok=True)
        logger.info(f"Image directory: {config.image_dir}")

        # Run server
        logger.info("Starting KIE Seedream + Cloudinary MCP Server...")

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )

    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
