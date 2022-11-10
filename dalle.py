#!/usr/bin/env python3

"""Playing with OpenAI."""

import string
from enum import Enum
from pathlib import Path
from typing import Final

import openai
import requests
from tqdm import tqdm
from typer import Typer

from project_secrets import set_openai_key

set_openai_key()

app = Typer()


class ImageSize(str, Enum):
    LARGE = "LARGE"
    MEDIUM = "MEDIUM"
    SMALL = "SMALL"


_IMAGE_SIZES: Final[dict[ImageSize, str]] = {
    ImageSize.LARGE: "1024x1024",
    ImageSize.MEDIUM: "512x512",
    ImageSize.SMALL: "256x256",
}


def _cache_dir(prompt: str, size: ImageSize) -> Path:
    clean_prompt = (
        prompt.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .replace(" ", "-")
    )
    name = f"{clean_prompt}_{_IMAGE_SIZES[size]}"
    cache_dir = Path("dalle-results") / name
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    return cache_dir


def _cache_path(prompt: str, idx: int, size: ImageSize) -> Path:
    return _cache_dir(prompt, size) / f"image-{idx:03d}.jpeg"


def _get_cached_generated_image(prompt: str, idx: int, size: ImageSize) -> Path | None:
    cache_fp = _cache_path(prompt, idx=idx, size=size)
    if not cache_fp.exists():
        return None
    return cache_fp


def _cache_generated_image(
    img_url: str, prompt: str, idx: int, size: ImageSize
) -> Path:
    cache_fp = _cache_path(prompt, idx=idx, size=size)
    img_data = requests.get(img_url).content
    with open(cache_fp, "wb") as handler:
        handler.write(img_data)
    return cache_fp


def image_generation(
    prompt: str, n: int = 1, size: ImageSize = ImageSize.LARGE
) -> list[Path]:
    """Generate images for a given prompt and save them to file.

    Args:
        prompt (str): Prompt for DALL-E
        n (int, optional): Number of images. Defaults to 1.
        size (ImageSize, optional): Size of the images. There is no additional cost to
        generating larger images. Defaults to ImageSize.LARGE.

    Returns:
        list[Path]: List of paths for the saved images.
    """
    results: list[Path] = []
    print(f"Generating images for '{prompt}'")
    for i in tqdm(range(n)):
        if (
            result := _get_cached_generated_image(prompt, idx=i, size=size)
        ) is not None:
            results.append(result)
        else:
            response = openai.Image.create(prompt=prompt, n=1, size=_IMAGE_SIZES[size])
            image_url = response["data"][0]["url"]
            result = _cache_generated_image(image_url, prompt=prompt, idx=i, size=size)
            results.append(result)
    return results


@app.command()
def generate(prompt: str, n: int = 1, size: ImageSize = ImageSize.LARGE) -> None:
    """Generate images for a new prompt.

    Args:
        prompt (str): Custom prompt.
        n (int, optional): Number of images. Defaults to 1.
        size (ImageSize, optional): Size of the images (there is no extra cost for
        larger images). Defaults to ImageSize.LARGE.
    """
    image_generation(prompt, n=n, size=size)
    return None


@app.command()
def examples(n: int = 3) -> None:
    """Generate a few examples.

    Args:
        n (int, optional): Number of images per example prompt. Defaults to 3.
    """
    prompts = (
        "A man standing on a bridge in California.",
        "A cat waving to her mom.",
        "Elvis jumping out of a cake.",
        "Golden retriever riding the space shuttle to the moon.",
    )
    for prompt in prompts:
        image_generation(prompt, n=n, size=ImageSize.LARGE)
    return None


if __name__ == "__main__":
    app()
