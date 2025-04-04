import pickle
from pathlib import Path

import pytest
import torch
from loguru import logger

from src.annotated.dust3r.dust3r import AnnotatedAsymmetricCroCo3DStereo
from src.vendored.dust3r.inference import inference
from src.vendored.dust3r.load_images import LoadConfig, load_images
from src.vendored.dust3r.make_pairs import make_pairs
from src.vendored.dust3r.model import AsymmetricCroCo3DStereo


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--force-regen-predictions", action="store_true", help="Force regeneration of model predictions cache"
    )


@pytest.fixture(scope="module")
def model_predictions(request):
    """Cache model predictions between test runs to avoid recalculation.

    To force regeneration, use: pytest --force-regen-predictions
    """
    # Check if we should force regeneration
    force_regen = request.config.getoption("--force-regen-predictions", default=False)
    logger.info(f"Force regeneration flag: {force_regen}")

    # Check if cached predictions exist
    cache_file = Path(__file__).parent / "test_data" / "cached_predictions.pkl"
    logger.info(f"Cache file path: {cache_file}")

    if cache_file.exists() and not force_regen:
        # Load cached predictions if available
        with open(cache_file, "rb") as f:
            logger.info("Loading cached model predictions")
            return pickle.load(f)

    # Set fixed seed for reproducibility
    logger.info("Generating new model predictions (this may take a moment)...")
    torch.manual_seed(42)

    # Initialize model
    model = AnnotatedAsymmetricCroCo3DStereo.from_pretrained_naver_DUSt3R_ViTLarge_BaseDecoder_512_dpt(
        AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")
    )

    # Create sample inputs
    images_data = load_images(
        [
            str(Path(__file__).parent / "test_data" / "Chateau1.png"),
            str(Path(__file__).parent / "test_data" / "Chateau1 copy.png"),
            str(Path(__file__).parent / "test_data" / "Chateau2.png"),
            str(Path(__file__).parent / "test_data" / "Chateau2 copy.png"),
        ],
        config=LoadConfig(size=512),
    )
    pairs = make_pairs(images_data)
    output = inference(pairs, model, "cpu", batch_size=len(images_data))

    # Prepare results dictionary
    results = {"view1": output["view1"], "view2": output["view2"], "pred1": output["pred1"], "pred2": output["pred2"]}

    # Save to cache file
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, "wb") as f:
        logger.info(f"Saving predictions to cache file: {cache_file}")
        pickle.dump(results, f)

    # Return all necessary data for tests
    return results
