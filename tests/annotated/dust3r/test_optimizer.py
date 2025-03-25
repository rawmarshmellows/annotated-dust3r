from src.annotated.dust3r.dust3r import AnnotatedAsymmetricCroCo3DStereo
from src.annotated.dust3r.optimizer import AnnotatedPointCloudOptimizer
from src.vendored.dust3r.optimizer import PointCloudOptimizer


def test_optimizer_equivalence():
    annotated_model = AnnotatedAsymmetricCroCo3DStereo(
        img_size=224, patch_size=16, output_mode="pts3d", head_type="linear", pos_embed="RoPE100"
    )
    pass
