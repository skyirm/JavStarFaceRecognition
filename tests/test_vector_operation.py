from pathlib import Path

import numpy as np
import pytest

from vector_operation import compare_vector, get_face_vector_from_file, get_result_from_array

IMG_ROOT = Path(__file__).parent.parent / "validation_images"


def _pick(person: str, index: int = 0) -> Path:
    files = sorted((IMG_ROOT / person).glob("*.jpg"))
    if not files:
        pytest.skip(f"no images for {person}")
    return files[min(index, len(files) - 1)]


def _vector(person: str, index: int = 0) -> np.ndarray:
    vec = get_face_vector_from_file(str(_pick(person, index)))
    assert vec is not None, f"no face found in {person} #{index}"
    return vec


@pytest.fixture(scope="module")
def vec_a0() -> np.ndarray:
    return _vector("天使もえ", 0)


@pytest.fixture(scope="module")
def vec_a1() -> np.ndarray:
    return _vector("天使もえ", 1)


@pytest.fixture(scope="module")
def vec_b() -> np.ndarray:
    return _vector("早坂ひめ", 0)


def test_vector_shape_and_normalized(vec_a0):
    assert vec_a0.shape == (512,)
    assert vec_a0.dtype == np.float32
    assert abs(np.linalg.norm(vec_a0) - 1.0) < 1e-5


def test_same_person_more_similar_than_diff_person(vec_a0, vec_a1, vec_b):
    same = compare_vector(vec_a0, vec_a1)
    diff = compare_vector(vec_a0, vec_b)
    assert -1.0 <= diff <= 1.0
    assert same > diff, f"same={same:.4f} diff={diff:.4f}"
    print(f"same-person similarity: {same:.4f}, diff-person: {diff:.4f}")


def test_compare_vector_semantics(vec_a0):
    assert compare_vector(vec_a0, vec_a0) == pytest.approx(1.0, abs=1e-5)
    assert compare_vector(vec_a0, vec_a0) == compare_vector(vec_a0, vec_a0)


def test_detection_result_fields():
    results = get_result_from_array(
        __import__("cv2").imdecode(
            np.fromfile(str(_pick("天使もえ", 0)), np.uint8), 1
        )
    )
    assert results is not None and len(results) == 1
    face = results[0]
    assert face.kps is not None and face.kps.shape == (5, 2)
    assert face.bbox is not None and len(face.bbox) == 4
    assert face.det_score >= 0.7
