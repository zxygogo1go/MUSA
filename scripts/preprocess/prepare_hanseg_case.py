"""Prepare one HaN-Seg case directory for DIR-MUSA training.

HaN-Seg stores CT images and one binary NRRD mask per OAR. This script reads
the CT and masks, builds the repository-standard prepared arrays:

    <out-root>/images/<case-id>.npy
    <out-root>/seg_o/<case-id>.npy
    <out-root>/seg_b/<case-id>.npy
    <out-root>/metadata/<case-id>.json

The HaN-Seg label map is fixed globally so that missing structures in a single
case do not shift labels for the remaining organs.
"""

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from prepare_case import (
    CT_CLIP,
    TARGET_SHAPE,
    TARGET_SPACING,
    center_crop_or_pad,
    normalize_ct,
    parse_tuple,
    resample_to_spacing,
    save_array,
)


DEFAULT_HANSEG_STRUCTURES: Tuple[str, ...] = (
    "OAR_A_Carotid_L",
    "OAR_A_Carotid_R",
    "OAR_Arytenoid",
    "OAR_Bone_Mandible",
    "OAR_Brainstem",
    "OAR_BuccalMucosa",
    "OAR_Cavity_Oral",
    "OAR_Cochlea_L",
    "OAR_Cochlea_R",
    "OAR_Cricopharyngeus",
    "OAR_Esophagus_S",
    "OAR_Eye_AL",
    "OAR_Eye_AR",
    "OAR_Eye_PL",
    "OAR_Eye_PR",
    "OAR_Glnd_Lacrimal_L",
    "OAR_Glnd_Lacrimal_R",
    "OAR_Glnd_Submand_L",
    "OAR_Glnd_Submand_R",
    "OAR_Glnd_Thyroid",
    "OAR_Glottis",
    "OAR_Larynx_SG",
    "OAR_Lips",
    "OAR_OpticChiasm",
    "OAR_OpticNrv_L",
    "OAR_OpticNrv_R",
    "OAR_Parotid_L",
    "OAR_Parotid_R",
    "OAR_Pituitary",
    "OAR_SpinalCord",
)

DEFAULT_BONE_STRUCTURES: Tuple[str, ...] = ("OAR_Bone_Mandible",)

NRRD_DTYPE_MAP = {
    "signed char": np.int8,
    "int8": np.int8,
    "int8_t": np.int8,
    "uchar": np.uint8,
    "unsigned char": np.uint8,
    "uint8": np.uint8,
    "uint8_t": np.uint8,
    "short": np.int16,
    "short int": np.int16,
    "signed short": np.int16,
    "signed short int": np.int16,
    "int16": np.int16,
    "int16_t": np.int16,
    "ushort": np.uint16,
    "unsigned short": np.uint16,
    "unsigned short int": np.uint16,
    "uint16": np.uint16,
    "uint16_t": np.uint16,
    "int": np.int32,
    "signed int": np.int32,
    "int32": np.int32,
    "int32_t": np.int32,
    "uint": np.uint32,
    "unsigned int": np.uint32,
    "uint32": np.uint32,
    "uint32_t": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


def parse_name_list(value: Optional[str]) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def hanseg_case_id(case_dir_name: str, prefix: str = "hanseg") -> str:
    match = re.fullmatch(r"case_(\d+)", case_dir_name)
    if match:
        return f"{prefix}_{int(match.group(1)):04d}"
    return case_dir_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one HaN-Seg case directory for DIR-MUSA.")
    parser.add_argument("--case-dir", required=True, help="Directory containing case_XX_IMG_CT.nrrd and OAR masks.")
    parser.add_argument("--case-id", default=None, help="Output case ID. Default: case_XX -> hanseg_00XX.")
    parser.add_argument("--case-id-prefix", default="hanseg", help="Prefix used when deriving case IDs.")
    parser.add_argument("--out-root", required=True, help="Output root containing images, seg_o, seg_b, metadata.")
    parser.add_argument("--image-name", default=None, help="CT filename inside --case-dir. Default: *_IMG_CT.nrrd.")
    parser.add_argument(
        "--structure-names",
        default=",".join(DEFAULT_HANSEG_STRUCTURES),
        help="Comma-separated global HaN-Seg structure names in fixed label order.",
    )
    parser.add_argument(
        "--bone-structures",
        default=",".join(DEFAULT_BONE_STRUCTURES),
        help="Comma-separated structure names used to build binary seg_b.",
    )
    parser.add_argument(
        "--include-bone-in-seg-o",
        action="store_true",
        help="Also include bone structures in the multi-label seg_o output.",
    )
    parser.add_argument(
        "--strict-structures",
        action="store_true",
        help="Fail when a structure in --structure-names is missing. Off by default because case_19 lacks chiasm.",
    )
    parser.add_argument(
        "--skip-missing-bone",
        action="store_true",
        help="Continue when a requested bone structure is absent or empty.",
    )
    parser.add_argument(
        "--target-shape",
        default=TARGET_SHAPE,
        type=lambda value: parse_tuple(value, 3, int),
        help="Output array shape as x,y,z. Default: 160,160,192.",
    )
    parser.add_argument(
        "--target-spacing",
        default=TARGET_SPACING,
        type=lambda value: parse_tuple(value, 3, float),
        help="Target spacing in mm as x,y,z. Default: 2,2,2.",
    )
    parser.add_argument(
        "--ct-clip",
        default=CT_CLIP,
        type=lambda value: parse_tuple(value, 2, float),
        help="CT clipping range as min,max. Default: -1024,3000.",
    )
    return parser.parse_args()


def _split_nrrd_header(payload: bytes) -> Tuple[str, memoryview]:
    candidates = []
    for separator in (b"\n\n", b"\r\n\r\n"):
        index = payload.find(separator)
        if index >= 0:
            candidates.append((index, len(separator)))
    if not candidates:
        raise ValueError("Could not find NRRD header terminator")
    index, sep_len = min(candidates, key=lambda item: item[0])
    header = payload[:index].decode("latin1")
    data = memoryview(payload)[index + sep_len :]
    return header, data


def _parse_nrrd_header(header_text: str) -> Dict[str, str]:
    lines = header_text.replace("\r\n", "\n").split("\n")
    if not lines or not lines[0].startswith("NRRD"):
        raise ValueError("Input is not an NRRD file")

    header: Dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        header[key.strip().lower()] = value.strip()
    return header


def _parse_space_directions(value: str) -> Tuple[float, ...]:
    vectors = re.findall(r"\(([^)]*)\)|none", value, flags=re.IGNORECASE)
    spacings: List[float] = []
    for vector in vectors:
        if not vector:
            continue
        values = [float(part.strip()) for part in vector.split(",") if part.strip()]
        spacings.append(float(math.sqrt(sum(item * item for item in values))))
    return tuple(spacings)


def load_nrrd(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, str]]:
    """Load a simple 3D NRRD file as an x,y,z NumPy array.

    The HaN-Seg files use embedded gzip encoding and list axes as x,y,z. NRRD's
    first axis is fastest in memory, so the flat buffer is reshaped with
    Fortran order to keep arrays in the repository's x,y,z convention.
    """

    header_text, encoded_data = _split_nrrd_header(path.read_bytes())
    header = _parse_nrrd_header(header_text)

    dimension = int(header.get("dimension", "0"))
    if dimension != 3:
        raise ValueError(f"{path}: expected 3D NRRD, got dimension={dimension}")

    sizes = tuple(int(part) for part in header["sizes"].split())
    if len(sizes) != 3:
        raise ValueError(f"{path}: expected three sizes, got {sizes}")

    type_key = header["type"].lower()
    if type_key not in NRRD_DTYPE_MAP:
        raise ValueError(f"{path}: unsupported NRRD type {header['type']!r}")
    dtype = np.dtype(NRRD_DTYPE_MAP[type_key])
    if dtype.itemsize > 1:
        endian = header.get("endian", "little").lower()
        if endian not in {"little", "big"}:
            raise ValueError(f"{path}: unsupported endian value {endian!r}")
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")

    encoding = header.get("encoding", "raw").lower()
    if encoding in {"gzip", "gz"}:
        raw_data = gzip.decompress(encoded_data)
    elif encoding == "raw":
        raw_data = encoded_data
    else:
        raise ValueError(f"{path}: unsupported NRRD encoding {encoding!r}")

    expected_count = int(np.prod(sizes))
    array = np.frombuffer(raw_data, dtype=dtype, count=expected_count)
    if array.size != expected_count:
        raise ValueError(f"{path}: expected {expected_count} voxels, got {array.size}")
    array = array.reshape(sizes, order="F")
    if not array.dtype.isnative:
        array = array.byteswap().view(array.dtype.newbyteorder("="))

    if "space directions" in header:
        spacing = _parse_space_directions(header["space directions"])
    elif "spacings" in header:
        spacing = tuple(float(part) for part in header["spacings"].split())
    else:
        raise ValueError(f"{path}: missing space directions/spacings")
    if len(spacing) != 3:
        raise ValueError(f"{path}: expected three spacing values, got {spacing}")

    return array, tuple(float(value) for value in spacing), header


def find_ct_path(case_dir: Path, image_name: Optional[str]) -> Path:
    if image_name:
        path = case_dir / image_name
        if not path.is_file():
            raise FileNotFoundError(f"CT image not found: {path}")
        return path

    matches = sorted(case_dir.glob("*_IMG_CT.nrrd"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one *_IMG_CT.nrrd in {case_dir}, found {len(matches)}")
    return matches[0]


def structure_name_from_path(path: Path, case_dir_name: str) -> str:
    name = path.name
    if name.endswith(".seg.nrrd"):
        name = name[: -len(".seg.nrrd")]
    prefix = f"{case_dir_name}_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return name


def discover_structure_paths(case_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for path in sorted(case_dir.glob("*.seg.nrrd")):
        paths[structure_name_from_path(path, case_dir.name)] = path
    if not paths:
        raise ValueError(f"No HaN-Seg OAR mask files found in {case_dir}")
    return paths


def check_geometry(
    array: np.ndarray,
    spacing: Sequence[float],
    reference_shape: Sequence[int],
    reference_spacing: Sequence[float],
    name: str,
) -> None:
    if tuple(array.shape) != tuple(reference_shape):
        raise ValueError(f"{name} shape {array.shape} does not match CT shape {tuple(reference_shape)}")
    if not np.allclose(np.asarray(spacing), np.asarray(reference_spacing), rtol=1e-4, atol=1e-4):
        raise ValueError(f"{name} spacing {tuple(spacing)} does not match CT spacing {tuple(reference_spacing)}")


def merge_oar_masks(
    structure_paths: Dict[str, Path],
    selected_names: Sequence[str],
    reference_shape: Sequence[int],
    reference_spacing: Sequence[float],
    strict_structures: bool,
) -> Tuple[np.ndarray, Dict[str, int], List[str], List[str], List[str]]:
    merged = np.zeros(tuple(reference_shape), dtype=np.int16)
    label_map = {name: label for label, name in enumerate(selected_names, start=1)}
    used: List[str] = []
    missing: List[str] = []
    empty: List[str] = []

    for name, label in label_map.items():
        path = structure_paths.get(name)
        if path is None:
            missing.append(name)
            continue
        mask, spacing, _ = load_nrrd(path)
        check_geometry(mask, spacing, reference_shape, reference_spacing, name)
        mask_bool = mask > 0
        if not np.any(mask_bool):
            empty.append(name)
            continue
        merged[mask_bool] = label
        used.append(name)

    if missing and strict_structures:
        raise ValueError(f"Missing structures: {missing}")
    if not used:
        raise ValueError("All selected OAR structures are missing or empty")
    return merged, label_map, used, missing, empty


def merge_bone_masks(
    structure_paths: Dict[str, Path],
    bone_names: Sequence[str],
    reference_shape: Sequence[int],
    reference_spacing: Sequence[float],
    skip_missing_bone: bool,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    merged = np.zeros(tuple(reference_shape), dtype=np.int16)
    used: List[str] = []
    missing: List[str] = []
    empty: List[str] = []

    for name in bone_names:
        path = structure_paths.get(name)
        if path is None:
            missing.append(name)
            continue
        mask, spacing, _ = load_nrrd(path)
        check_geometry(mask, spacing, reference_shape, reference_spacing, name)
        mask_bool = mask > 0
        if not np.any(mask_bool):
            empty.append(name)
            continue
        merged[mask_bool] = 1
        used.append(name)

    if (missing or empty) and not skip_missing_bone:
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if empty:
            problems.append(f"empty={empty}")
        raise ValueError(f"Bone structures unavailable: {', '.join(problems)}")
    if not used:
        raise ValueError("No non-empty bone structures were found for seg_b")
    return merged, used, missing, empty


def prepare_array(
    array: np.ndarray,
    source_spacing: Sequence[float],
    target_shape: Sequence[int],
    target_spacing: Sequence[float],
    interpolation_order: int,
    pad_value: float,
) -> np.ndarray:
    resampled = resample_to_spacing(array, source_spacing, target_spacing, order=interpolation_order)
    return center_crop_or_pad(resampled, target_shape, pad_value=pad_value)


def write_metadata(out_root: Path, case_id: str, payload: Dict[str, object]) -> Path:
    path = out_root / "metadata" / f"{case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir)
    out_root = Path(args.out_root)
    case_id = args.case_id or hanseg_case_id(case_dir.name, args.case_id_prefix)
    target_shape = tuple(args.target_shape)
    target_spacing = tuple(args.target_spacing)
    ct_clip = tuple(args.ct_clip)

    structure_names = parse_name_list(args.structure_names)
    bone_names = parse_name_list(args.bone_structures)
    if not structure_names:
        raise ValueError("--structure-names produced an empty structure list")
    if not bone_names:
        raise ValueError("--bone-structures produced an empty bone list")

    image_path = find_ct_path(case_dir, args.image_name)
    image_raw, spacing, image_header = load_nrrd(image_path)
    image_prepared = prepare_array(image_raw, spacing, target_shape, target_spacing, 1, ct_clip[0])
    image_prepared = normalize_ct(image_prepared, ct_clip)
    save_array(out_root / "images" / f"{case_id}.npy", image_prepared.astype(np.float32))

    reference_shape = image_raw.shape
    reference_spacing = spacing
    del image_raw, image_prepared

    structure_paths = discover_structure_paths(case_dir)
    if args.include_bone_in_seg_o:
        selected_oar_names = structure_names
    else:
        bone_name_set = set(bone_names)
        selected_oar_names = [name for name in structure_names if name not in bone_name_set]

    seg_o_raw, label_map, used_oar, missing_oar, empty_oar = merge_oar_masks(
        structure_paths,
        selected_names=selected_oar_names,
        reference_shape=reference_shape,
        reference_spacing=reference_spacing,
        strict_structures=args.strict_structures,
    )
    seg_b_raw, used_bone, missing_bone, empty_bone = merge_bone_masks(
        structure_paths,
        bone_names=bone_names,
        reference_shape=reference_shape,
        reference_spacing=reference_spacing,
        skip_missing_bone=args.skip_missing_bone,
    )

    seg_o_prepared = prepare_array(seg_o_raw, reference_spacing, target_shape, target_spacing, 0, 0)
    seg_b_prepared = prepare_array(seg_b_raw, reference_spacing, target_shape, target_spacing, 0, 0)
    seg_o_prepared = np.rint(seg_o_prepared).astype(np.int16)
    seg_b_prepared = (seg_b_prepared > 0).astype(np.int16)
    save_array(out_root / "seg_o" / f"{case_id}.npy", seg_o_prepared)
    save_array(out_root / "seg_b" / f"{case_id}.npy", seg_b_prepared)

    meta = {
        "dataset": "HaN-Seg",
        "case_id": case_id,
        "case_dir": str(case_dir),
        "image_path": str(image_path),
        "input_shape": list(reference_shape),
        "input_spacing": list(reference_spacing),
        "target_shape": list(target_shape),
        "target_spacing": list(target_spacing),
        "ct_clip": list(ct_clip),
        "nrrd_image_type": image_header.get("type"),
        "label_map": label_map,
        "include_bone_in_seg_o": args.include_bone_in_seg_o,
        "structure_names_global": list(structure_names),
        "seg_o_structures_used": used_oar,
        "seg_o_structures_missing": missing_oar,
        "seg_o_structures_empty": empty_oar,
        "bone_structures_requested": bone_names,
        "bone_structures_used": used_bone,
        "bone_structures_missing": missing_bone,
        "bone_structures_empty": empty_bone,
        "seg_o_labels_after_preprocess": [int(value) for value in np.unique(seg_o_prepared)],
        "seg_b_labels_after_preprocess": [int(value) for value in np.unique(seg_b_prepared)],
    }
    meta_path = write_metadata(out_root, case_id, meta)

    print(f"Wrote {out_root / 'images' / f'{case_id}.npy'}")
    print(f"Wrote {out_root / 'seg_o' / f'{case_id}.npy'}")
    print(f"Wrote {out_root / 'seg_b' / f'{case_id}.npy'}")
    print(f"Wrote {meta_path}")
    print(f"Merged {len(used_oar)} OAR structures; used {len(used_bone)} bone structures.")
    if missing_oar:
        print(f"[WARN] Missing OAR structures kept as absent labels: {missing_oar}")


if __name__ == "__main__":
    main()
