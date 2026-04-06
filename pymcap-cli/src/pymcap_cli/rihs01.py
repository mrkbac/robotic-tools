"""RIHS01: ROS Interface Hashing Standard v1

- https://roscon.ros.org/2023/talks/ROS_2_Types_On-the-wire_Type_Descriptions_and_Hashing_in_Iron_and_onwards.pdf
- https://github.com/ros-infrastructure/rep/pull/381

"""

import hashlib
import json

from ros_parser.models import Type as _RosType
from ros_parser.ros2_msg import parse_schema_to_definitions

_RIHS01_PREFIX = "RIHS01_"
_RIHS01_PRIMITIVE_IDS: dict[str, int] = {
    "int8": 2,
    "uint8": 3,
    "int16": 4,
    "uint16": 5,
    "int32": 6,
    "uint32": 7,
    "int64": 8,
    "uint64": 9,
    "float32": 10,
    "float64": 11,
    "char": 13,
    "wchar": 14,
    "bool": 15,
    "boolean": 15,
    "byte": 16,
    "octet": 16,
    "string": 17,
    "wstring": 18,
}


def _rihs01_field_type(t: _RosType) -> dict[str, int | str]:
    if t.package_name is not None:
        base_id = 1  # NESTED_TYPE
        nested_type_name = f"{t.package_name}/msg/{t.type_name}"
    else:
        base_id = _RIHS01_PRIMITIVE_IDS.get(t.type_name, 0)
        nested_type_name = ""
        if t.string_upper_bound is not None:
            base_id = 21 if t.type_name == "string" else 22

    string_capacity = t.string_upper_bound or 0
    capacity = 0

    if t.is_array:
        if t.array_size is not None and not t.is_upper_bound:
            type_id, capacity = base_id + 48, t.array_size
        elif t.is_upper_bound:
            type_id, capacity = base_id + 96, t.array_size or 0
        else:
            type_id = base_id + 144
    else:
        type_id = base_id

    return {
        "type_id": type_id,
        "capacity": capacity,
        "string_capacity": string_capacity,
        "nested_type_name": nested_type_name,
    }


def _rihs01_type_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) == 2 and parts[1][0].isupper():
        return f"{parts[0]}/msg/{parts[1]}"
    return name


def _rihs01_individual_type_desc(name: str, msgdef: object) -> dict[str, object]:
    return {
        "type_name": _rihs01_type_name(name),
        "fields": [
            {"name": f.name, "type": _rihs01_field_type(f.type)}
            for f in msgdef.fields  # type: ignore[union-attr]
        ],
    }


def _collect_refs(schema_name: str, canonical: dict[str, object]) -> list[str]:
    visited: set[str] = set()
    queue = [schema_name]
    while queue:
        name = queue.pop()
        if name in canonical:
            resolved = name
        else:
            parts = name.split("/")
            resolved = f"{parts[0]}/{parts[2]}" if len(parts) >= 3 and parts[1] == "msg" else name
        if resolved in visited or resolved not in canonical:
            continue
        visited.add(resolved)
        for f in canonical[resolved].fields:  # type: ignore[union-attr]
            if f.type.package_name is not None:
                ref = f"{f.type.package_name}/msg/{f.type.type_name}"
                if ref not in visited:
                    queue.append(ref)
    visited.discard(schema_name)
    return sorted(visited)


def _find_main_def(schema_name: str, canonical: dict[str, object]) -> tuple[str, object]:
    if schema_name in canonical:
        return schema_name, canonical[schema_name]
    parts = schema_name.split("/")
    if len(parts) >= 3 and parts[1] == "msg":
        alt = f"{parts[0]}/{parts[2]}"
        if alt in canonical:
            return alt, canonical[alt]
    if len(parts) >= 2:
        alt = parts[-1]
        if alt in canonical:
            return alt, canonical[alt]
    raise ValueError(f"Schema {schema_name} not found in definitions")


def compute_rihs01(schema_name: str, schema_data: bytes) -> str:
    """Compute RIHS01 hash for a ros2msg schema per REP-2011."""
    definitions = parse_schema_to_definitions(schema_name, schema_data)

    seen: set[int] = set()
    canonical: dict[str, object] = {}
    for key, msgdef in definitions.items():
        if id(msgdef) not in seen:
            seen.add(id(msgdef))
            canonical[key] = msgdef

    main_name, main_def = _find_main_def(schema_name, canonical)
    ref_names = _collect_refs(main_name, canonical)

    hashable = {
        "type_description": _rihs01_individual_type_desc(main_name, main_def),
        "referenced_type_descriptions": [
            _rihs01_individual_type_desc(n, canonical[n]) for n in ref_names
        ],
    }

    sha256 = hashlib.sha256(json.dumps(hashable, separators=(", ", ": ")).encode()).hexdigest()
    return f"{_RIHS01_PREFIX}{sha256}"
