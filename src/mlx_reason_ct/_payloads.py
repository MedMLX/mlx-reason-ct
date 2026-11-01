"""Pinned JSON asset schemas; host tooling need not load the native runtime."""

from typing import NotRequired, TypedDict

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None


class ArithmeticTable(TypedDict):
    file: str


class ArithmeticManifest(TypedDict):
    tables: dict[str, ArithmeticTable]


class EncodedTable(TypedDict):
    dtype: str
    codec: str
    decoded_bytes: int
    shape: list[int]


class KernelSpec(TypedDict):
    inputs: list[str]
    outputs: list[str]
    header: str
    source: str
    atomic_outputs: NotRequired[bool]


class BundleManifest(TypedDict):
    engine: str
    revision: str
    dtype: str
    conversion: str
    shards: list[str]
    excluded_source_tensors: list[str]
    exclusion_reason: str
