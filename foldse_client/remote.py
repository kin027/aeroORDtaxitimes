from __future__ import annotations

import csv
import importlib.resources as resources
import json
import os
import random
import secrets
import string
from pathlib import Path
from typing import Any, Optional, Union

from foldse_client.crypto import decrypt_json_response, encrypt_json_request, load_public_key

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


DEFAULT_SERVER_URL = "https://foldse-server.onrender.com"


def _server_url(server_url: Optional[str]) -> str:
    return (server_url or os.environ.get("FOLD_SERVER_URL", DEFAULT_SERVER_URL)).rstrip("/")


def _public_key_bytes(public_key_path: Optional[Union[str, Path]]) -> bytes:
    key_path = public_key_path or os.environ.get("FOLD_SERVER_PUBLIC_KEY")
    if key_path:
        return Path(key_path).read_bytes()
    return resources.files("foldse_client").joinpath("server_public.pem").read_bytes()


class Foldse:
    """Drop-in client that trains remotely and predicts locally from returned rules."""

    def __init__(
        self,
        str_attrs: Optional[list[str]] = None,
        num_attrs: Optional[list[str]] = None,
        label: Optional[str] = None,
        pos_val: Optional[str] = None,
        *,
        server_url: Optional[str] = None,
        public_key_path: Optional[Union[str, Path]] = None,
        timeout: int = 120,
    ) -> None:
        self.str_attrs = str_attrs or []
        self.num_attrs = num_attrs or []
        self.label = label
        self.pos_val = pos_val
        self.server_url = _server_url(server_url)
        self.public_key_path = Path(public_key_path) if public_key_path is not None else None
        self.timeout = timeout
        self.flat_rules: Optional[list[tuple[Any, list[Any], list[Any]]]] = None
        self.training_metrics: Optional[dict[str, Any]] = None
        self._obfuscator: Optional[_Obfuscator] = None
        self._asp: Optional[list[str]] = None
        self._public_key = None

    @property
    def attrs(self) -> list[str]:
        return [attr.lower().replace(" ", "_") for attr in self.str_attrs + self.num_attrs]

    @property
    def rule_head(self) -> tuple[Optional[str], str, Optional[str]]:
        label = self.label.lower().replace(" ", "_") if self.label else None
        return (label, "==", self.pos_val)

    @property
    def training_seconds(self) -> Optional[float]:
        if self.training_metrics is None:
            return None
        value = self.training_metrics.get("training_seconds")
        return float(value) if value is not None else None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests is required to use the remote Foldse client")
        if self._public_key is None:
            self._public_key = load_public_key(_public_key_bytes(self.public_key_path))

        envelope, request_key = encrypt_json_request(payload, self._public_key)
        response = requests.post(
            f"{self.server_url}/v1/run",
            json=envelope,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            raise requests.HTTPError(
                f"{exc}. Server response: {detail}",
                response=response,
            ) from exc
        return decrypt_json_response(response.json(), request_key)

    def _config(self) -> dict[str, Any]:
        return {
            "str_attrs": self.str_attrs,
            "num_attrs": self.num_attrs,
            "label": self.label,
            "pos_val": self.pos_val,
        }

    def reset(self) -> None:
        self.flat_rules = None
        self.training_metrics = None
        self._obfuscator = None
        self._asp = None

    def load_data(self, file_name: Union[str, Path], amount: int = -1) -> list[dict[str, Any]]:
        return _load_data_locally(file_name, self.str_attrs, self.num_attrs, self.label, self.pos_val, amount)

    def fit(self, data: list[dict[str, Any]], ratio: float = 0.5, tail: float = 5e-3) -> None:
        _validate_client_config(self.str_attrs, self.num_attrs, self.label, self.pos_val)
        _validate_client_training_data(data, self.attrs)
        _validate_training_parameters(ratio, tail)
        obfuscator = _Obfuscator(self.str_attrs, self.num_attrs, self.label, self.pos_val)
        obfuscated_config = obfuscator.obfuscate_config()
        obfuscated_data = [obfuscator.obfuscate_row(row) for row in data]
        result = self._request(
            {
                "action": "train",
                "config": obfuscated_config,
                "data": obfuscated_data,
                "ratio": ratio,
                "tail": tail,
            }
        )
        deobfuscated_model = obfuscator.deobfuscate_model(result["model"])
        self._obfuscator = obfuscator
        self._load_model_payload(deobfuscated_model, result.get("metrics"))

    def _load_model_payload(self, payload: dict[str, Any], metrics: Optional[dict[str, Any]] = None) -> None:
        self.str_attrs = payload["str_attrs"]
        self.num_attrs = payload["num_attrs"]
        self.label = payload["label"]
        self.pos_val = payload["pos_val"]
        self.flat_rules = [_map_rule(rule) for rule in payload["flat_rules"]]
        self._asp = payload.get("asp")
        self.training_metrics = metrics if metrics is not None else payload.get("training_metrics")

    def _require_rules(self) -> list[tuple[Any, list[Any], list[Any]]]:
        if self.flat_rules is None:
            raise RuntimeError("Model has not been trained or loaded")
        return self.flat_rules

    def classify(self, x: dict[str, Any]) -> bool:
        return _justify_one(self._require_rules(), x, [], self.rule_head) >= 0

    def predict(self, data: list[dict[str, Any]]) -> list[bool]:
        return [self.classify(x) for x in data]

    def asp(self) -> list[str]:
        if self._asp is None:
            self._asp = _decode_rules(self._require_rules())
        return self._asp

    def proof_rules(self, x: dict[str, Any], get_all: bool = False) -> list[str]:
        rules = self._require_rules()
        ret = [str(_explain_data(rules, x))]
        for rule_set in _explain(rules, x, self.rule_head, get_all):
            ret.extend(_decode_rules(rule_set, x))
        return ret

    def proof_trees(self, x: dict[str, Any], get_all: bool = False) -> list[str]:
        rules = self._require_rules()
        ret = [str(_explain_data(rules, x))]
        for rule_set in _explain(rules, x, self.rule_head, get_all):
            ret.extend(_proof_tree(rule_set, x, self.rule_head))
        return ret


class Foldsem:
    """Multi-class FOLD-SE client with remote training and local prediction."""

    def __init__(
        self,
        str_attrs: Optional[list[str]] = None,
        num_attrs: Optional[list[str]] = None,
        label: Optional[str] = None,
        *,
        server_url: Optional[str] = None,
        public_key_path: Optional[Union[str, Path]] = None,
        timeout: int = 120,
    ) -> None:
        self.str_attrs = str_attrs or []
        self.num_attrs = num_attrs or []
        self.label = label
        self.server_url = _server_url(server_url)
        self.public_key_path = Path(public_key_path) if public_key_path is not None else None
        self.timeout = timeout
        self.flat_rules: Optional[list[tuple[Any, list[Any], list[Any]]]] = None
        self.training_metrics: Optional[dict[str, Any]] = None
        self._asp: Optional[list[str]] = None
        self._public_key = None

    @property
    def attrs(self) -> list[str]:
        return [_normalize_attr(attr) for attr in self.str_attrs + self.num_attrs]

    @property
    def training_seconds(self) -> Optional[float]:
        if self.training_metrics is None:
            return None
        value = self.training_metrics.get("training_seconds")
        return float(value) if value is not None else None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests is required to use the remote Foldsem client")
        if self._public_key is None:
            self._public_key = load_public_key(_public_key_bytes(self.public_key_path))

        envelope, request_key = encrypt_json_request(payload, self._public_key)
        response = requests.post(
            f"{self.server_url}/v1/run",
            json=envelope,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            raise requests.HTTPError(
                f"{exc}. Server response: {detail}",
                response=response,
            ) from exc
        return decrypt_json_response(response.json(), request_key)

    def _config(self) -> dict[str, Any]:
        return {
            "str_attrs": self.str_attrs,
            "num_attrs": self.num_attrs,
            "label": self.label,
        }

    def reset(self) -> None:
        self.flat_rules = None
        self.training_metrics = None
        self._asp = None

    def load_data(self, file_name: Union[str, Path], amount: int = -1) -> list[dict[str, Any]]:
        return _load_multiclass_data_locally(file_name, self.str_attrs, self.num_attrs, self.label, amount)

    def fit(self, data: list[dict[str, Any]], ratio: float = 0.5, tail: float = 5e-3) -> None:
        _validate_multiclass_client_config(self.str_attrs, self.num_attrs, self.label)
        _validate_multiclass_training_data(data, self.attrs)
        _validate_training_parameters(ratio, tail)
        obfuscator = _MulticlassObfuscator(self.str_attrs, self.num_attrs, self.label)
        result = self._request(
            {
                "action": "train",
                "algorithm": "foldsem",
                "config": obfuscator.obfuscate_config(),
                "data": [obfuscator.obfuscate_row(row) for row in data],
                "ratio": ratio,
                "tail": tail,
            }
        )
        self._load_model_payload(obfuscator.deobfuscate_model(result["model"]), result.get("metrics"))

    def _load_model_payload(self, payload: dict[str, Any], metrics: Optional[dict[str, Any]] = None) -> None:
        self.str_attrs = payload["str_attrs"]
        self.num_attrs = payload["num_attrs"]
        self.label = payload["label"]
        self.flat_rules = [_map_rule(rule) for rule in payload["flat_rules"]]
        self._asp = payload.get("asp")
        self.training_metrics = metrics if metrics is not None else payload.get("training_metrics")

    def _require_rules(self) -> list[tuple[Any, list[Any], list[Any]]]:
        if self.flat_rules is None:
            raise RuntimeError("Model has not been trained or loaded")
        return self.flat_rules

    def classify(self, x: dict[str, Any]) -> Optional[str]:
        return _classify_multiclass(self._require_rules(), x, _normalize_attr(self.label or "label"))

    def predict(self, data: list[dict[str, Any]]) -> list[Optional[str]]:
        return [self.classify(x) for x in data]

    def asp(self) -> list[str]:
        if self._asp is None:
            self._asp = _decode_rules(self._require_rules())
        return self._asp


def _map_item(item: list[Any]) -> tuple[Any, Any, Any]:
    return (item[0], item[1], item[2])


def _map_rule(rule: Any) -> tuple[Any, list[Any], list[Any]]:
    if isinstance(rule, dict):
        return (
            tuple(rule["head"]),
            [_map_item(item) for item in rule["main_items"]],
            [tuple(item) for item in rule["ab_items"]],
        )
    return (
        tuple(rule[0]),
        [_map_item(item) for item in rule[1]],
        [tuple(item) for item in rule[2]],
    )


def _rule_to_json(rule: tuple[Any, list[Any], list[Any]]) -> dict[str, Any]:
    return {"head": rule[0], "main_items": rule[1], "ab_items": rule[2]}


def _normalize_attr(attr: str) -> str:
    return attr.lower().replace(" ", "_")


def _token(prefix: str) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return prefix + "_" + "".join(secrets.choice(alphabet) for _ in range(16))


def _load_data_locally(
    file_name: Union[str, Path],
    str_attrs: list[str],
    num_attrs: list[str],
    label: Optional[str],
    pos_val: Optional[str],
    amount: int = -1,
) -> list[dict[str, Any]]:
    if amount == 0 or amount < -1:
        raise ValueError("amount must be -1 or a positive integer")
    if label is None or pos_val is None:
        raise ValueError("label and pos_val are required")
    _validate_client_config(str_attrs, num_attrs, label, pos_val)

    ret = []
    with Path(file_name).open("r", encoding="utf-8", newline="") as data_file:
        reader = csv.reader(data_file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV file is empty") from exc

        _validate_csv_header(header, str_attrs, num_attrs, label)
        str_attr_idx = [i for i, s in enumerate(header) if s in str_attrs]
        num_attr_idx = [i for i, n in enumerate(header) if n in num_attrs]
        str_names = [_normalize_attr(attr) for attr in str_attrs]
        num_names = [_normalize_attr(attr) for attr in num_attrs]
        try:
            lbl_idx = header.index(label)
        except ValueError as exc:
            raise ValueError(f"CSV label column not found: {label}") from exc

        for row in reader:
            if amount == 0:
                break
            line_dict = {}
            for i, value in enumerate(row):
                if value == "":
                    continue
                if i in num_attr_idx:
                    key = num_names[num_attr_idx.index(i)]
                    try:
                        line_dict[key] = float(value)
                    except ValueError:
                        line_dict[key] = value
                elif i in str_attr_idx:
                    key = str_names[str_attr_idx.index(i)]
                    line_dict[key] = value
            line_dict["label"] = row[lbl_idx] == pos_val
            ret.append(line_dict)
            amount -= 1
    _validate_client_training_data(ret, [_normalize_attr(attr) for attr in str_attrs + num_attrs])
    return ret


def _load_multiclass_data_locally(
    file_name: Union[str, Path],
    str_attrs: list[str],
    num_attrs: list[str],
    label: Optional[str],
    amount: int = -1,
) -> list[dict[str, Any]]:
    if amount == 0 or amount < -1:
        raise ValueError("amount must be -1 or a positive integer")
    _validate_multiclass_client_config(str_attrs, num_attrs, label)

    ret = []
    with Path(file_name).open("r", encoding="utf-8", newline="") as data_file:
        reader = csv.reader(data_file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV file is empty") from exc

        _validate_csv_header(header, str_attrs, num_attrs, label or "")
        str_attr_idx = [i for i, s in enumerate(header) if s in str_attrs]
        num_attr_idx = [i for i, n in enumerate(header) if n in num_attrs]
        str_names = [_normalize_attr(attr) for attr in str_attrs]
        num_names = [_normalize_attr(attr) for attr in num_attrs]
        lbl_idx = header.index(label)

        for row in reader:
            if amount == 0:
                break
            line_dict = {}
            for i, value in enumerate(row):
                if value == "":
                    continue
                if i in num_attr_idx:
                    key = num_names[num_attr_idx.index(i)]
                    try:
                        line_dict[key] = float(value)
                    except ValueError:
                        line_dict[key] = value
                elif i in str_attr_idx:
                    key = str_names[str_attr_idx.index(i)]
                    line_dict[key] = value
            line_dict["label"] = row[lbl_idx]
            ret.append(line_dict)
            amount -= 1
    _validate_multiclass_training_data(ret, [_normalize_attr(attr) for attr in str_attrs + num_attrs])
    return ret


def _validate_client_config(
    str_attrs: list[str],
    num_attrs: list[str],
    label: Optional[str],
    pos_val: Optional[str],
) -> None:
    if not isinstance(str_attrs, list) or not isinstance(num_attrs, list):
        raise ValueError("str_attrs and num_attrs must be lists")
    if not all(isinstance(attr, str) and attr.strip() for attr in str_attrs + num_attrs):
        raise ValueError("str_attrs and num_attrs must contain only non-empty strings")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    if not isinstance(pos_val, str) or not pos_val.strip():
        raise ValueError("pos_val must be a non-empty string")
    normalized_attrs = [_normalize_attr(attr) for attr in str_attrs + num_attrs]
    if len(normalized_attrs) != len(set(normalized_attrs)):
        raise ValueError("configured feature names must be unique after normalization")
    if _normalize_attr(label) in set(normalized_attrs):
        raise ValueError("label must not also be listed as a feature attribute")


def _validate_multiclass_client_config(
    str_attrs: list[str],
    num_attrs: list[str],
    label: Optional[str],
) -> None:
    if not isinstance(str_attrs, list) or not isinstance(num_attrs, list):
        raise ValueError("str_attrs and num_attrs must be lists")
    if not all(isinstance(attr, str) and attr.strip() for attr in str_attrs + num_attrs):
        raise ValueError("str_attrs and num_attrs must contain only non-empty strings")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    normalized_attrs = [_normalize_attr(attr) for attr in str_attrs + num_attrs]
    if len(normalized_attrs) != len(set(normalized_attrs)):
        raise ValueError("configured feature names must be unique after normalization")
    if _normalize_attr(label) in set(normalized_attrs):
        raise ValueError("label must not also be listed as a feature attribute")


def _validate_training_parameters(ratio: Any, tail: Any) -> None:
    for name, value in (("ratio", ratio), ("tail", tail)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number between 0 and 1")
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be a number between 0 and 1")


def _validate_csv_header(header: list[str], str_attrs: list[str], num_attrs: list[str], label: str) -> None:
    missing_attrs = [attr for attr in str_attrs + num_attrs if attr not in header]
    if missing_attrs:
        preview = ", ".join(missing_attrs[:5])
        suffix = "" if len(missing_attrs) <= 5 else f", ... ({len(missing_attrs)} total)"
        raise ValueError(f"CSV is missing configured feature columns: {preview}{suffix}")
    if label not in header:
        raise ValueError(f"CSV label column not found: {label}")


def _validate_client_training_data(data: Any, attrs: list[str]) -> None:
    if not isinstance(data, list):
        raise ValueError("data must be a list of row objects")
    if not data:
        raise ValueError("data must contain at least one row")
    positive_count = 0
    negative_count = 0
    for row_idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"data[{row_idx}] must be a row object")
        if "label" not in row:
            raise ValueError(f"data[{row_idx}].label is required")
        label_value = row["label"]
        if isinstance(label_value, bool):
            positive_count += int(label_value)
            negative_count += int(not label_value)
        elif label_value in (0, 1):
            positive_count += int(label_value == 1)
            negative_count += int(label_value == 0)
        else:
            raise ValueError(f"data[{row_idx}].label must be boolean or 0/1")
        for attr in attrs:
            if attr in row and (isinstance(row[attr], bool) or not isinstance(row[attr], (int, float, str))):
                raise ValueError(f"data[{row_idx}].{attr} must be a string or number")
    if positive_count == 0:
        raise ValueError("data has no positive rows; check pos_val and label configuration")
    if negative_count == 0:
        raise ValueError("data has no negative rows; check pos_val and label configuration")


def _validate_multiclass_training_data(data: Any, attrs: list[str]) -> None:
    if not isinstance(data, list):
        raise ValueError("data must be a list of row objects")
    if not data:
        raise ValueError("data must contain at least one row")
    label_values = set()
    for row_idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"data[{row_idx}] must be a row object")
        if "label" not in row:
            raise ValueError(f"data[{row_idx}].label is required")
        label_value = row["label"]
        if not isinstance(label_value, str) or not label_value:
            raise ValueError(f"data[{row_idx}].label must be a non-empty string")
        label_values.add(label_value)
        for attr in attrs:
            if attr in row and (isinstance(row[attr], bool) or not isinstance(row[attr], (int, float, str))):
                raise ValueError(f"data[{row_idx}].{attr} must be a string or number")
    if len(label_values) < 2:
        raise ValueError("data must contain at least two label classes")


class _Obfuscator:
    def __init__(self, str_attrs: list[str], num_attrs: list[str], label: Optional[str], pos_val: Optional[str]) -> None:
        if label is None or pos_val is None:
            raise ValueError("label and pos_val are required")
        self.str_attrs = str_attrs
        self.num_attrs = num_attrs
        self.label = label
        self.pos_val = pos_val
        self.attrs = [_normalize_attr(attr) for attr in str_attrs + num_attrs]
        self.feature_to_token = {attr: _token("f") for attr in self.attrs}
        self.token_to_feature = {token: attr for attr, token in self.feature_to_token.items()}
        self.category_to_token: dict[str, dict[str, str]] = {attr: {} for attr in self.attrs}
        self.token_to_category: dict[str, dict[str, str]] = {attr: {} for attr in self.attrs}
        self.numeric_params = {attr: _NumericTransform.random() for attr in self.attrs}
        self.label_token = _token("y")
        self.pos_val_token = _token("v")

    def obfuscate_config(self) -> dict[str, Any]:
        return {
            "str_attrs": [self.feature_to_token[_normalize_attr(attr)] for attr in self.str_attrs],
            "num_attrs": [self.feature_to_token[_normalize_attr(attr)] for attr in self.num_attrs],
            "label": self.label_token,
            "pos_val": self.pos_val_token,
        }

    def obfuscate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        obfuscated = {"label": bool(row["label"])}
        for attr in self.attrs:
            if attr not in row:
                continue
            obfuscated[self.feature_to_token[attr]] = self._obfuscate_value(attr, row[attr])
        return obfuscated

    def deobfuscate_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        flat_rules = [_rule_to_json(self.deobfuscate_rule(_map_rule(rule))) for rule in payload["flat_rules"]]
        return {
            "str_attrs": self.str_attrs,
            "num_attrs": self.num_attrs,
            "label": self.label,
            "pos_val": self.pos_val,
            "flat_rules": flat_rules,
        }

    def deobfuscate_rule(
        self,
        rule: tuple[Any, list[Any], list[Any]],
    ) -> tuple[Any, list[Any], list[Any]]:
        head, main_items, ab_items = rule
        return (
            self._deobfuscate_head(tuple(head)),
            [self._deobfuscate_item(item) for item in main_items],
            [self._deobfuscate_head(tuple(item)) for item in ab_items],
        )

    def _obfuscate_value(self, attr: str, value: Any) -> Any:
        if isinstance(value, bool):
            return self._obfuscate_category(attr, str(value))
        if isinstance(value, (int, float)):
            return self.numeric_params[attr].forward(float(value))
        if isinstance(value, str):
            return self._obfuscate_category(attr, value)
        raise ValueError(f"{attr} value must be a string or number")

    def _obfuscate_category(self, attr: str, value: str) -> str:
        if value not in self.category_to_token[attr]:
            token = _token("v")
            self.category_to_token[attr][value] = token
            self.token_to_category[attr][token] = value
        return self.category_to_token[attr][value]

    def _deobfuscate_head(self, head: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        attr, op, value = head
        if attr == self.label_token:
            return (_normalize_attr(self.label), op, self.pos_val)
        return (self.token_to_feature.get(attr, attr), op, value)

    def _deobfuscate_item(self, item: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        attr_token, op, value = item
        attr = self.token_to_feature[attr_token]
        if isinstance(value, str):
            value = self.token_to_category[attr].get(value, value)
        else:
            value = round(self.numeric_params[attr].inverse(float(value)), 12)
        return (attr, op, value)


class _NumericTransform:
    def __init__(self, scale: float, offset: float) -> None:
        self.scale = scale
        self.offset = offset

    @classmethod
    def random(cls) -> "_NumericTransform":
        return cls(scale=random.uniform(0.2, 10.0), offset=random.uniform(-10000.0, 10000.0))

    def forward(self, value: float) -> float:
        return self.scale * value + self.offset

    def inverse(self, value: float) -> float:
        return (value - self.offset) / self.scale


class _MulticlassObfuscator:
    def __init__(self, str_attrs: list[str], num_attrs: list[str], label: Optional[str]) -> None:
        if label is None:
            raise ValueError("label is required")
        self.str_attrs = str_attrs
        self.num_attrs = num_attrs
        self.label = label
        self.attrs = [_normalize_attr(attr) for attr in str_attrs + num_attrs]
        self.feature_to_token = {attr: _token("f") for attr in self.attrs}
        self.token_to_feature = {token: attr for attr, token in self.feature_to_token.items()}
        self.category_to_token: dict[str, dict[str, str]] = {attr: {} for attr in self.attrs}
        self.token_to_category: dict[str, dict[str, str]] = {attr: {} for attr in self.attrs}
        self.label_to_token: dict[str, str] = {}
        self.token_to_label: dict[str, str] = {}
        self.numeric_params = {attr: _NumericTransform.random() for attr in self.attrs}
        self.label_token = _token("y")

    def obfuscate_config(self) -> dict[str, Any]:
        return {
            "str_attrs": [self.feature_to_token[_normalize_attr(attr)] for attr in self.str_attrs],
            "num_attrs": [self.feature_to_token[_normalize_attr(attr)] for attr in self.num_attrs],
            "label": self.label_token,
        }

    def obfuscate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        obfuscated = {"label": self._obfuscate_label(row["label"])}
        for attr in self.attrs:
            if attr not in row:
                continue
            obfuscated[self.feature_to_token[attr]] = self._obfuscate_value(attr, row[attr])
        return obfuscated

    def deobfuscate_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        flat_rules = [_rule_to_json(self.deobfuscate_rule(_map_rule(rule))) for rule in payload["flat_rules"]]
        return {
            "algorithm": "foldsem",
            "str_attrs": self.str_attrs,
            "num_attrs": self.num_attrs,
            "label": self.label,
            "flat_rules": flat_rules,
        }

    def deobfuscate_rule(self, rule: tuple[Any, list[Any], list[Any]]) -> tuple[Any, list[Any], list[Any]]:
        head, main_items, ab_items = rule
        return (
            self._deobfuscate_head(tuple(head)),
            [self._deobfuscate_item(item) for item in main_items],
            [self._deobfuscate_head(tuple(item)) for item in ab_items],
        )

    def _obfuscate_label(self, value: str) -> str:
        if value not in self.label_to_token:
            token = _token("y")
            self.label_to_token[value] = token
            self.token_to_label[token] = value
        return self.label_to_token[value]

    def _obfuscate_value(self, attr: str, value: Any) -> Any:
        if isinstance(value, bool):
            return self._obfuscate_category(attr, str(value))
        if isinstance(value, (int, float)):
            return self.numeric_params[attr].forward(float(value))
        if isinstance(value, str):
            return self._obfuscate_category(attr, value)
        raise ValueError(f"{attr} value must be a string or number")

    def _obfuscate_category(self, attr: str, value: str) -> str:
        if value not in self.category_to_token[attr]:
            token = _token("v")
            self.category_to_token[attr][value] = token
            self.token_to_category[attr][token] = value
        return self.category_to_token[attr][value]

    def _deobfuscate_head(self, head: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        attr, op, value = head
        if attr in {"label", self.label_token}:
            return (_normalize_attr(self.label), op, self.token_to_label.get(value, value))
        return (self.token_to_feature.get(attr, attr), op, value)

    def _deobfuscate_item(self, item: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        attr_token, op, value = item
        attr = self.token_to_feature[attr_token]
        if isinstance(value, str):
            value = self.token_to_category[attr].get(value, value)
        else:
            value = round(self.numeric_params[attr].inverse(float(value)), 12)
        return (attr, op, value)


def _eval_item(item: tuple[Any, Any, Any], x: dict[str, Any]) -> bool:
    attr, op, val = item
    if attr is None:
        return False
    if op == "!=<":
        return not _eval_item((attr, "=<", val), x)
    if op == "!>":
        return not _eval_item((attr, ">", val), x)
    x_val = x[attr] if attr in x else ""
    if isinstance(val, str):
        if op == "==":
            return x_val == val
        if op == "!=":
            return x_val != val
        return False
    if isinstance(x_val, str):
        return False
    if op == "=<":
        return x_val <= val
    if op == ">":
        return x_val > val
    return False


def _justify_multiclass(
    flat_rules: list[tuple[Any, list[Any], list[Any]]],
    x: dict[str, Any],
    rule_head: Optional[tuple[Any, Any, Any]] = None,
    flag: bool = True,
    start: int = 0,
) -> bool:
    head, main_items, ab_items = flat_rules[start]
    idx = start
    if rule_head is not None:
        for j, rule in enumerate(flat_rules):
            head, main_items, ab_items = rule
            if head == rule_head:
                idx = j
                break
    if rule_head is not None and rule_head != head:
        return False
    if not all([_eval_item(j, x) for j in main_items]):
        return False
    if len(ab_items) > 0:
        for j in ab_items:
            if _justify_multiclass(flat_rules, x, rule_head=j, flag=not flag):
                return False
    return True


def _classify_multiclass(
    flat_rules: list[tuple[Any, list[Any], list[Any]]],
    x: dict[str, Any],
    label: str,
) -> Optional[str]:
    for i, rule in enumerate(flat_rules):
        head, _, _ = rule
        attr, _, val = head
        if attr == label and _justify_multiclass(flat_rules, x, start=i):
            return val
    return None


def _justify_one(
    flat_rules: list[tuple[Any, list[Any], list[Any]]],
    x: dict[str, Any],
    res: list[Any],
    rule_head: tuple[Any, Any, Any],
    start: int = 0,
) -> int:
    for i in range(start, len(flat_rules)):
        head, main_items, ab_items = flat_rules[i]
        if tuple(head) != tuple(rule_head):
            continue
        if not all((_eval_item(j, x) for j in main_items)):
            continue
        if ab_items:
            continue_flag = False
            for j in ab_items:
                if _rebut_one(flat_rules, x, res=res, rule_head=j) < 0:
                    continue_flag = True
                    break
            if continue_flag:
                continue
        res.append(flat_rules[i])
        return i
    return -1


def _rebut_one(
    flat_rules: list[tuple[Any, list[Any], list[Any]]],
    x: dict[str, Any],
    res: list[Any],
    rule_head: tuple[Any, Any, Any],
    start: int = 0,
) -> int:
    for i in range(start, len(flat_rules)):
        head, main_items, ab_items = flat_rules[i]
        if tuple(head) != tuple(rule_head):
            continue
        if not all((_eval_item(j, x) for j in main_items)):
            res.append(flat_rules[i])
            return i
        if ab_items:
            for j in ab_items:
                if _justify_one(flat_rules, x, res=res, rule_head=j) > -1:
                    res.append(flat_rules[i])
                    return i
    return -1


def _explain(
    flat_rules: list[tuple[Any, list[Any], list[Any]]],
    x: dict[str, Any],
    rule_head: tuple[Any, Any, Any],
    get_all: bool = False,
) -> list[list[tuple[Any, list[Any], list[Any]]]]:
    ret, i = [], 0
    y = _justify_one(flat_rules, x, res=[], rule_head=rule_head) > -1
    f = _justify_one if y else _rebut_one
    get_all = get_all if y else True
    while i < len(flat_rules):
        res: list[Any] = []
        idx = f(flat_rules, x, res=res, rule_head=rule_head, start=i)
        if idx > -1:
            ret.append(res)
            i = idx + 1
            if not get_all:
                break
        else:
            break
    return ret


def _explain_data(flat_rules: list[tuple[Any, list[Any], list[Any]]], x: dict[str, Any]) -> dict[str, Any]:
    ret = {}
    for rule in flat_rules:
        for item in rule[1]:
            attr, _, _ = item
            ret[attr] = x.get(attr)
    return ret


def _decode_rules(rules: list[tuple[Any, list[Any], list[Any]]], x: Optional[dict[str, Any]] = None) -> list[str]:
    attr_map, attr_idx = {}, 1

    def _attr_idx(attr: str) -> str:
        nonlocal attr_idx
        if attr not in attr_map:
            attr_map[attr] = attr_idx
            attr_idx += 1
        return str(attr_map[attr])

    def _item_prefix(item: tuple[Any, Any, Any]) -> str:
        if x is None:
            return ""
        return "[T]" if _eval_item(item, x) else "[F]"

    def _rule_prefix(rule_idx: tuple[Any, Any, Any]) -> str:
        if x is None:
            return ""
        if rule_idx in [r[0] for r in rules]:
            return "[T]" if _justify_one(rules, x, res=[], rule_head=rule_idx) > -1 else "[F]"
        return "[U]"

    def _decode_item(item: tuple[Any, Any, Any]) -> str:
        neg_map = {"[T]": "[F]", "[F]": "[T]", "[U]": "[U]", "": ""}
        neg_ops = {"!=<": "=<", "!>": ">"}
        attr, op, val = item
        rendered_val = "'" + val + "'" if isinstance(val, str) else val
        if op == "==":
            return f"{_item_prefix(item)}{attr}(X,{rendered_val})"
        if op == "!=":
            return f"not {neg_map[_item_prefix(item)]}{attr}(X,{rendered_val})"
        if op in neg_ops:
            idx = _attr_idx(attr)
            return f"{attr}(X,N{idx}), {_item_prefix(item)}not(N{idx}{neg_ops[op]}{round(val, 3)})"
        idx = _attr_idx(attr)
        return f"{attr}(X,N{idx}), {_item_prefix(item)}N{idx}{op}{round(val, 3)}"

    def _decode_head(head: tuple[Any, Any, Any]) -> str:
        return f"{_rule_prefix(head)}{head[0]}(X,'{head[2]}')"

    def _uniq_list(seq: list[str]) -> list[str]:
        seen = set()
        seen_add = seen.add
        return [s for s in seq if not (s in seen or seen_add(s))]

    def _decode_rule(rule: tuple[Any, list[Any], list[Any]]) -> str:
        head = _decode_head(rule[0])
        body = ", ".join([_decode_item(i) for i in rule[1]])
        body = ", ".join(_uniq_list(body.split(", ")))
        tail = ", ".join(["not " + _decode_head(i) for i in rule[2]])
        return head + " :- " + body + (", " + tail if len(tail) > 0 else "") + "."

    return [_decode_rule(r) for r in rules]


def _proof_tree(
    rules: list[tuple[Any, list[Any], list[Any]]],
    x: dict[str, Any],
    rule_head: tuple[Any, Any, Any],
) -> list[str]:
    def _item_suffix(item: tuple[Any, Any, Any]) -> str:
        return " does hold" if _eval_item(item, x) else " does not hold"

    def _rule_suffix(_rule_head: tuple[Any, Any, Any]) -> str:
        return " does hold" if _justify_one(rules, x, rule_head=_rule_head, res=[]) > -1 else " does not hold"

    def _strify(_val: Any) -> str:
        return "'" + _val + "'" if isinstance(_val, str) else str(round(_val, 3))

    def _decode_item(item: tuple[Any, Any, Any]) -> str:
        attr, op, val = item
        current = _strify(x[attr]) if attr in x else "null"
        if op == "==":
            return "the value of " + attr + " is " + current + " which should equal " + _strify(val) + _item_suffix(item)
        if op == "!=":
            return "the value of " + attr + " is " + current + " which should not equal " + _strify(val) + _item_suffix(item)
        if op == "=<":
            return "the value of " + attr + " is " + current + " which should be less equal to " + _strify(val) + _item_suffix(item)
        if op == "!=<":
            return "the value of " + attr + " is " + current + " which should be greater than " + _strify(val) + " or be NaN" + _item_suffix(item)
        if op == ">":
            return "the value of " + attr + " is " + current + " which should be greater than " + _strify(val) + _item_suffix(item)
        if op == "!>":
            return "the value of " + attr + " is " + current + " which should be less equal to " + _strify(val) + " or be NaN" + _item_suffix(item)
        return str(item)

    def _decode_head(head: tuple[Any, Any, Any]) -> str:
        prefix = "" if head == rule_head else "exception "
        return prefix + head[0] + "(X,'" + head[2] + "')" + _rule_suffix(head) + " because"

    def _decode_rule(rule: tuple[Any, list[Any], list[Any]], indent: int = 0) -> str:
        head = "\t" * indent + _decode_head(rule[0]) + "\n"
        body = "".join(["\t" * (indent + 1) + _decode_item(i) + " and\n" for i in rule[1]])
        tail = "".join([_decode_rule(rules[i], indent + 1) for i in range(len(rules)) if rules[i][0] in rule[2]])
        return head + body[:-4] + "\n" + tail

    return [_decode_rule(r) for r in rules if r[0] == rule_head]


def split_data(
    data: list[dict[str, Any]],
    ratio: float = 0.8,
    rand: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied = list(data)
    if rand:
        random.shuffle(copied)
    num = int(len(copied) * ratio)
    return copied[:num], copied[num:]


def get_scores(predictions: list[bool], labels: list[bool]) -> tuple[float, float, float, float]:
    n = len(labels)
    if n == 0:
        return 0, 0, 0, 0

    tp = tn = fp = fn = 0.0
    for predicted, label in zip(predictions, labels):
        tp = tp + 1.0 if label and predicted else tp
        tn = tn + 1.0 if not label and not predicted else tn
        fn = fn + 1.0 if label and not predicted else fn
        fp = fp + 1.0 if not label and predicted else fp

    if tp < 1:
        p = 0 if fp < 1 else tp / (tp + fp)
        r = 0 if fn < 1 else tp / (tp + fn)
    else:
        p, r = tp / (tp + fp), tp / (tp + fn)
    f1 = 0 if r * p == 0 else 2 * r * p / (r + p)
    return (tp + tn) / n, p, r, f1


def save_model_to_file(model: Union[Foldse, Foldsem], file_name: Union[str, Path]) -> None:
    flat_rules = model._require_rules()
    algorithm = "foldsem" if isinstance(model, Foldsem) else "foldse"
    payload = {
        "algorithm": algorithm,
        "str_attrs": model.str_attrs,
        "num_attrs": model.num_attrs,
        "label": model.label,
        "flat_rules": [_rule_to_json(rule) for rule in flat_rules],
        "asp": model.asp(),
        "training_metrics": model.training_metrics,
    }
    if isinstance(model, Foldse):
        payload["pos_val"] = model.pos_val
    Path(file_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model_from_file(file_name: Union[str, Path]) -> Union[Foldse, Foldsem]:
    payload = json.loads(Path(file_name).read_text(encoding="utf-8"))
    if payload.get("algorithm") == "foldsem":
        model: Union[Foldse, Foldsem] = Foldsem(
            payload["str_attrs"],
            payload["num_attrs"],
            payload["label"],
        )
    else:
        model = Foldse(
            payload["str_attrs"],
            payload["num_attrs"],
            payload["label"],
            payload["pos_val"],
        )
    model._load_model_payload(payload)
    return model
