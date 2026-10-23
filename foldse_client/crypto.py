from __future__ import annotations

import base64
import json
import os
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


AAD_REQUEST = b"fold-api:v1:request"
AAD_RESPONSE = b"fold-api:v1:response"


def b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def b64decode(encoded: str) -> bytes:
    return base64.b64decode(encoded.encode("ascii"), validate=True)


def load_public_key(pem: bytes) -> rsa.RSAPublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, rsa.RSAPublicKey):
        raise TypeError("Expected an RSA public key")
    return key


def encrypt_json_request(
    payload: dict[str, Any],
    public_key: rsa.RSAPublicKey,
) -> tuple[dict[str, str], bytes]:
    request_key = os.urandom(32)
    nonce = os.urandom(12)
    plaintext = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ciphertext = AESGCM(request_key).encrypt(nonce, plaintext, AAD_REQUEST)
    wrapped_key = public_key.encrypt(
        request_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return (
        {
            "version": "v1",
            "alg": "RSA-OAEP-SHA256+A256GCM",
            "wrapped_key": b64encode(wrapped_key),
            "nonce": b64encode(nonce),
            "ciphertext": b64encode(ciphertext),
        },
        request_key,
    )


def decrypt_json_response(envelope: dict[str, str], request_key: bytes) -> dict[str, Any]:
    if envelope.get("version") != "v1":
        raise ValueError("Unsupported response envelope version")
    plaintext = AESGCM(request_key).decrypt(
        b64decode(envelope["nonce"]),
        b64decode(envelope["ciphertext"]),
        AAD_RESPONSE,
    )
    decoded = json.loads(plaintext.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Expected a JSON object response")
    return decoded

