"""Transparent encrypted client for the private FOLD-SE API service."""

from foldse_client.remote import Foldse, Foldsem, get_scores, load_model_from_file, save_model_to_file, split_data

__all__ = [
    "Foldse",
    "Foldsem",
    "get_scores",
    "load_model_from_file",
    "save_model_to_file",
    "split_data",
]
