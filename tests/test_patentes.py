import re
import pytest

PATTERN_OLD = re.compile(r"^[A-Z]{2}-?\d{4}$")  # AA-1234 or AA1234
PATTERN_NEW = re.compile(r"^[A-Z]{4}-?\d{2}$")  # AAAA-12 or AAAA12

def es_patente_valida(placa: str) -> bool:
    """Devuelve True si la patente coincide con los formatos antiguos o nuevos."""
    if placa is None:
        return False
    placa_norm = str(placa).strip().upper()
    if not placa_norm:
        return False
    return bool(PATTERN_OLD.fullmatch(placa_norm) or PATTERN_NEW.fullmatch(placa_norm))


def test_patentes_validas_no_son_invalidas():
    validas = ["AB1234", "AB-1234", "ABCD12", "ABCD-12"]
    for patente in validas:
        assert es_patente_valida(patente), f"{patente} debería ser válida"


def test_patentes_invalidas_y_nulas_detectadas_sin_excepcion():
    invalidas = ["A1234", "ABCDE12", "1234AB", "AB-123", None, ""]
    for patente in invalidas:
        assert not es_patente_valida(patente), f"{patente} debería ser inválida"
