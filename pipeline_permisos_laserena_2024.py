# --------------------------------------------------------------
# pipeline_permisos_laserena_2024.py
# Limpieza + Auditoría normativa (Chile 2024) + Gráficos legibles
# --------------------------------------------------------------

from pathlib import Path
import re
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =========================
# CONFIGURACIÓN
# =========================
INPUT_FILE   = "permiso-de-circulacion-2024.xlsx"
SHEET_NAME   = 0
OUTPUT_DIR   = Path("salidas"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_YEAR = 2024         # año de los permisos
CURRENT_YEAR = 2025
MAKE_PLOTS   = True          # generar PNG
TOPK_LABELS  = 3             # etiqueta solo top-3
MODO_AUDITORIA = True        # True: NO elimina, solo marca y reporta. False: aplica filtros duros.

NA_VALUES = ["", "NA", "N/A", "null", "None", "-", "--", "(Ninguno)"]

# Rangos lógicos para "cap" (recorte visual/numérico razonable)
RANGE_RULES = {
    "año_fabricacion": (1950, CURRENT_YEAR),
    "año_permiso":     (DATASET_YEAR, DATASET_YEAR),
    "cc":              (500, 8000),
    "tasacion":        (0, 200_000_000),
    "valor_permiso":   (0, 20_000_000),
    "asientos":        (1, 70),
    "carga":           (0, 5_000),
}

# Ventanas de pago oficiales 2024 (1ª cuota y 2ª cuota)
W1_START = pd.Timestamp("2024-02-01"); W1_END = pd.Timestamp("2024-03-31")
W2_START = pd.Timestamp("2024-08-01"); W2_END = pd.Timestamp("2024-08-31")

# Mínimo legal 0.5 UTM ene-2024 (UTM ene= 64.666)
UTM_ENERO_2024    = 64666
MIN_PERMISO_2024  = int(round(UTM_ENERO_2024 * 0.5))  # = 32333

# Conjuntos y mapeos
VALID_SELLOS = {"verde", "amarillo", "rojo"}
VALID_COMB   = {"bencina", "diésel", "híbrido", "eléctrico", "gnv", "glp"}
COMBUSTIBLE_MAP = {
    "benc": "bencina", "gasolina": "bencina",
    "dies": "diésel", "díes": "diésel", "diesel": "diésel", "d": "diésel",
    "hibr": "híbrido", "híbr": "híbrido", "hybrid": "híbrido",
    "electrico": "eléctrico"
}
TRANSMISION_MAP = {
    "mec": "mecánica", "mecanica": "mecánica",
    "aut": "automática", "automatiza": "automática",
    "automatico": "automática", "automatica": "automática",
    "cvt": "automática"
}
EQUIPADO_MAP = {"norm": "normal"}

LINE = "─" * 80

# =========================
# HELPERS
# =========================
def normalize_str(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.lower())

def cap_series(s: pd.Series, lo, hi) -> pd.Series:
    return s.clip(lower=lo, upper=hi)

def fmt_int(n):
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)

def money_fmt(x, pos):
    # 1 000 000 -> 1.0 M
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f} M"
    elif x >= 1_000:
        return f"{x/1_000:.0f} K"
    return str(int(x))

def add_topk_labels_bars(ax, k=TOPK_LABELS):
    patches = list(ax.patches)
    heights = [p.get_height() for p in patches]
    if not heights:
        return
    top_idx = sorted(range(len(heights)), key=lambda i: heights[i], reverse=True)[:k]
    for i in top_idx:
        p = patches[i]
        h = heights[i]
        if h <= 0 or pd.isna(h): continue
        x = p.get_x() + p.get_width() / 2
        y = h
        ax.annotate(fmt_int(h), (x, y), ha="center", va="bottom", fontsize=9,
                    xytext=(0, 3), textcoords="offset points")

def print_section(title: str):
    print("\n" + LINE); print(title); print(LINE)

def boxplot_legible(s: pd.Series, nombre: str, outpath: Path):
    """Boxplot horizontal legible: limita eje al p99, anota mediana y muestra ticks formateados."""
    s = s.dropna()
    if s.empty: return
    p99 = np.percentile(s, 99)  # recorta vista (no datos)
    med = float(np.median(s))
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    ax.boxplot(s, vert=False, whis=1.5, showfliers=True)
    ax.set_title(f"Boxplot: {nombre}")
    ax.set_xlabel(nombre)
    ax.grid(True, linestyle=":", alpha=.25)
    # Limitar la vista para leer mejor
    ax.set_xlim(left=float(s.min()), right=float(p99))
    # Anotar mediana
    ax.axvline(med, color="tab:orange", linestyle="--", linewidth=1)
    ax.annotate(f"Mediana: {fmt_int(med)}", (med, 0.98), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=9, xytext=(0, -6), textcoords="offset points")
    # Formato $ en millones para variables grandes
    if s.max() >= 1_000_000:
        ax.xaxis.set_major_formatter(FuncFormatter(money_fmt))
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def hist_legible(s: pd.Series, nombre: str, outpath: Path, bins="auto"):
    """Histograma con etiquetas solo en Top-k bins (k controlado por TOPK_LABELS) y eje X formateado; recorta al p99."""
    s = s.dropna()
    if s.empty: return
    p99 = np.percentile(s, 99)
    data = s[s <= p99]
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    n, bins_edges, patches = ax.hist(data, bins=bins)
    ax.set_title(f"Histograma: {nombre}")
    ax.set_xlabel(nombre); ax.set_ylabel("Frecuencia")
    ax.grid(True, linestyle=":", alpha=.25)
    if s.max() >= 1_000_000:
        ax.xaxis.set_major_formatter(FuncFormatter(money_fmt))
    # Top-k bins
    counts = list(n)
    if any(counts):
        top_idx = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:TOPK_LABELS]
        for i in top_idx:
            h = counts[i]
            if h <= 0: continue
            patch = patches[i]
            x = patch.get_x() + patch.get_width()/2; y = h
            ax.annotate(fmt_int(h), (x, y), ha="center", va="bottom", fontsize=9,
                        xytext=(0, 3), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# =========================
# 1) CARGA
# =========================
print("Leyendo Excel…")
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, na_values=NA_VALUES, engine="openpyxl")
orig_rows = len(df)

# =========================
# 2) RENOMBRAR + NORMALIZAR
# =========================
rename_map = {
    "Placa":"placa","Vehiculo":"vehiculo","Marca":"marca","Modelo":"modelo","Color":"color",
    "Año Fabricacion":"año_fabricacion","Carga":"carga","Asientos":"asientos","Sello":"sello","CC":"cc",
    "Combustible":"combustible","Transmision":"transmision","Equipado":"equipado",
    "Año Permiso":"año_permiso","Período":"periodo","Calculo":"calculo","Codigo_SII":"codigo_sii",
    "Tasacion":"tasacion","Tipo Pago":"tipo_pago","Valor Permiso":"valor_permiso","Concepto":"concepto",
    "Fecha Pago":"fecha_pago",
}
df = df.rename(columns=rename_map)

if "placa" in df: df["placa"] = df["placa"].astype("string").str.upper().str.replace(r"\s+","",regex=True)
for c in ["vehiculo","marca","modelo","color","combustible","transmision","equipado",
          "periodo","calculo","tipo_pago","concepto","sello"]:
    if c in df: df[c] = normalize_str(df[c])
if "fecha_pago" in df.columns:
    df["fecha_pago"] = pd.to_datetime(df["fecha_pago"], errors="coerce", dayfirst=True)

# Normalizaciones fuertes
if "combustible" in df:
    df["combustible_norm"] = df["combustible"].replace(COMBUSTIBLE_MAP)
if "transmision" in df:
    df["transmision_norm"] = df["transmision"].replace(TRANSMISION_MAP)
if "equipado" in df:
    df["equipado_norm"] = df["equipado"].replace(EQUIPADO_MAP)

# Sello
if "sello" in df:
    df["sello_clean"] = df["sello"].str.replace(r"^sello\s+", "", regex=True)
    df["sello_clean"] = df["sello_clean"].replace({
        "v":"verde","ver":"verde","sel verde":"verde",
        "a":"amarillo","amar":"amarillo","amaril":"amarillo",
        "r":"rojo","roj":"rojo"
    })
    df["sello_valido"] = df["sello_clean"].isin(VALID_SELLOS)
    mask_sello_inv = ~df["sello_valido"]
    if mask_sello_inv.any():
        df.loc[mask_sello_inv, ["placa","año_permiso","sello","sello_clean"]]\
          .to_csv(OUTPUT_DIR/"errores_sello_invalido.csv", index=False, encoding="utf-8")
        print(f"[SELLO] {int(mask_sello_inv.sum())} con sello no válido -> errores_sello_invalido.csv")

# =========================
# 3) DEDUPLICADOS LÓGICOS
# =========================
if {"placa","año_permiso","fecha_pago"}.issubset(df.columns):
    df = (df.sort_values("fecha_pago", ascending=False)
          .drop_duplicates(subset=["placa","año_permiso"], keep="first"))

# =========================
# 4) AUDITORÍA NORMATIVA/NEGOCIO (NO elimina filas)
# =========================
# Ventanas de pago 2024
if "fecha_pago" in df.columns:
    df["pago_en_ventana"] = (
        ((df["fecha_pago"] >= W1_START) & (df["fecha_pago"] <= W1_END)) |
        ((df["fecha_pago"] >= W2_START) & (df["fecha_pago"] <= W2_END))
    )
    fuera = (~df["pago_en_ventana"]) & df["fecha_pago"].notna()
    if fuera.any():
        df.loc[fuera, ["placa","año_permiso","fecha_pago","tipo_pago","valor_permiso"]]\
          .to_csv(OUTPUT_DIR/"alerta_pagos_fuera_ventana.csv", index=False, encoding="utf-8")
        print(f"[PAGOS] {int(fuera.sum())} fuera de ventana oficial 2024 -> alerta_pagos_fuera_ventana.csv")

# Mínimo legal (0.5 UTM ene-2024)
if "valor_permiso" in df.columns:
    bajo_min = df["valor_permiso"] < MIN_PERMISO_2024
    if bajo_min.any():
        df.loc[bajo_min, ["placa","año_permiso","valor_permiso","tasacion","tipo_pago"]]\
          .to_csv(OUTPUT_DIR/"error_permiso_bajo_minimo.csv", index=False, encoding="utf-8")
        print(f"[MINIMO] {int(bajo_min.sum())} < $ {fmt_int(MIN_PERMISO_2024)} -> error_permiso_bajo_minimo.csv")

# Coherencia gruesa con tasación (tope 4.5% + 10% tolerancia)
if {"valor_permiso","tasacion"}.issubset(df.columns):
    upper = (df["tasacion"] * 0.045) * 1.10
    incoh = df["valor_permiso"] > upper
    if incoh.any():
        df.loc[incoh, ["placa","tasacion","valor_permiso"]]\
          .to_csv(OUTPUT_DIR/"alerta_permiso_sospechosamente_alto.csv", index=False, encoding="utf-8")
        print(f"[COHERENCIA] {int(incoh.sum())} >> 4.5% (tolerancia) -> alerta_permiso_sospechosamente_alto.csv")

# Código SII: 2 letras + 7 dígitos
if "codigo_sii" in df.columns:
    pat_sii = re.compile(r"^[A-Z]{2}\d{7}$")
    df["codigo_sii_norm"] = df["codigo_sii"].astype("string").str.strip().str.upper().str.replace(r"\s+","",regex=True)
    bad_sii = ~df["codigo_sii_norm"].fillna("").str.match(pat_sii)
    if bad_sii.any():
        df.loc[bad_sii, ["placa","año_permiso","codigo_sii","codigo_sii_norm"]]\
          .to_csv(OUTPUT_DIR/"error_codigo_sii_formato.csv", index=False, encoding="utf-8")
        print(f"[SII] {int(bad_sii.sum())} Código SII inválido -> error_codigo_sii_formato.csv")

# Patentes/chapa
if "placa" in df.columns:
    pat_old = re.compile(r"^[A-Z]{2}-?\d{4}$")  # AA-1234
    pat_new = re.compile(r"^[A-Z]{4}-?\d{2}$")  # BBBB-12
    df["placa_norm"] = df["placa"].astype("string").str.strip().str.upper()
    bad_placa = ~(df["placa_norm"].str.match(pat_old) | df["placa_norm"].str.match(pat_new))
    if bad_placa.any():
        df.loc[bad_placa, ["placa","placa_norm","año_permiso"]]\
          .to_csv(OUTPUT_DIR/"error_patente_formato.csv", index=False, encoding="utf-8")
        print(f"[PATENTE] {int(bad_placa.sum())} formato no reconocido -> error_patente_formato.csv")

# Combustible permitido tras normalizar
if "combustible_norm" in df.columns:
    bad_comb = ~df["combustible_norm"].isin(VALID_COMB)
    if bad_comb.any():
        df.loc[bad_comb, ["placa","combustible","combustible_norm"]]\
          .to_csv(OUTPUT_DIR/"alerta_combustible_no_estandar.csv", index=False, encoding="utf-8")
        print(f"[COMBUSTIBLE] {int(bad_comb.sum())} fuera del set -> alerta_combustible_no_estandar.csv")

# Año del dataset y fechas fuera de 2024
if "año_permiso" in df.columns:
    fuera_year = df["año_permiso"] != DATASET_YEAR
    if fuera_year.any():
        df.loc[fuera_year, ["placa","año_permiso","fecha_pago"]]\
          .to_csv(OUTPUT_DIR/"alerta_anio_permiso_distinto_2024.csv", index=False, encoding="utf-8")
        print(f"[AÑO] {int(fuera_year.sum())} con año_permiso != {DATASET_YEAR} -> alerta_anio_permiso_distinto_2024.csv")

if "fecha_pago" in df.columns:
    fuera_2024 = df["fecha_pago"].dt.year.notna() & (df["fecha_pago"].dt.year != DATASET_YEAR)
    if fuera_2024.any():
        df.loc[fuera_2024, ["placa","año_permiso","fecha_pago"]]\
          .to_csv(OUTPUT_DIR/"alerta_fecha_pago_fuera_2024.csv", index=False, encoding="utf-8")
        print(f"[FECHA] {int(fuera_2024.sum())} con fecha_pago fuera de 2024 -> alerta_fecha_pago_fuera_2024.csv")

# =========================
# 5) CAP/RECORTE (limita extremos a rangos razonables; no elimina)
# =========================
for col, (lo, hi) in RANGE_RULES.items():
    if col in df and pd.api.types.is_numeric_dtype(df[col]):
        before_min, before_max = df[col].min(), df[col].max()
        df[col] = cap_series(df[col], lo, hi)
        after_min, after_max = df[col].min(), df[col].max()
        print(f"[cap] {col}: {before_min}..{before_max} -> {after_min}..{after_max}")

# =========================
# 6) REGLAS DURAS (solo si MODO_AUDITORIA=False)
# =========================
if not MODO_AUDITORIA:
    if "año_permiso" in df.columns:
        df = df[df["año_permiso"] == DATASET_YEAR]
    if "asientos" in df.columns:
        df = df[df["asientos"].fillna(0) >= 1]

# =========================
# 7) RESUMEN DE CONSOLA
# =========================
print_section("Resumen de Permisos de Circulación La Serena 2024")
print(f"Filas: {fmt_int(len(df))} (original: {fmt_int(orig_rows)})")
print(f"Columnas: {fmt_int(len(df.columns))}")
mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
print(f"Memoria estimada: {mem_mb:,.2f} MB".replace(",", "."))

print_section("Esquema (tipo, nulos, únicos)")
schema = pd.DataFrame({
    "columna": df.columns,
    "dtype": [str(t) for t in df.dtypes],
    "n_nulos": df.isna().sum().values,
    "%_nulos": (df.isna().mean().values * 100).round(2),
    "n_unicos": [df[c].nunique(dropna=True) for c in df.columns]
})
print(schema.head(24).to_string(index=False))

print_section("Top nulos (%)")
print((df.isna().mean()*100).round(2).sort_values(ascending=False).head(10).to_string())

if "fecha_pago" in df.columns:
    print_section("Rango de fechas (fecha_pago)")
    print(f"min: {df['fecha_pago'].min()} | max: {df['fecha_pago'].max()}")

# =========================
# 8) GRÁFICOS LEGIBLES (Top-3 etiquetas) — Matplotlib puro
# =========================
if MAKE_PLOTS:
    num_cols = [c for c in ["año_fabricacion","asientos","carga","cc","tasacion","valor_permiso"] if c in df.columns]
    for c in num_cols:
        s = df[c].dropna()
        if s.empty: continue
        # Histograma y Boxplot con p99
        hist_legible(s, c, OUTPUT_DIR / f"hist_{c}.png", bins="auto")
        boxplot_legible(s, c, OUTPUT_DIR / f"box_{c}.png")

    for c in [x for x in ["combustible_norm","transmision_norm","vehiculo","marca","color","sello_clean"] if x in df.columns]:
        vc = df[c].value_counts().head(15)
        if vc.empty: continue
        fig, ax = plt.subplots(figsize=(9.6, 7.2))
        vc.plot(kind="bar", ax=ax)
        ax.set_title(f"Top categorías: {c}")
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", linestyle=":", alpha=.25)
        ax.tick_params(axis="x", labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
        add_topk_labels_bars(ax, k=TOPK_LABELS)
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"bar_{c}.png", dpi=300); plt.close(fig)

# =========================
# 9) REPORTES Y EXPORTES
# =========================
(df.isna().mean()*100).round(2).sort_values(ascending=False)\
  .to_csv(OUTPUT_DIR / "reporte_nulos_%_.csv", header=["% nulos"], encoding="utf-8")

try:
    desc = df.describe(include="all", datetime_is_numeric=True).T
except TypeError:
    desc = df.describe(include="all").T
desc.to_csv(OUTPUT_DIR / "resumen_describe.csv", encoding="utf-8")

df.to_parquet(OUTPUT_DIR / "permisos_limpio.parquet", index=False)
df.to_csv(OUTPUT_DIR / "permisos_limpio.csv", index=False, encoding="utf-8")
with pd.ExcelWriter(OUTPUT_DIR / "permisos_limpio.xlsx", engine="openpyxl") as w:
    df.to_excel(w, sheet_name="limpio", index=False)

print("\n✅ Listo")
print(f"Filas finales: {fmt_int(len(df))}")
print(f"Salidas en: {OUTPUT_DIR.resolve()}")
print("CSV de auditoría (solo si hubo hallazgos):")
print(" - errores_sello_invalido.csv")
print(" - alerta_pagos_fuera_ventana.csv")
print(" - error_permiso_bajo_minimo.csv")
print(" - alerta_permiso_sospechosamente_alto.csv")
print(" - error_codigo_sii_formato.csv")
print(" - error_patente_formato.csv")
print(" - alerta_combustible_no_estandar.csv")
print(" - alerta_anio_permiso_distinto_2024.csv")
print(" - alerta_fecha_pago_fuera_2024.csv")
