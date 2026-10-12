#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_inspect_data.py — Inspección SOLO LECTURA de los datos reales del taller.

No modifica ningún archivo de datos. Escribe un reporte de texto compacto
(`inspect_report.txt`) para pegarlo/adjuntarlo en el chat y así diseñar el
subconjunto del taller sobre el esquema REAL, sin suponer nada.

Uso (desde la raíz del repo, con el entorno del curso activado):
    python scripts/00_inspect_data.py
    python scripts/00_inspect_data.py --biom datos/otro.biom --meta datos/otra_meta.csv
    python scripts/00_inspect_data.py --sep ";"            # si el separador no se detecta bien
    python scripts/00_inspect_data.py --no-tsv             # omitir el TSV de 156 MB

Requiere: pandas, numpy, scipy, biom-format (ya están en requirements.txt).
Nunca convierte la tabla completa a denso (el BIOM se mantiene disperso).
"""
import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Palabras clave para localizar columnas de ambiente / diseño en el metadato
ENV_NAME_RE = re.compile(
    r"(env|ecosys|biome|habitat|source|sample_?type|material|lake|soil|aquat|"
    r"fresh|empo|categor|ontolog|lago|suelo|ambiente)", re.I)
GEO_TECH_NAME_RE = re.compile(
    r"(study|project|bioproject|country|region|geo|lat|lon|depth|\bph\b|temp|"
    r"year|date|platform|technolog|assay|seq|library|read|depth|biosample)", re.I)
# Valores que delatan una columna de ambiente aunque se llame raro
ENV_VALUE_RE = re.compile(r"(?:lake|soil|lago|suelo|fresh|sediment|pond|reservoir|aquatic|terrestrial)", re.I)


class Tee:
    """Duplica stdout a un archivo."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def section(title):
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


def short(x, n=90):
    x = str(x)
    return x if len(x) <= n else x[: n - 3] + "..."


def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


# ----------------------------------------------------------------------------
def inspect_biom(path):
    from biom import load_table

    section("1. BIOM")
    print(f"Archivo: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    t = load_table(str(path))
    n_feat, n_samp = t.shape
    print(f"Forma: {n_feat} features x {n_samp} muestras")
    print(f"Densidad (fracción no-cero): {t.get_table_density():.4%}")

    mat = t.matrix_data.tocsr()  # features x muestras, disperso
    print(f"dtype de la matriz: {mat.dtype}")
    vals = mat.data
    if vals.size:
        rng = np.random.default_rng(0)
        sv = vals if vals.size <= 200_000 else rng.choice(vals, 200_000, replace=False)
        frac_int = float(np.mean(np.isclose(sv, np.round(sv))))
        print(f"Valores no-cero: min={vals.min():.6g}  mediana={np.median(sv):.6g}  "
              f"max={vals.max():.6g}  fracción enteros={frac_int:.1%}")
    else:
        frac_int = float("nan")
        print("La matriz no tiene valores no-cero.")

    sums = np.asarray(t.sum(axis="sample")).ravel()
    q = np.percentile(sums, [0, 5, 50, 95, 100])
    print("Suma por muestra (min/p5/mediana/p95/max): " + " / ".join(f"{v:.6g}" for v in q))
    print(f"Muestras con suma 0: {(sums == 0).sum()}")
    if np.allclose(sums, 1, atol=1e-3):
        kind = "ABUNDANCIA RELATIVA (cada muestra suma ≈ 1)"
    elif np.allclose(sums, 100, atol=0.1):
        kind = "ABUNDANCIA RELATIVA en porcentaje (cada muestra suma ≈ 100)"
    elif frac_int > 0.99:
        kind = "CONTEOS (valores enteros; la profundidad varía entre muestras)"
    else:
        kind = "NO ENTEROS con sumas variables → normalizado (TPM/CPM/cobertura u otro). Revisar origen."
    print(f">>> Interpretación automática: {kind}")

    ids_f = t.ids("observation")
    ids_s = t.ids("sample")
    print(f"IDs de features (primeros 3): {[str(i) for i in ids_f[:3]]}")
    print(f"IDs de muestras (primeros 3 / últimos 2): {[str(i) for i in ids_s[:3]]} / {[str(i) for i in ids_s[-2:]]}")

    for axis, ids in (("observation", ids_f), ("sample", ids_s)):
        md = t.metadata(id=ids[0], axis=axis)
        label = "features" if axis == "observation" else "muestras"
        if md:
            print(f"Metadata embebida en {label}: claves={[str(k) for k in md.keys()]}")
            for k, v in list(md.items())[:3]:
                print(f"    {k}: {short(v)}")
        else:
            print(f"Metadata embebida en {label}: NINGUNA")

    prev = mat.getnnz(axis=1)
    print("Features que sobreviven a un filtro de prevalencia (presentes en ≥ X% de las muestras):")
    for pct in (1, 2, 5, 10, 25, 50):
        thr = int(np.ceil(pct / 100 * n_samp))
        print(f"    ≥{pct:>2}% ({thr:>3} muestras): {(prev >= thr).sum():>9,d} features")
    rich = mat.getnnz(axis=0)
    print("Features detectados por muestra (min/mediana/max): "
          f"{rich.min()} / {int(np.median(rich))} / {rich.max()}")
    return [str(i) for i in ids_s]


# ----------------------------------------------------------------------------
def sniff_sep(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    counts = {s: head.count(s) for s in [",", "\t", ";", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def inspect_metadata(path, sep, biom_ids, max_cols):
    section("2. METADATA")
    print(f"Archivo: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    sep = sep or sniff_sep(path)
    print(f"Separador usado: {sep!r}")
    df = pd.read_csv(path, sep=sep, low_memory=False)
    print(f"Forma: {df.shape[0]} filas x {df.shape[1]} columnas")

    # --- ¿qué columna contiene los IDs de muestra?
    biom_set = set(biom_ids)
    cands = []
    for col in df.columns:
        s = df[col]
        if is_text(s) or pd.api.types.is_integer_dtype(s):
            ov = int(s.astype(str).drop_duplicates().isin(biom_set).sum())  # IDs ÚNICOS, no filas
            if ov:
                cands.append((ov, col))
    cands.sort(reverse=True)
    section("2a. Columna de ID de muestra (solapamiento con los IDs del BIOM)")
    if cands:
        for ov, col in cands[:4]:
            print(f"    '{col}': {ov} de {len(biom_set)} IDs del BIOM encontrados "
                  f"({ov / len(biom_set):.0%})")
        best = cands[0][1]
        meta_ids = set(df[best].astype(str))
        print(f">>> Columna de ID probable: '{best}'")
        print(f"    BIOM sin metadata : {len(biom_set - meta_ids)}   "
              f"| metadata sin BIOM: {len(meta_ids - biom_set)}")
        print(f"    IDs duplicados en esa columna: {int(df[best].duplicated().sum())}")
    else:
        best = None
        print("    NINGUNA columna coincide con los IDs del BIOM (¿prefijo/sufijo distinto?).")
        print(f"    Ejemplos de IDs del BIOM: {sorted(biom_set)[:3]}")
        first_text = [c for c in df.columns if is_text(df[c])][:3]
        for c in first_text:
            print(f"    Ejemplos en columna '{c}': {df[c].astype(str).head(3).tolist()}")

    # --- listado de columnas
    section(f"2b. Columnas (primeras {max_cols} de {df.shape[1]}): nombre | dtype | % vacío | n únicos")
    for i, c in enumerate(df.columns[:max_cols]):
        s = df[c]
        print(f"{i:>4} {short(c, 42):<42} {str(s.dtype):<8} miss={s.isna().mean():>4.0%} "
              f"uniq={s.nunique(dropna=True)}")
    if df.shape[1] > max_cols:
        rest = [c for c in df.columns[max_cols:]]
        print(f"... {len(rest)} columnas más. Las que coinciden con palabras clave de ambiente/diseño:")
        for c in rest:
            if ENV_NAME_RE.search(str(c)) or GEO_TECH_NAME_RE.search(str(c)):
                print(f"     {short(c, 60)}  uniq={df[c].nunique(dropna=True)}")

    # --- columnas candidatas a 'ambiente' por NOMBRE
    section("2c. Columnas candidatas a 'ambiente' (por nombre) — top valores")
    shown = 0
    for c in df.columns:
        if ENV_NAME_RE.search(str(c)) and df[c].nunique(dropna=True) <= 100:
            vc = df[c].value_counts(dropna=False).head(12)
            print(f"\n[{c}]  (n únicos={df[c].nunique(dropna=True)}, vacíos={df[c].isna().sum()})")
            print(vc.to_string())
            shown += 1
            if shown >= 12:
                print("... (se omiten más columnas candidatas)")
                break
    if shown == 0:
        print("(ninguna columna con nombre de ambiente y ≤100 valores únicos)")

    # --- columnas candidatas por VALOR (lake/soil/...)
    section("2d. Columnas de texto con valores tipo lago/suelo (búsqueda por VALOR)")
    found = 0
    for c in df.columns:
        s = df[c]
        if not is_text(s) or s.nunique(dropna=True) > 500:
            continue
        hit = s[s.astype(str).str.contains(ENV_VALUE_RE, na=False)]
        if len(hit):
            vc = hit.value_counts().head(8)
            print(f"\n[{c}]  filas con coincidencia: {len(hit)}")
            print(vc.to_string())
            found += 1
            if found >= 15:
                print("... (se omiten más)")
                break
    if found == 0:
        print("(ninguna columna de texto con ≤500 valores únicos contiene lake/soil/lago/suelo/...)")
    return best


# ----------------------------------------------------------------------------
def inspect_tsv(path, biom_ids):
    section("3. TSV de abundancias (solo encabezado)")
    print(f"Archivo: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        first = f.readline().rstrip("\n").split("\t")
        n_lines = 2 + sum(1 for _ in f)
    print(f"Líneas: {n_lines:,d} (incluye encabezado)   Columnas: {len(header):,d}")
    print(f"Encabezado (primeros 6): {[short(h, 40) for h in header[:6]]}")
    print(f"Primera fila (primeros 4): {[short(x, 60) for x in first[:4]]}")
    biom_set = set(biom_ids)
    ov_header = sum(1 for h in header if h in biom_set)
    print(f"Campos del encabezado que coinciden con IDs de muestra del BIOM: {ov_header}")
    if ov_header == 0:
        col0 = pd.read_csv(path, sep="\t", usecols=[0], low_memory=False).iloc[:, 0].astype(str)
        print(f"Valores de la 1ª columna que coinciden con IDs del BIOM: {int(col0.isin(biom_set).sum())}")
    tax_cols = [h for h in header if re.search(r"(tax|lineage|domain|phylum|kingdom|genus|species)", h, re.I)]
    print(f"Columnas con pinta de taxonomía en el encabezado: {tax_cols[:6] or 'ninguna'}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Inspección solo lectura de BIOM + metadata (+TSV).")
    ap.add_argument("--biom", default="datos/all_abundances_598_260616.biom")
    ap.add_argument("--meta", default="datos/full_metadata.csv")
    ap.add_argument("--tsv", default="datos/comparative_abundances_complete.tsv")
    ap.add_argument("--sep", default=None, help="separador del metadato (por defecto se detecta)")
    ap.add_argument("--max-cols", type=int, default=120)
    ap.add_argument("--no-tsv", action="store_true")
    ap.add_argument("--out", default="inspect_report.txt")
    args = ap.parse_args()

    sys.stdout = Tee(args.out)
    print("REPORTE DE INSPECCIÓN — solo lectura")
    print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")

    biom_ids = []
    try:
        biom_ids = inspect_biom(Path(args.biom))
    except Exception:
        print("\n[ERROR en sección BIOM]\n" + traceback.format_exc(limit=3))

    try:
        inspect_metadata(Path(args.meta), args.sep, biom_ids, args.max_cols)
    except Exception:
        print("\n[ERROR en sección METADATA]\n" + traceback.format_exc(limit=3))

    if not args.no_tsv and Path(args.tsv).exists():
        try:
            inspect_tsv(Path(args.tsv), biom_ids)
        except Exception:
            print("\n[ERROR en sección TSV]\n" + traceback.format_exc(limit=3))

    print("\n=== FIN DEL REPORTE === (guardado en " + args.out + ")")
    sys.stdout = sys.stdout.out


if __name__ == "__main__":
    main()
