#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_inspect_structure.py — Segunda inspección (SOLO LECTURA), guiada por el primer reporte.

Responde las preguntas que decidirán el diseño del subconjunto del taller:
  A. ¿Qué hay en el "metadata" a nivel de MUESTRA (el archivo trae una fila por BGC)?
     ¿Hay muestras de lago? ¿Qué pasa con las muestras del BIOM sin metadata?
  B. ¿Qué rangos taxonómicos mezcla el BIOM? ¿Los conteos son ACUMULADOS por clado
     (el padre incluye a sus hijos) o DIRECTOS? ¿Cómo cambia la profundidad según el rango?
  C. ¿Cómo se relaciona el TSV con el BIOM?

Uso (desde la raíz del repo, con el entorno del curso activado):
    python scripts/01_inspect_structure.py
    python scripts/01_inspect_structure.py --no-tsv        # omitir el TSV de 163 MB
    python scripts/01_inspect_structure.py --id-col sample --cat-max 40

Escribe `inspect_structure_report.txt`. No modifica ningún dato.
Nunca convierte la tabla completa a denso.
"""
import argparse
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RANK_LABELS = {-1: "sin rango", 0: "k (dominio)", 1: "p (filo)", 2: "c (clase)",
               3: "o (orden)", 4: "f (familia)", 5: "g (género)", 6: "s (especie)"}
LAKE_RE = (r"(?:lake|lago|laguna|fresh|water|aquatic|river|\brio\b|pond|reservoir|"
           r"wetland|sediment|pelagic|limnet)")


class Tee:
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


def short(x, n=80):
    x = str(x)
    return x if len(x) <= n else x[: n - 3] + "..."


def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def sniff_sep(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    counts = {s: head.count(s) for s in [",", "\t", ";", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


# ----------------------------------------------------------------------------
# BIOM: carga silenciosa + parseo de taxonomía
# ----------------------------------------------------------------------------
def parse_taxonomy(tax):
    """Devuelve 7 nombres (k,p,c,o,f,g,s); cadena vacía si el rango no está definido."""
    if tax is None:
        return [""] * 7
    parts = [x.strip() for x in tax.split(";")] if isinstance(tax, str) else [str(x).strip() for x in tax]
    names = [re.sub(r"^[A-Za-z]+__", "", p) for p in parts[:7]]
    return names + [""] * (7 - len(names))


def load_biom(path):
    from biom import load_table
    t = load_table(str(path))
    mat = t.matrix_data.tocsr()  # features x muestras (disperso)
    sample_ids = [str(i) for i in t.ids("sample")]
    feature_ids = [str(i) for i in t.ids("observation")]
    try:
        md_all = t.metadata(axis="observation")
    except Exception:
        md_all = None
    if md_all is None:
        tax_list = [None] * len(feature_ids)
    else:
        tax_list = [(m.get("taxonomy") if m else None) for m in md_all]
    names_list = [parse_taxonomy(x) for x in tax_list]
    rank_idx = np.array([max([i for i, n in enumerate(nm) if n] or [-1]) for nm in names_list])
    has_tax = any(x is not None for x in tax_list)
    return dict(mat=mat, sample_ids=sample_ids, feature_ids=feature_ids,
                names_list=names_list, rank_idx=rank_idx, has_tax=has_tax)


# ----------------------------------------------------------------------------
# A. Metadata a nivel de muestra
# ----------------------------------------------------------------------------
def inspect_samples(meta_path, biom, id_col, cat_max, cross):
    section("A. METADATA A NIVEL DE MUESTRA")
    sep = sniff_sep(meta_path)
    df = pd.read_csv(meta_path, sep=sep, low_memory=False)
    print(f"Archivo: {meta_path} | filas={len(df):,d} columnas={df.shape[1]} | col. de ID: '{id_col}'")
    if id_col not in df.columns:
        print(f"[!] No existe la columna '{id_col}'. Columnas: {list(df.columns)}")
        return
    ids_meta = df[id_col].astype(str)
    n_samples = ids_meta.nunique()
    print(f"Muestras únicas en el metadata: {n_samples} | filas por muestra (mediana): "
          f"{int(ids_meta.value_counts().median())}")

    # columnas constantes dentro de cada muestra = nivel muestra
    g = df.groupby(id_col, sort=False)
    const, varying = [], []
    for c in df.columns:
        if c == id_col:
            continue
        frac = float((g[c].nunique(dropna=False) == 1).mean())
        (const if frac >= 0.99 else varying).append(c)
    print(f"\nColumnas de NIVEL MUESTRA ({len(const)}): {const}")
    print(f"Columnas que VARÍAN dentro de la muestra, nivel fila/BGC/bin ({len(varying)}): {varying}")

    sdf = df.drop_duplicates(subset=id_col).set_index(id_col)[const]
    sdf.index = sdf.index.astype(str)
    print(f"\nTabla a nivel de muestra: {sdf.shape[0]} muestras x {sdf.shape[1]} columnas")

    section("A1. Valores por columna (UNA FILA POR MUESTRA, no por BGC)")
    for c in sdf.columns:
        s = sdf[c]
        nun = s.nunique(dropna=True)
        miss = int(s.isna().sum())
        if nun <= cat_max:
            print(f"\n[{c}]  n únicos={nun}  vacíos={miss}")
            print(s.value_counts(dropna=False).head(12).to_string())
        elif pd.api.types.is_numeric_dtype(s):
            print(f"\n[{c}]  numérica  vacíos={miss}  min/mediana/max = "
                  f"{s.min():.6g} / {s.median():.6g} / {s.max():.6g}")
        else:
            print(f"\n[{c}]  n únicos={nun}  vacíos={miss}  top 5:")
            print(s.value_counts().head(5).to_string())

    a, b = cross
    if a in sdf.columns and b in sdf.columns:
        section(f"A2. Tabla cruzada por muestra: {a} x {b}")
        print(pd.crosstab(sdf[a].fillna("NA"), sdf[b].fillna("NA")).to_string())

    section("A3. ¿Dónde hay lagos? Búsqueda en TODAS las columnas de texto a nivel de muestra")
    found = 0
    for c in sdf.columns:
        s = sdf[c]
        if not is_text(s):
            continue
        m = s.astype(str).str.contains(LAKE_RE, case=False, regex=True, na=False)
        if m.any():
            found += 1
            print(f"\n[{c}]  muestras con coincidencia: {int(m.sum())}")
            print(s[m].value_counts().head(8).to_string())
    if not found:
        print("NINGUNA columna de texto a nivel de muestra contiene lake/lago/fresh/water/aquatic/river/"
              "pond/reservoir/wetland/sediment/pelagic/limnet.")

    section("A4. Muestras del BIOM SIN metadata")
    biom_ids = set(biom["sample_ids"])
    missing = sorted(biom_ids - set(sdf.index))
    extra = sorted(set(sdf.index) - biom_ids)
    print(f"BIOM sin metadata: {len(missing)} | metadata sin BIOM: {len(extra)}")
    sums = dict(zip(biom["sample_ids"], np.asarray(biom["mat"].sum(axis=0)).ravel()))
    if missing:
        print(f"IDs (primeros 25): {missing[:25]}")
        pm = Counter(i[:3] for i in missing)
        po = Counter(i[:3] for i in biom_ids - set(missing))
        print(f"Prefijos de ID  sin metadata: {dict(pm)}  |  con metadata: {dict(po)}")
        dm = np.array([sums[i] for i in missing])
        do = np.array([sums[i] for i in biom_ids - set(missing)])
        print(f"Suma de lecturas (mediana)  sin metadata: {np.median(dm):.6g}  |  con metadata: {np.median(do):.6g}")


# ----------------------------------------------------------------------------
# B. Rangos taxonómicos y tipo de conteo
# ----------------------------------------------------------------------------
def parent_child_test(biom, rp, rc):
    mat, names_list, rank_idx = biom["mat"], biom["names_list"], biom["rank_idx"]
    kids_by_key, parent_row = {}, {}
    for i, (nm, r) in enumerate(zip(names_list, rank_idx)):
        key = tuple(nm[: rp + 1])
        if r == rc:
            kids_by_key.setdefault(key, []).append(i)
        elif r == rp:
            parent_row[key] = i
    n_par = n_pairs = n_ge = 0
    ratios = []
    for key, prow in parent_row.items():
        kids = kids_by_key.get(key)
        if not kids:
            continue
        n_par += 1
        pv = mat[prow].toarray().ravel()
        cs = np.asarray(mat[kids].sum(axis=0)).ravel()
        ok = cs > 0
        if ok.any():
            n_pairs += int(ok.sum())
            n_ge += int((pv[ok] >= cs[ok]).sum())
            ratios.append(pv[ok] / cs[ok])
    return n_par, n_pairs, n_ge, (np.concatenate(ratios) if ratios else np.array([]))


def inspect_ranks(biom):
    section("B. RANGOS TAXONÓMICOS DEL BIOM")
    if not biom["has_tax"]:
        print("El BIOM no tiene taxonomía embebida; no se puede analizar por rango.")
        return
    mat, rank_idx = biom["mat"], biom["rank_idx"]
    n_feat, n_samp = mat.shape
    prev = mat.getnnz(axis=1)
    print(f"{'rango':<14}{'n features':>11}{'suma/muestra (mediana)':>25}"
          f"{'prev≥5%':>9}{'≥10%':>8}{'≥25%':>8}")
    sums_by_rank = {}
    for r in range(-1, 7):
        rows = np.where(rank_idx == r)[0]
        if not len(rows):
            continue
        s = np.asarray(mat[rows].sum(axis=0)).ravel()
        sums_by_rank[r] = s
        pr = prev[rows]
        t5, t10, t25 = (int(np.ceil(p / 100 * n_samp)) for p in (5, 10, 25))
        print(f"{RANK_LABELS[r]:<14}{len(rows):>11,d}{np.median(s):>25,.0f}"
              f"{(pr >= t5).sum():>9,d}{(pr >= t10).sum():>8,d}{(pr >= t25).sum():>8,d}")
    print("(Pista: si la suma por muestra es casi igual en todos los rangos, cada rango reparte las "
          "MISMAS lecturas ⇒ conteos acumulados.)")

    section("B1. ¿Conteos ACUMULADOS por clado o DIRECTOS? (padre ≥ suma de sus hijos)")
    verdicts = []
    for rp, rc in ((1, 2), (3, 4), (5, 6)):
        n_par, n_pairs, n_ge, ratios = parent_child_test(biom, rp, rc)
        if n_pairs == 0:
            print(f"{RANK_LABELS[rp]} -> {RANK_LABELS[rc]}: sin pares comparables")
            continue
        frac = n_ge / n_pairs
        verdicts.append(frac)
        print(f"{RANK_LABELS[rp]} -> {RANK_LABELS[rc]}: {n_par} padres con hijos | pares (padre, muestra)="
              f"{n_pairs:,d} | padre ≥ suma de hijos en {frac:.1%} | razón padre/hijos mediana="
              f"{np.median(ratios):.3g}")
    if verdicts:
        v = float(np.mean(verdicts))
        if v >= 0.99:
            msg = ("ACUMULADOS POR CLADO: el conteo de un padre incluye a sus hijos. "
                   "NO se pueden sumar features de distintos rangos (doble conteo).")
        elif v <= 0.5:
            msg = ("DIRECTOS: cada feature son lecturas asignadas a ESE nodo. Sumar es válido, "
                   "pero mezclar rangos sigue siendo conceptualmente confuso para diversidad.")
        else:
            msg = "MIXTO / INDETERMINADO: revisar cómo se generó el BIOM."
        print(f">>> Interpretación automática: {msg}")

    section("B2. Profundidad por muestra según el conjunto de features usado")
    thr = (1e4, 5e4, 1e5, 3e5, 1e6)
    sets = [("todas las filas", np.asarray(mat.sum(axis=0)).ravel())]
    for r in (5, 6):
        if r in sums_by_rank:
            sets.append((f"solo {RANK_LABELS[r]}", sums_by_rank[r]))
    print(f"{'conjunto':<22}{'min':>12}{'p5':>12}{'mediana':>14}{'p95':>14}{'max':>14}   "
          + "  ".join(f"≥{int(t):,d}".rjust(10) for t in thr))
    for label, s in sets:
        q = np.percentile(s, [0, 5, 50, 95, 100])
        print(f"{label:<22}" + "".join(f"{v:>12,.0f}" if i < 3 else f"{v:>14,.0f}" for i, v in enumerate(q))
              + "   " + "  ".join(f"{int((s >= t).sum()):>10,d}" for t in thr))
    print(f"(n muestras = {n_samp})")


# ----------------------------------------------------------------------------
# C. TSV
# ----------------------------------------------------------------------------
def inspect_tsv(path, biom):
    section("C. TSV vs BIOM")
    print(f"Archivo: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
    sset = set(biom["sample_ids"])
    non_samples = [h for h in header if h not in sset]
    print(f"Columnas: {len(header)} = {len(header) - len(non_samples)} muestras (en el BIOM) + "
          f"{len(non_samples)} otras: {non_samples}")
    cols = [c for c in ("Name", "Rank", "TID") if c in header]
    if not cols:
        print("No hay columnas Name/Rank/TID; se omite la comparación.")
        return
    t = pd.read_csv(path, sep="\t", usecols=cols, dtype=str, low_memory=False)
    print(f"Filas de datos: {len(t):,d}")
    if "Rank" in t.columns:
        print("Rank en el TSV:\n" + t["Rank"].value_counts().head(15).to_string())
    if "TID" in t.columns:
        fset = set(biom["feature_ids"])
        tid = t["TID"].astype(str)
        print(f"\nFeatures del BIOM presentes en el TSV (por TID): {len(fset & set(tid)):,d} de {len(fset):,d}")
        print(f"Filas del TSV cuyo TID NO está en el BIOM: {int((~tid.isin(fset)).sum()):,d}")
        if "Rank" in t.columns and biom["has_tax"]:
            rmap = dict(zip(tid, t["Rank"]))
            parsed = [RANK_LABELS[r].split()[0] for r in biom["rank_idx"]]
            tsv_rank = [rmap.get(f, "(no está)") for f in biom["feature_ids"]]
            print("\nRango parseado de la taxonomía del BIOM (filas) vs 'Rank' del TSV (columnas):")
            print(pd.crosstab(pd.Series(parsed, name="BIOM"), pd.Series(tsv_rank, name="TSV")).to_string())


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Segunda inspección solo lectura.")
    ap.add_argument("--biom", default="datos/all_abundances_598_260616.biom")
    ap.add_argument("--meta", default="datos/full_metadata.csv")
    ap.add_argument("--tsv", default="datos/comparative_abundances_complete.tsv")
    ap.add_argument("--id-col", default="sample")
    ap.add_argument("--cat-max", type=int, default=40)
    ap.add_argument("--cross", nargs=2, default=["plant_part", "crop_system"])
    ap.add_argument("--no-tsv", action="store_true")
    ap.add_argument("--out", default="inspect_structure_report.txt")
    args = ap.parse_args()

    sys.stdout = Tee(args.out)
    print("REPORTE 2 — estructura (solo lectura)")
    print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")

    try:
        biom = load_biom(Path(args.biom))
        print(f"BIOM: {biom['mat'].shape[0]:,d} features x {biom['mat'].shape[1]} muestras")
    except Exception:
        print("[ERROR cargando el BIOM]\n" + traceback.format_exc(limit=3))
        sys.stdout = sys.stdout.out
        return

    for fn, a in ((inspect_samples, (Path(args.meta), biom, args.id_col, args.cat_max, args.cross)),
                  (inspect_ranks, (biom,))):
        try:
            fn(*a)
        except Exception:
            print(f"\n[ERROR en {fn.__name__}]\n" + traceback.format_exc(limit=3))
    if not args.no_tsv and Path(args.tsv).exists():
        try:
            inspect_tsv(Path(args.tsv), biom)
        except Exception:
            print("\n[ERROR en inspect_tsv]\n" + traceback.format_exc(limit=3))

    print("\n=== FIN DEL REPORTE === (guardado en " + args.out + ")")
    sys.stdout = sys.stdout.out


if __name__ == "__main__":
    main()
