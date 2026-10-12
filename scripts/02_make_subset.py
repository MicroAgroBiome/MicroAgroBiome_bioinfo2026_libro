#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_make_subset.py — Subconjunto canónico del taller a partir del BIOM + metadata reales.

NO modifica los archivos de entrada. Tres modos:

  --explore   Qué agrupaciones son viables: tamaños, nº de estudios, y cuánto se CONFUNDE el
              grupo con el estudio de origen. No escribe datos.
  --dry-run   Plan completo para --group-col/--groups elegidos (qué muestras, qué dimensiones
              finales). No escribe datos.
  (ninguno)   Escribe en --out-dir:  taller_<rango>.biom, taller_metadata.csv, taller_samples.csv,
              subset_log.txt (con proveniencia: sha256 de las entradas, versiones, comando, huella)

Reproducibilidad: `taller_samples.csv` es la receta exacta (IDs públicos de SRA/ENA). Para
reconstruir el mismo subconjunto en cualquier máquina, sin depender del generador aleatorio:
    python scripts/02_make_subset.py --samples-file taller_samples.csv

Qué hace (en orden):
  1. Cruza BIOM con el metadata a nivel de MUESTRA (el CSV trae una fila por BGC).
  2. Filtra: --assay (p. ej. WGS), --filter COL=VAL (p. ej. plant_part=Rhizosphere) y --min-depth
     (lecturas al rango de trabajo).
  3. Elige --n-per-group muestras por grupo, repartiéndolas entre estudios (--study-col)
     con semilla fija.
  4. Colapsa los conteos al rango --rank (los conteos del BIOM son DIRECTOS, así que sumar
     especies dentro de su género es válido). Las lecturas asignadas a rangos más altos que
     --rank se descartan y se registra cuántas.
  5. Filtra features por prevalencia (--min-prevalence) dentro del subconjunto.

Decisiones del instructor (⚐) = parámetros: --group-col, --groups, --n-per-group, --rank,
--assay, --min-depth, --min-prevalence, --seed.

Ejemplos (desde la raíz del repo, con el entorno activado):
    python scripts/02_make_subset.py --explore
    python scripts/02_make_subset.py --dry-run  --group-col plant_part --groups Soil Rhizosphere
    python scripts/02_make_subset.py --group-col plant_part --groups Soil Rhizosphere
"""
import argparse
import hashlib
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RANK_CODES = ["k", "p", "c", "o", "f", "g", "s"]
RANK_NAMES = {"o": "order", "f": "family", "g": "genus", "s": "species"}
DEFAULT_CANDIDATES = ["plant_part", "crop_system", "host_name_family", "host_name_genus", "host_name",
                      "country", "continent", "assay_type", "seq_type"]


class Tee:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def section(t):
    print("\n" + "=" * 8 + f" {t} " + "=" * 8)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance(args):
    """Bloque de proveniencia: qué datos, qué versiones y qué comando produjeron el resultado."""
    import scipy
    try:
        import biom
        bv = biom.__version__
    except Exception:
        bv = "?"
    print("\nPROVENIENCIA")
    print("  comando : python " + " ".join(sys.argv))
    print(f"  versiones: python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
          f"scipy {scipy.__version__} | biom-format {bv}")
    for label, path in (("BIOM", args.biom), ("metadata", args.meta)):
        print(f"  sha256 {label:<8}: {sha256_file(path)}  ({Path(path).name}, {Path(path).stat().st_size / 1e6:.1f} MB)")


def sniff_sep(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    c = {s: head.count(s) for s in [",", "\t", ";", "|"]}
    b = max(c, key=c.get)
    return b if c[b] > 0 else ","


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def parse_taxonomy(tax):
    if tax is None:
        return [""] * 7
    parts = [x.strip() for x in tax.split(";")] if isinstance(tax, str) else [str(x).strip() for x in tax]
    names = [re.sub(r"^[A-Za-z]+__", "", p) for p in parts[:7]]
    return names + [""] * (7 - len(names))


def load_biom(path):
    from biom import load_table
    t = load_table(str(path))
    try:
        md_all = t.metadata(axis="observation")
    except Exception:
        md_all = None
    if md_all is None:
        raise SystemExit("El BIOM no tiene taxonomía embebida en los features; no puedo colapsar por rango.")
    names_list = [parse_taxonomy(m.get("taxonomy") if m else None) for m in md_all]
    rank_idx = np.array([max([i for i, n in enumerate(nm) if n] or [-1]) for nm in names_list])
    return dict(mat=t.matrix_data.tocsr(),
                sample_ids=[str(i) for i in t.ids("sample")],
                feature_ids=[str(i) for i in t.ids("observation")],
                names_list=names_list, rank_idx=rank_idx)


def load_sample_metadata(path, id_col):
    """El CSV puede traer una fila por BGC/bin: se queda con las columnas constantes por muestra."""
    df = pd.read_csv(path, sep=sniff_sep(path), low_memory=False)
    if id_col not in df.columns:
        raise SystemExit(f"No existe la columna de ID '{id_col}'. Columnas: {list(df.columns)}")
    df[id_col] = df[id_col].astype(str)
    g = df.groupby(id_col, sort=False)
    const = [c for c in df.columns if c != id_col and float((g[c].nunique(dropna=False) == 1).mean()) >= 0.99]
    sdf = df.drop_duplicates(subset=id_col).set_index(id_col)[const]
    sdf.index.name = "sample_id"
    return sdf, len(df), [c for c in df.columns if c not in const and c != id_col]


# ----------------------------------------------------------------------------
# Selección
# ----------------------------------------------------------------------------
def pick_spread(sub, n, study_col, rng):
    """Elige n muestras repartiéndolas lo más posible entre estudios (round-robin con semilla)."""
    study = sub[study_col].fillna("NA").astype(str) if study_col in sub.columns else pd.Series("NA", index=sub.index)
    pools = {k: list(rng.permutation(np.array(v.index))) for k, v in study.groupby(study)}
    order = [str(k) for k in rng.permutation(list(pools.keys()))]
    chosen = []
    while len(chosen) < n and any(pools.values()):
        for k in order:
            if pools[k]:
                chosen.append(str(pools[k].pop()))
                if len(chosen) == n:
                    break
    return chosen


def explore(cand_all, cand_f, cols, study_col, group_col, topn=12):
    section("EXPLORACIÓN: ¿qué agrupaciones son viables?")
    print(f"Muestras en BIOM ∩ metadata: {len(cand_all)}   |   tras filtros (--assay, --min-depth): {len(cand_f)}")
    for col in cols:
        if col not in cand_all.columns:
            continue
        s_all = cand_all[col].fillna("NA").astype(str)
        s_f = cand_f[col].fillna("NA").astype(str)
        rows = []
        for cat, n_all in s_all.value_counts().head(topn).items():
            sub = cand_f[s_f == cat]
            if len(sub):
                vc = sub[study_col].fillna("NA").value_counts()
                rows.append((cat, n_all, len(sub), len(vc), int((vc >= 3).sum()), vc.iloc[0] / len(sub)))
            else:
                rows.append((cat, n_all, 0, 0, 0, np.nan))
        out = pd.DataFrame(rows, columns=["categoría", "n_sin_filtros", "n_tras_filtros", "n_estudios",
                                          "estudios≥3", "estudio_mayor"])
        print(f"\n[{col}]  ({s_all.nunique()} categorías; se muestran las {min(topn, s_all.nunique())} mayores)")
        print(out.to_string(index=False, formatters={"estudio_mayor": lambda v: "-" if pd.isna(v) else f"{v:.0%}"}))

    if group_col in cand_f.columns:
        section(f"CONFUSIÓN CON ESTUDIO para '{group_col}' (tras filtros)")
        g = cand_f[group_col].fillna("NA").astype(str)
        st = cand_f[study_col].fillna("NA").astype(str)
        ct = pd.crosstab(st, g)
        keep = [c for c in ct.columns if ct[c].sum() >= 5]
        ct = ct[keep]
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index].head(15)
        print(f"Muestras por estudio (filas, top 15) x grupo (columnas con ≥5 muestras):\n{ct.to_string()}")
        sets = {c: set(st[g == c]) for c in keep}
        mat = pd.DataFrame([[len(sets[a] & sets[b]) for b in keep] for a in keep], index=keep, columns=keep)
        print("\nEstudios en común entre grupos (diagonal = nº de estudios del grupo).")
        print("Si dos grupos NO comparten ningún estudio, su diferencia está 100 % confundida con el estudio:")
        print(mat.to_string())


# ----------------------------------------------------------------------------
# Construcción
# ----------------------------------------------------------------------------
def feature_labels(keys, R):
    names = [k[R] for k in keys]
    cnt = Counter(names)
    seen, labels = Counter(), []
    for k, nm in zip(keys, names):
        if cnt[nm] == 1:
            labels.append(nm)
        else:
            parent = k[R - 1] if R > 0 and k[R - 1] else "NA"
            lab = f"{nm}|{parent}"
            seen[lab] += 1
            labels.append(lab if seen[lab] == 1 else f"{lab}#{seen[lab]}")
    return labels, sum(1 for v in cnt.values() if v > 1)


def collapse(mat, names_list, rank_idx, R, sample_cols):
    rows, gidx, keys = [], [], {}
    for i, (nm, ri) in enumerate(zip(names_list, rank_idx)):
        if ri >= R and nm[R] != "":
            gidx.append(keys.setdefault(tuple(nm[: R + 1]), len(keys)))
            rows.append(i)
    G = sp.csr_matrix((np.ones(len(rows)), (gidx, rows)), shape=(len(keys), mat.shape[0]))
    sub = mat.tocsc()[:, sample_cols].tocsr()
    return (G @ sub).tocsr(), list(keys.keys())


def build(args, biom, meta):
    R = RANK_CODES.index(args.rank)
    mat, ids = biom["mat"], biom["sample_ids"]
    kept = np.where([(ri >= R and nm[R] != "") for nm, ri in zip(biom["names_list"], biom["rank_idx"])])[0]
    depth_all = pd.Series(np.asarray(mat.sum(axis=0)).ravel(), index=ids)
    depth_rank = pd.Series(np.asarray(mat[kept].sum(axis=0)).ravel(), index=ids)

    cand_all = meta.loc[meta.index.intersection(ids)].copy()
    cand_all["depth_all"] = depth_all.reindex(cand_all.index)
    cand_all["depth_rank"] = depth_rank.reindex(cand_all.index)

    log = [f"BIOM: {mat.shape[0]:,d} features x {mat.shape[1]} muestras | metadata (nivel muestra): {len(meta)} muestras",
           f"BIOM ∩ metadata: {len(cand_all)} muestras ({len(set(ids) - set(meta.index))} del BIOM sin metadata)"]
    cand_f = cand_all
    if args.assay.lower() != "any":
        if args.assay_col not in cand_f.columns:
            raise SystemExit(f"No existe la columna --assay-col '{args.assay_col}'")
        before = len(cand_f)
        cand_f = cand_f[cand_f[args.assay_col] == args.assay]
        log.append(f"Filtro {args.assay_col} == {args.assay!r}: {before} -> {len(cand_f)}")
    for flt in (args.filter or []):
        col, _, vals = flt.partition("=")
        vs = [v.strip() for v in vals.split(",") if v.strip()]
        if not vs or col not in cand_f.columns:
            raise SystemExit(f"--filter inválido: '{flt}' (usa COLUMNA=VALOR[,VALOR2]; columnas: {list(cand_f.columns)[:12]}...)")
        before = len(cand_f)
        cand_f = cand_f[cand_f[col].isin(vs)]
        log.append(f"Filtro {col} in {vs}: {before} -> {len(cand_f)}")
    before = len(cand_f)
    cand_f = cand_f[cand_f["depth_rank"] >= args.min_depth]
    log.append(f"Filtro profundidad al rango '{RANK_NAMES[args.rank]}' >= {args.min_depth:,d}: {before} -> {len(cand_f)}")
    return R, cand_all, cand_f, log, depth_all, depth_rank


def plan_and_build(args, biom, meta, R, cand_all, cand_f, log, depth_all, depth_rank):
    group_of = {}
    if args.samples_file:
        sf = pd.read_csv(args.samples_file)
        if "sample_id" not in sf.columns:
            raise SystemExit(f"{args.samples_file} debe tener una columna 'sample_id'")
        chosen = [str(x) for x in sf["sample_id"]]
        missing = [x for x in chosen if x not in cand_all.index]
        if missing:
            raise SystemExit(f"{len(missing)} muestra(s) de --samples-file no están en BIOM ∩ metadata: {missing[:5]}")
        bad = [x for x in chosen if x not in cand_f.index]
        log.append(f"\nMuestras tomadas de {args.samples_file}: {len(chosen)} (sin sorteo)")
        if bad:
            log.append(f"  ⚠ {len(bad)} no cumplen los filtros actuales (--assay/--min-depth): {bad[:5]}")
        if "group" in sf.columns:
            group_of = dict(zip(chosen, sf["group"].astype(str)))
            log.append("  por grupo: " + str(pd.Series(group_of).value_counts().to_dict()))
    else:
        if not args.group_col or not args.groups:
            raise SystemExit("Indica --group-col y --groups (usa --explore para ver las opciones).")
        if args.group_col not in cand_f.columns:
            raise SystemExit(f"No existe --group-col '{args.group_col}'.")
        avail = cand_f[args.group_col].value_counts()
        missing = [g for g in args.groups if g not in avail.index]
        if missing:
            raise SystemExit(f"Grupos sin muestras tras los filtros: {missing}. Disponibles: {avail.head(15).to_dict()}")
        rng = np.random.default_rng(args.seed)
        chosen = []
        log.append(f"\nSelección (semilla={args.seed}, n por grupo={args.n_per_group}, reparto entre '{args.study_col}'):")
        for g in args.groups:
            sub = cand_f[cand_f[args.group_col] == g]
            k = min(args.n_per_group, len(sub))
            pick = pick_spread(sub, k, args.study_col, rng)
            n_st = sub.loc[pick, args.study_col].fillna("NA").nunique()
            warn = "" if k == args.n_per_group else f"   ⚠ solo hay {len(sub)} (< {args.n_per_group}): grupo desbalanceado"
            log.append(f"  {g:<18} disponibles={len(sub):>4}  elegidas={k:>3}  estudios representados={n_st}{warn}")
            chosen += pick
            group_of.update({x: g for x in pick})

    coll, keys = collapse(biom["mat"], biom["names_list"], biom["rank_idx"], R,
                          [biom["sample_ids"].index(s) for s in chosen])
    assert np.allclose(np.asarray(coll.sum(axis=0)).ravel(), depth_rank.loc[chosen].values), \
        "Conservación de lecturas violada al colapsar"
    n_feat_all = coll.shape[0]
    thr = int(np.ceil(args.min_prevalence * len(chosen)))
    prev = np.asarray((coll > 0).sum(axis=1)).ravel()
    keep = np.where(prev >= thr)[0]
    coll = coll[keep]
    keys = [keys[i] for i in keep]
    labels, n_dup = feature_labels(keys, R)
    depth_final = np.asarray(coll.sum(axis=0)).ravel()

    log.append(f"\nColapso a '{RANK_NAMES[args.rank]}': {n_feat_all:,d} taxa -> {len(keys):,d} tras prevalencia >= "
               f"{args.min_prevalence:.0%} (>= {thr} de {len(chosen)} muestras)")
    if n_dup:
        log.append(f"  Nombres repetidos entre linajes distintos: {n_dup} (se desambiguaron con '|<padre>')")
    kept_frac = depth_rank.loc[chosen] / depth_all.loc[chosen]
    log.append(f"Lecturas resueltas hasta '{RANK_NAMES[args.rank]}' (fracción de las lecturas del BIOM): "
               f"min/mediana/max = {kept_frac.min():.1%} / {kept_frac.median():.1%} / {kept_frac.max():.1%}")
    log.append(f"Lecturas retenidas tras el filtro de prevalencia (fracción de las del rango): "
               f"mediana = {np.median(depth_final / depth_rank.loc[chosen].values):.1%}")

    meta_out = meta.loc[chosen].copy()
    meta_out["depth_all_biom"] = depth_all.loc[chosen].values
    meta_out["depth_rank"] = depth_rank.loc[chosen].values
    meta_out["fraction_resolved"] = kept_frac.values
    meta_out["depth_final"] = depth_final

    taxonomy = [[f"{c}__{k[j]}" if j <= R else f"{c}__" for j, c in enumerate(RANK_CODES)] for k in keys]
    # Huella del contenido: taxa + muestras + conteos. Debe coincidir en cualquier máquina con las mismas entradas.
    h = hashlib.sha256()
    h.update("|".join(labels).encode("utf-8"))
    h.update("|".join(chosen).encode("utf-8"))
    h.update(np.ascontiguousarray(np.rint(coll.toarray()).astype(np.int64)).tobytes())
    fingerprint = h.hexdigest()
    log.append(f"Huella del contenido (sha256 de taxa+muestras+conteos): {fingerprint}")
    return dict(chosen=chosen, table=coll, labels=labels, taxonomy=taxonomy, meta_out=meta_out, log=log,
                group_of=group_of, fingerprint=fingerprint)


def quality_warnings(cand, selected, lat="latitude", lon="longitude"):
    """Avisa de coordenadas imposibles (cada columna por separado)."""
    out = []
    for col, lim in ((lat, 90), (lon, 180)):
        if col in cand.columns:
            bad = cand[cand[col].abs() > lim]
            if len(bad):
                sel = set(selected)
                out.append(f"⚠ '{col}' fuera de rango (|{col}|>{lim}) en {len(bad)} muestra(s): "
                           f"{list(bad.index[:5])}; en la selección: {[i for i in bad.index if i in sel]}")
    return out


def write_outputs(args, res, out_dir):
    from biom.table import Table
    from biom.util import biom_open
    out_dir.mkdir(parents=True, exist_ok=True)
    name = RANK_NAMES[args.rank]
    t = Table(res["table"], res["labels"], res["chosen"], observation_metadata=[{"taxonomy": x} for x in res["taxonomy"]],
              sample_metadata=[{"Id": s} for s in res["chosen"]])
    biom_path = out_dir / f"taller_{name}.biom"
    with biom_open(str(biom_path), "w") as f:
        t.to_hdf5(f, "02_make_subset.py")
    meta_path = out_dir / "taller_metadata.csv"
    res["meta_out"].to_csv(meta_path)
    samples_path = out_dir / "taller_samples.csv"
    pd.DataFrame({"sample_id": res["chosen"], "group": [res["group_of"].get(x, "") for x in res["chosen"]]}
                 ).to_csv(samples_path, index=False)
    return biom_path, meta_path, samples_path


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Subconjunto canónico del taller (solo lectura de entradas).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--biom", default="datos/all_abundances_598_260616.biom")
    ap.add_argument("--meta", default="datos/full_metadata.csv")
    ap.add_argument("--id-col", default="sample")
    ap.add_argument("--study-col", default="doi")
    ap.add_argument("--assay-col", default="assay_type")
    ap.add_argument("--assay", default="WGS", help="valor de --assay-col a conservar; 'any' = no filtrar")
    ap.add_argument("--filter", action="append", metavar="COL=VAL[,VAL]",
                    help="conserva solo filas con COL en esos valores (repetible). Ej.: --filter plant_part=Rhizosphere")
    ap.add_argument("--rank", default="g", choices=list(RANK_NAMES))
    ap.add_argument("--min-depth", type=int, default=300_000, help="lecturas mínimas al rango de trabajo")
    ap.add_argument("--group-col")
    ap.add_argument("--groups", nargs="+")
    ap.add_argument("--n-per-group", type=int, default=20)
    ap.add_argument("--min-prevalence", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples-file", help="CSV con columna 'sample_id' (y opcional 'group'): usa EXACTAMENTE esas "
                    "muestras, sin sorteo (reproducción exacta; no depende de la versión de NumPy)")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    ap.add_argument("--out-dir", default="datos/subset")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    report = ("subset_explore_report.txt" if args.explore else
              "subset_dryrun_report.txt" if args.dry_run else str(out_dir / "subset_log.txt"))
    sys.stdout = Tee(report)
    try:
        print(f"02_make_subset.py | python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")
        print("Argumentos: " + " ".join(f"{k}={v}" for k, v in vars(args).items() if v not in (None, False)))
        provenance(args)
        biom = load_biom(Path(args.biom))
        meta, n_rows, varying = load_sample_metadata(Path(args.meta), args.id_col)
        R, cand_all, cand_f, log, depth_all, depth_rank = build(args, biom, meta)
        print(f"Metadata: {n_rows:,d} filas -> {len(meta)} muestras (columnas por BGC/bin ignoradas: {varying})")
        for line in log:
            print(line)

        if args.explore:
            explore(cand_all, cand_f, args.candidates, args.study_col, args.group_col or "plant_part")
            for w in quality_warnings(cand_all, []):
                print("\n" + w)
        else:
            res = plan_and_build(args, biom, meta, R, cand_all, cand_f, [], depth_all, depth_rank)
            section("PLAN")
            print("\n".join(res["log"]))
            n_f, n_s = res["table"].shape
            print(f"\nDimensiones finales: {n_f:,d} taxa ({RANK_NAMES[args.rank]}) x {n_s} muestras")
            print("Tamaño denso aprox.: " + f"{n_f * n_s * 8 / 1e6:.1f} MB")
            for w in quality_warnings(cand_all, res["chosen"]):
                print("\n" + w)
            if args.dry_run:
                print("\n(--dry-run: no se escribió ningún archivo)")
            else:
                bp, mp, sp_ = write_outputs(args, res, out_dir)
                print(f"\nEscrito:\n  {bp}  ({bp.stat().st_size / 1e6:.2f} MB)\n  {mp}  ({mp.stat().st_size / 1e3:.1f} KB)"
                      f"\n  {sp_}  <- la RECETA exacta: versiónala en git"
                      f"\n  {out_dir / 'subset_log.txt'}")
    except SystemExit as e:
        print(f"\n[!] {e}")
    except Exception:
        print("\n[ERROR]\n" + traceback.format_exc(limit=4))
    finally:
        sys.stdout = sys.stdout.out


if __name__ == "__main__":
    main()
