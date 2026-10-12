"""
Agrega etiquetas de Quarto (#| label / #| fig-cap) a las celdas de los notebooks
cuyas salidas se incrustan en el libro con {{< embed >}}.

Uso (desde la raíz del repo):
    python scripts/etiquetar_notebooks.py

- Identifica cada celda por un fragmento ÚNICO de su código (no por su posición),
  así que sigue funcionando si se agregan o mueven celdas.
- Es idempotente: si la celda ya tiene '#| label:', no la toca.
- NO modifica salidas: las líneas '#|' son comentarios para Python
  (en Colab/VSCode se ven como comentarios y no cambian nada al ejecutar).
- Si un fragmento no aparece o aparece en más de una celda, se detiene con error.
"""
import json
import sys
from pathlib import Path

# notebook -> lista de (fragmento único del código, label, caption)
# Prefijo fig- = figura, tbl- = tabla (Quarto los numera y permite @referencias)
LABELS = {
    "notebooks/leccion_1_introduccion_datos.ipynb": [
        ('ax.set_title("Media y mediana en datos asimétricos")', "fig-l1-media-mediana",
         "Media y mediana de los conteos de *Streptomyces*: en datos asimétricos la media se aleja de la mediana."),
        ("pd.crosstab(metadata_df[\"host_name_family\"], metadata_df[\"host_name\"])", "tbl-l1-plantas",
         "Plantas hospederas incluidas en cada familia."),
        ('ax.set_title("Muestras por estudio y familia")', "fig-l1-estudio-familia",
         "Número de muestras por estudio y familia de planta: casi cada estudio muestreó una sola familia."),
        ('ax.set_title("Los 10 géneros más abundantes")', "fig-l1-top-generos",
         "Los 10 géneros con mayor abundancia relativa media, coloreados por dominio."),
        ('ax.set_title("Las 10 muestras con más ADN humano")', "fig-l1-homo",
         "Porcentaje de lecturas asignadas a *Homo* en las 10 muestras más afectadas."),
        ('ax.set_ylabel("Abundancia relativa")', "fig-l1-filos",
         "Composición por filo de cada muestra, ordenadas por familia de planta (etiqueta = estudio)."),
        ('ax.set_title("Profundidad de secuenciación")', "fig-l1-profundidad",
         "Distribución de la profundidad de secuenciación (lecturas de Bacteria + Archaea por muestra)."),
        ('ax.set_title("Prevalencia de géneros")', "fig-l1-prevalencia",
         "Prevalencia de géneros: en cuántas de las 75 muestras aparece cada uno."),
        ('ax.set_title("La riqueza depende de la profundidad")', "fig-l1-riqueza-profundidad",
         "Riqueza de géneros frente a la profundidad de secuenciación."),
        ('ax.set_title("Diversidad α por familia")', "fig-l1-shannon-familia",
         "Diversidad de Shannon por familia de planta."),
        ('ax.set_title("Diversidad α por estudio (color = familia de la planta)")', "fig-l1-shannon-estudio",
         "Diversidad de Shannon por estudio, coloreada por familia de planta."),
    ],
    "notebooks/leccion_2_clustering.ipynb": [
        ('ax.set_title("Distancias entre pares de muestras")', "fig-l2-distancias",
         "Distribución de las distancias de Bray-Curtis entre todos los pares de muestras."),
        ('axes[1].set_title("Coloreado por estudio")', "fig-l2-pcoa",
         "PCoA sobre Bray-Curtis: las mismas muestras coloreadas por familia de planta (izq.) y por estudio (der.)."),
        ('ax.set_title("Dendrograma (enlace promedio)")', "fig-l2-dendrograma",
         "Dendrograma con enlace promedio (etiqueta = estudio; color = familia de planta)."),
        ("g = sns.clustermap(dist_df", "fig-l2-clustermap",
         "Matriz de distancias reordenada según el dendrograma, con franjas de familia y estudio."),
        ('ax.set_title("Elección de k")', "fig-l2-eleccion-k",
         "Silhouette promedio de k-means para k = 2 a 8."),
        ('ax.set_title(f"k-means con k={best_k}")', "fig-l2-kmeans",
         "Clusters de k-means sobre las dos primeras coordenadas del PCoA."),
        ('ax.set_title("Diagrama de silhouette")', "fig-l2-silhouette",
         "Diagrama de silhouette: una barra por muestra, agrupadas por cluster."),
        ("ari_table = pd.DataFrame(", "tbl-l2-ari",
         "Índice de Rand ajustado entre los clusters y tres variables del metadato."),
        ("permanova_table = pd.DataFrame(", "tbl-l2-permanova",
         "PERMANOVA (Bray-Curtis, 75 muestras): variación explicada por cada variable."),
        ('ax.set_title("Distancia entre pares de muestras según lo que comparten")', "fig-l2-pares",
         "Distancias entre pares de muestras según si comparten estudio y/o familia de planta."),
        ('axes[1].set_title("Sin atípicos: estudio")', "fig-l2-pcoa-sin-atipicos",
         "PCoA recalculado sin las muestras atípicas: familia de planta (izq.) y estudio (der.)."),
        ('ax.set_title("¿Cuánta variación entre comunidades explica cada variable?")', "fig-l2-r2",
         "R² ajustado de la PERMANOVA por variable, con todas las muestras y sin atípicos."),
    ],
}


def label_notebook(nb_path, specs):
    """Inserta '#| label' y '#| fig-cap'/'#| tbl-cap' al inicio de cada celda indicada."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    changed = 0
    for fragment, label, caption in specs:
        matches = [c for c in code_cells if fragment in "".join(c["source"])]
        if len(matches) != 1:
            sys.exit(f"[{nb_path}] '{fragment}' aparece en {len(matches)} celdas (debe ser 1)")
        cell = matches[0]
        src = "".join(cell["source"])
        if src.lstrip().startswith("#| label:"):
            continue                                   # ya etiquetada: no duplicar
        cap_key = "tbl-cap" if label.startswith("tbl-") else "fig-cap"
        header = f'#| label: {label}\n#| {cap_key}: "{caption}"\n'
        cell["source"] = (header + src).splitlines(keepends=True)
        changed += 1
    # Guardar con el mismo formato que Jupyter (indentación 1, sin escapar acentos)
    Path(nb_path).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{nb_path}: {changed} celdas etiquetadas")


if __name__ == "__main__":
    for path, specs in LABELS.items():
        label_notebook(path, specs)
