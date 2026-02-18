from typing import Any, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import json
import os
import seaborn as sns
from estado import AgentState, Nivel, programa_nacional

def nodo_lector_snies(state: AgentState) -> Dict[str, Any]:
    print('\nAgente: análisis de información existente de SNIES')
    nombre = state.nombre
    nivel = state.nivel
    descripcion = state.descripcion
    requerido = state.requerido
    print('Revisando si ya hay información: ', len(state.informacion_programas_nacionales))

    if(len(state.informacion_programas_nacionales or []) > 0):
        print('Ya existe información de programas nacionales cargada. Se omite la consulta a SNIES.')
        return {}
    else:
        #Verificamos si la informción de este programa ya fue cargada o no. En ese caso no se consulta de nuevo
        resultado = lector_snies(state)   # llama la herramienta de captura de la información de snies. 
        print(resultado['snies'])
        return {
            "snies": resultado['snies'],
            "informacion_programas_nacionales": resultado['informacion_programas_nacionales']
        }

def lector_snies(state) -> dict:
    print('Lector de Snies')
    #Primero verificamos si existe un campo de informacion_programas_nacionales en el estado. 
    #Pero en el estado que tenemos guardado en el archivo de texto si exsite. Si ese campo existe, entonces no se hace la consulta
    #pero debe verificar que el nombre del programa y la información básica sean correctas. 

    def cargar_parquet_cache(url: str, local_path: str):
        if os.path.exists(local_path):
            return pd.read_parquet(local_path)
        df = pd.read_parquet(url)
        df.to_parquet(local_path, index=False)
        return df

    def normalizar_texto(cadena: str) -> str:
        cadena = cadena.lower()
        cadena = cadena.replace("ñ", "n").replace("Ñ", "n")
        cadena_normalizada = unicodedata.normalize("NFD", cadena)
        cadena_sin_tildes = "".join(
            c for c in cadena_normalizada
            if unicodedata.category(c) != "Mn"
        )
        return cadena_sin_tildes

    respuesta: dict = {
        "snies": {},          # aquí irán los datos numéricos de cada gráfica
        "informacion_programas_nacionales": [],        # Programas que se cargan desde el SNIES
    }
    nombre = state.nombre
    nivel = state.nivel
    descripcion = state.descripcion
    requerido = state.requerido

    programa = set(normalizar_texto(nombre).split())
    requerido = {normalizar_texto(ci) for ci in requerido}
    prohibido = { normalizar_texto(ci) for ci in ["pregrado","técnica","tecnología","especializacion","maestria","doctorado"] 
        if ci != nivel}
    n = len(programa)

    print("Proceso de carga de los archivos de SNIES")
    maestro = cargar_parquet_cache(
        "https://robertohincapie.com/data/snies/MAESTRO.parquet",
        "MAESTRO.parquet",
    )
    oferta = cargar_parquet_cache(
        "https://robertohincapie.com/data/snies/OFERTA.parquet",
        "OFERTA.parquet",
    )
    programas = cargar_parquet_cache(
        "https://robertohincapie.com/data/snies/PROGRAMAS.parquet",
        "PROGRAMAS.parquet",
    )
    ies = cargar_parquet_cache(
        "https://robertohincapie.com/data/snies/IES.parquet",
        "IES.parquet",
    )
    print("Archivos de SNIES cargados correctamente")

    programas["PROGRAMA_ACADEMICO_NORMALIZADO"] = programas[
        "PROGRAMA_ACADEMICO"
    ].apply(lambda x: normalizar_texto(str(x)))

    # Selección de programas equivalentes
    equivalentes = []
    def evaluar(prog2, requerido, prohibido):
        # Si hay palabra prohibida exacta → False
        if any(p in prog2 for p in prohibido):
            return False
        
        # Si alguna requerida está contenida en alguna palabra de prog2 → True
        if any(req in palabra for req in requerido for palabra in prog2):
            return True
        
        return False

    for prg in programas["PROGRAMA_ACADEMICO_NORMALIZADO"].unique():
        prg2 = str(prg).lower().split()
        #Vamos a considerar que en alguno de los términos del programa estén las palabras de la lista requerida y no estén las palabras prohibidas. Además, el programa debe tener al menos n-1 términos en común con el programa objetivo (para permitir pequeñas variaciones)
        if(evaluar(prg2, requerido, prohibido)):
            equivalentes.append(prg)
    print('Programas equivalentes encontrados: ',equivalentes)
    programas2 = programas[
        programas["PROGRAMA_ACADEMICO_NORMALIZADO"].isin(equivalentes)
    ]
    snies2 = list(programas2["CODIGO_SNIES"].unique())
    maestro2 = maestro[maestro["CODIGO_SNIES"].isin(snies2)]

    maestro3 = maestro2.merge(
        programas, left_on="CODIGO_SNIES", right_on="CODIGO_SNIES", how="left"
    )
    maestro4 = maestro3.merge(oferta, on=["CODIGO_SNIES", "PERIODO"], how="left")
    maestro5 = maestro4.merge(
        ies[
            [
                "CODIGO_INSTITUCION",
                "INSTITUCION",
                "NATURALEZA_JURIDICA",
                "SECTOR_IES",
                "CARACTER_IES",
                "PAGINA_WEB",
                "ACREDITACION_ALTA_CALIDAD",
            ]
        ],
        left_on="CODIGO_INSTITUCION_x",
        right_on="CODIGO_INSTITUCION",
        how="left",
        suffixes=("__x", "__y"),
    )
    #maestro5.to_excel('borrar.xlsx', index=False)
    #os.makedirs("./figuras_snies", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Número de instituciones y programas en el tiempo
    # ------------------------------------------------------------------
    
    progs = (
        maestro5.groupby(by=["PERIODO", "SECTOR_IES__x", "DEPARTAMENTO_PROGRAMA"])
        .agg({"CODIGO_INSTITUCION_x": "nunique", "CODIGO_SNIES": "nunique"})
        .reset_index()
    )
    progs.columns = [
        "PERIODO",
        "SECTOR",
        "DEPARTAMENTO",
        "NUM_INSTITUCIONES",
        "NUM_PROGRAMAS",
    ]

    # Versión agregada por periodo y sector (para JSON)
    progs_periodo_sector = (
        progs.groupby(by=["PERIODO", "SECTOR"])
        .agg({"NUM_INSTITUCIONES": "sum", "NUM_PROGRAMAS": "sum"})
        .reset_index()
    )

    # Pivot para la figura
    progs_pivot = pd.pivot_table(
        data=progs_periodo_sector,
        index="PERIODO",
        columns="SECTOR",
        values=["NUM_INSTITUCIONES", "NUM_PROGRAMAS"],
        aggfunc="sum",
    )
    progs_pivot.columns = [
        "Num. Instituciones oficiales",
        "Num Instituciones privadas",
        "Num. Programas oficiales",
        "Num Programas privados",
    ]

    colores = ["red", "blue", "red", "blue"]
    estilos = [(1, 0), (1, 0), (2, 2), (2, 2)]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=progs_pivot, palette=colores, dashes=estilos)
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        f"./salida/num_programas_instituciones_tiempo.png", dpi=300
    )

    # JSON para el agente
    respuesta["snies"]["num_programas_instituciones_tiempo"] = progs_periodo_sector.to_dict(orient="records")

    # ------------------------------------------------------------------
    # 2. Dispersión matrícula 2024 vs promedio matriculados 2021-2023
    # ------------------------------------------------------------------
    maestro4["PROXY_PER"] = maestro4["PROXY_PER"].astype(int)
    df = maestro4[
        (maestro4["PROXY_PER"] >= 20211) & (maestro4["PROXY_PER"] <= 20242)
    ].copy()
    df.loc[:, "Nombre_ies"] = df["INSTITUCION"] + " - " + df["PROGRAMA_ACADEMICO"]
    df = df[df["PROCESO"] == "MATRICULADOS"].copy()
    df["CANTIDAD"] = df["CANTIDAD"].astype(int)

    df = df[
        [
            "MATRICULA",
            "CANTIDAD",
            "Nombre_ies",
            "PERIODO",
            "DEPARTAMENTO_PROGRAMA",
            "SECTOR_IES",
        ]
    ]
    df = df.dropna()
    df = df[df["MATRICULA"] != "null"].copy()
    df["MATRICULA"] = df["MATRICULA"].astype(float)

    df2 = (
        df.groupby(by="Nombre_ies")
        .agg(
            {
                "MATRICULA": "last",
                "CANTIDAD": "mean",
                "SECTOR_IES": "first",
                "DEPARTAMENTO_PROGRAMA": "first",
            }
        )
        .reset_index()
    )

    # JSON básico con la nube de puntos
    est_mat_ies_prog = {
        "programas": [
            {
                "nombre_ies_programa": row["Nombre_ies"],
                "departamento": row["DEPARTAMENTO_PROGRAMA"],
                "sector": row["SECTOR_IES"],
                "matricula_2024": float(row["MATRICULA"]),
                "num_estudiantes_promedio_2021_2023": float(row["CANTIDAD"]),
            }
            for _, row in df2.iterrows()
        ]
    }
    # Correlación global entre matrícula y número de estudiantes
    if len(df2) > 1:
        est_mat_ies_prog["correlacion_matricula_estudiantes"] = float(
            df2["MATRICULA"].corr(df2["CANTIDAD"])
        )

    respuesta["snies"]["dispersión_matricula_vs_estudiantes"] = est_mat_ies_prog

    # Figura
    plt.figure(figsize=(12, 6))
    df2["MATRICULA"] = df2["MATRICULA"].astype(float) / 1e6
    sns.scatterplot(
        data=df2,
        x="CANTIDAD",
        y="MATRICULA",
        hue="SECTOR_IES",
        palette={"Privado": "blue", "Oficial": "red"},
    )
    plt.xlabel("Número promedio de estudiantes matriculados (2021-2023)")
    plt.ylabel("Valor de la matrícula (2024) (Millones de COP)")
    plt.title("Relación entre número de estudiantes y valor de la matrícula")
    texts = []
    for _, row in df2.iterrows():
        tipo = " (" + row["DEPARTAMENTO_PROGRAMA"] + ")"
        color = "red" if row["SECTOR_IES"] == "Oficial" else "blue"
        t = plt.text(
            row["CANTIDAD"] + 0.03,
            row["MATRICULA"] + 0.03,
            "\n".join(
                [
                    row["Nombre_ies"].split(" - ")[0] + tipo,
                    row["Nombre_ies"].split(" - ")[1],
                ]
            ),
            fontsize=8,
            ha="center",
            va="bottom",
            color=color,
        )
        texts.append(t)
    #adjust_text(texts, arrowprops=dict(arrowstyle="->", color="black"))
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"./salida/dispersión_estudiantes_matricula.png", dpi=300)

    # ------------------------------------------------------------------
    # 3. Valor de matrícula en el tiempo por institución
    # ------------------------------------------------------------------
    valor = pd.pivot_table(
        df,
        index="Nombre_ies",
        columns="PERIODO",
        values="MATRICULA",
        aggfunc="mean",
        fill_value=0,
    ) / 1e6

    sectores = df[["Nombre_ies", "SECTOR_IES"]].drop_duplicates()
    valor = valor.merge(sectores, on="Nombre_ies", how="left")

    valor_long = valor.melt(
        id_vars=["Nombre_ies", "SECTOR_IES"],
        var_name="PERIODO",
        value_name="MATRICULA_M",
    )
    valor_long.columns = ["Nombre", "Sector", "Período", "Valor_Matricula"]

    # JSON con series por institución
    series_por_ies = []
    for nombre, grupo in valor_long.groupby("Nombre"):
        grupo_orden = grupo.sort_values("Período")
        serie = {
            "nombre_ies_programa": nombre,
            "sector": grupo_orden["Sector"].iloc[0],
            "serie": [
                {"periodo": per, "valor_matricula_millones": float(v)}
                for per, v in zip(
                    grupo_orden["Período"], grupo_orden["Valor_Matricula"]
                )
            ],
        }
        series_por_ies.append(serie)

    respuesta["snies"]["valor_matricula_tiempo"] = series_por_ies

    # Figura con etiquetas a la derecha
    plt.figure(figsize=(16, 6))
    texts_pos = {}
    colores = {}
    for nombre in valor_long["Nombre"].unique():
        df_temp = valor_long[valor_long["Nombre"] == nombre].sort_values(
            "Período"
        )
        x = range(len(df_temp))
        if df_temp["Sector"].unique()[0] == "Privado":
            line, = plt.plot(x, df_temp["Valor_Matricula"], "-")
        else:
            line, = plt.plot(x, df_temp["Valor_Matricula"], "--")
        color = line.get_color()
        texts_pos[nombre] = float(df_temp["Valor_Matricula"].iloc[-1])
        colores[nombre] = color

    ax = plt.gca()
    texts_pos = dict(sorted(texts_pos.items(), key=lambda x: x[1]))
    dL = 5
    base_y = 0.01
    dy = 0.6

    # usamos coordenadas de datos para y y eje extendido para x
    for i, (nombre, value) in enumerate(texts_pos.items()):
        x_data = len(x) - 1
        x_label = x_data + 1
        y_label = base_y + i * dy
        ax.text(x_label, y_label, nombre, color=colores[nombre])
        plt.plot([x_data, x_label], [value, y_label], ":", color=colores[nombre])

    plt.xticks(
        range(len(valor_long["Período"].unique())),
        sorted(valor_long["Período"].unique()),
        rotation=90,
    )
    plt.ylim(0, 15)
    plt.xlim(0, len(x) + dL)
    plt.xlabel("Período")
    plt.ylabel("Valor de matrícula en millones de COP")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"./salida/valor_matriculas_por_periodo.png", dpi=300)

    # ------------------------------------------------------------------
    # 4. Número de programas por departamento y municipio
    # ------------------------------------------------------------------
    df_geo = maestro4[
        (maestro4["PROXY_PER"] >= 20211) & (maestro4["PROXY_PER"] <= 20242)
    ].copy()
    df_geo.loc[:, "Nombre_ies"] = (
        df_geo["INSTITUCION"] + " - " + df_geo["PROGRAMA_ACADEMICO"]
    )
    df_geo = df_geo[df_geo["PROCESO"] == "MATRICULADOS"].copy()
    df_geo["CANTIDAD"] = df_geo["CANTIDAD"].astype(int)

    df_geo2 = (
        df_geo.groupby(["DEPARTAMENTO_PROGRAMA", "MUNICIPIO_PROGRAMA"])
        .agg({"CODIGO_SNIES": "nunique"})
        .sort_values(by="CODIGO_SNIES", ascending=False)
        .reset_index()
    )
    df_geo2.columns = ["Departamento", "Municipio", "Numero_programas"]
    df_geo2["Ubicacion"] = df_geo2["Departamento"] + " - " + df_geo2["Municipio"]

    # Figura
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="Numero_programas",
        y="Ubicacion",
        data=df_geo2,
        hue="Departamento",
        legend=False,
    )
    plt.tight_layout()
    plt.savefig(
        f"./salida/programas_por_departamento_municipio.png", dpi=300
    )

    # JSON
    respuesta["snies"]["programas_por_departamento_municipio"] = df_geo2[
            ["Departamento", "Municipio", "Numero_programas"]
        ].to_dict(orient="records")

    # ------------------------------------------------------------------
    # 5. Número de estudiantes en el tiempo (todos / oficial / privado)
    # ------------------------------------------------------------------
    maestro4 = maestro4[maestro4["CANTIDAD"] != "null"]
    maestro4["CANTIDAD"] = maestro4["CANTIDAD"].astype(float)

    resumen_num_est = {}

    for df_est, exp in [
        (maestro4, "Todos los sectores"),
        (maestro4[maestro4["SECTOR_IES"] == "Oficial"], "Universidades Oficiales"),
        (maestro4[maestro4["SECTOR_IES"] == "Privado"], "Universidades Privadas"),
    ]:
        num = pd.pivot_table(
            df_est,
            index="PERIODO",
            columns="PROCESO",
            values="CANTIDAD",
            fill_value=0,
            aggfunc="sum",
        )
        num_est = {
            "periodos": list(num.index),
            "procesos": list(num.columns),
            "valores": [
                {proc: float(num.loc[per, proc]) for proc in num.columns}
                for per in num.index
            ],
        }
        resumen_num_est[exp] = num_est

        plt.figure(figsize=(12, 6))
        sns.lineplot(num)
        plt.xlabel("Período académico")
        plt.ylabel("Número de estudiantes")
        plt.xticks(rotation=90)
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.tight_layout()
        plt.grid(True)
        plt.title("Número de estudiantes en el tiempo en " + exp)
        plt.savefig(
            f"./salida/num_estudiantes_tiempo_"
            + exp.replace(" ", "_")
            + ".png",
            dpi=300,
        )

    respuesta["snies"]["num_estudiantes_tiempo"] = resumen_num_est

    # ------------------------------------------------------------------
    # 6. Prompt con listado de programas (para otro agente)
    # ------------------------------------------------------------------
    programas = []
    i = 1
    for ies_name, prg, mpio, url in (
        maestro5[["INSTITUCION__y", "PROGRAMA_ACADEMICO", "MUNICIPIO_PROGRAMA","PAGINA_WEB"]]
        .drop_duplicates()
        .values
    ):
        programas.append(
            programa_nacional(
                Programa=prg,
                Institución=ies_name,
                Municipio=mpio,
                URL=url,
                URL_programa="",
                Descripcion="",
                Perfil="",
                Plan_de_estudios=[],
                iteraciones=0   #Significa que apenas estamos creando. Falta buscar la información detallada del programa y cargarla en este campo para que el agente de búsqueda de información pueda usarla como referencia para encontrar la información correcta.
            )
        )

    respuesta["informacion_programas_nacionales"] = programas

    return respuesta
