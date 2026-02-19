from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from estado import AgentState, Nivel
import json

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def nodo_analizar_num_programas_instituciones(
    state: AgentState
) -> Dict[str, Any]:
    print('\nAgente: análisis número de programas e instituciones en el tiempo')
    
    registros = state.snies["num_programas_instituciones_tiempo"]
    
    def parse_periodo(p: str):
        # "2001-1" -> (2001, 1)
        partes = p.split("-")
        return int(partes[0]), int(partes[1])

    registros_ordenados = sorted(
        registros,
        key=lambda r: parse_periodo(str(r["PERIODO"]))
    )

    datos_json_str = json.dumps(registros_ordenados, ensure_ascii=False, indent=2)

    sistema = SystemMessage(
        content=(
            "Eres un analista de datos educativos. "
            "Tienes información histórica por periodo (año-semestre) "
            "sobre número de instituciones y programas, diferenciando sector Oficial y Privado. "
            "Responde en español, de forma clara y sintética. Sobre todo sobre datos existentes, "
            "no inventes nada que no tengas la información"
        )
    )

    usuario = HumanMessage(
        content=(
            "Te doy datos agregados por periodo y sector. "
            "Cada registro tiene los campos: PERIODO (ej. 2001-1 o 2001-2), "
            "SECTOR (Oficial o Privado), NUM_INSTITUCIONES y NUM_PROGRAMAS.\n\n"
            "Datos:\n"
            f"{datos_json_str}\n\n"
            "A partir de estos datos, por favor:\n"
            "1. Indica si a lo largo del tiempo predominan las instituciones oficiales o privadas.\n"
            "2. Indica si a lo largo del tiempo predominan los programas oficiales o privados.\n"
            "3. Comenta brevemente si ves alguna tendencia clara (por ejemplo, crecimiento en un sector, "
            "equilibrio, cambios bruscos en algún periodo, etc.).\n"
            "Ten en cuenta que 2001-1 es el primer semestre de 2001 y 2001-2 el segundo semestre. "
            "Puedes hablar de 'primer semestre' y 'segundo semestre' si es útil."
        )
    )
    if(len(state.analisis_num_programas_instituciones_tiempo)>10):
        respuesta=state.analisis_num_programas_instituciones_tiempo
        print("Ya se tenía un análisis realizado")
    else:
        print('Se debe correr el análisis en el LLM')
        respuesta = llm.invoke([sistema, usuario]).content

    return {
        "analisis_num_programas_instituciones_tiempo": respuesta
    }

def nodo_analizar_matriculas_vs_estudiantes(
    state: AgentState
) -> Dict[str, Any]:
    print('\nAgente: Análisis de la dispersión de matrículas respecto a los estudiantes')
    registros = state.snies["dispersión_matricula_vs_estudiantes"]["programas"]
    
    # Los pasamos a JSON “bonito” para que el LLM lo lea bien
    datos_json_str = json.dumps(registros, ensure_ascii=False, indent=2)

    sistema = SystemMessage(
        content=(
            "Eres un analista de datos educativos. "
            "no inventes nada que no tengas la información"
            "Se requiere que analices los datos de la manera más cercana a un experto en analítica de datos y estadísitica"
        )
    )

    usuario = HumanMessage(
        content=(
            "Tienes información que resume una duración de 4 años, para diferentes instituciones educativas. "
            "Para cada programa tienes metadatos como el nombre de la IES y el programa, el departamento donde está ubicado el programa"
            "El sector que indica si es un programa oficial o privado. "
            "Respecto a cada programa, tienes el valor promedio de la matricula en pesos colombianos, así como el número promedio "
            "de estudiantes matriculados. " 
            "Datos:\n"
            f"{datos_json_str}\n\n"
            "A partir de estos datos, por favor:\n"
            "1. Encuentra si hay una tendencia entre las universidades privadas y oficiales respecto al precio\n"
            "2. Encuentra si hay una tendencia entre las universidades privadas y oficiales respecto a la cantidad de estudiantes\n"
            "3. Analiza si puede existir una correlación entre precio y estudiantes, lo que muestre si el precio de un programa debería ser menor para tener más estudiantes. \n"
            "4. Haz un análisis de oportunidad para este programa en una universidad privada, de acuerdo con las tendencias observadas."
            
        )
    )
    if(len(state.analisis_dispersion_matricula_vs_estudiantes)>10):
        respuesta=state.analisis_dispersion_matricula_vs_estudiantes
        print("Se leyeron los datos de una corrida previa")
    else: 
        respuesta = llm.invoke([sistema, usuario]).content
        print('Se cargó la información consultando al LLM')
    return {
        "analisis_dispersion_matricula_vs_estudiantes": respuesta
    }

def nodo_analizar_matriculas_vs_tiempo(
    state: AgentState
) -> Dict[str, Any]:
    print('\nAgente: análisis del valor de la matrícula en el tiempo para los programas')
    registros=state.snies["valor_matricula_tiempo"]
    datos_json_str = json.dumps(registros, ensure_ascii=False, indent=2)

    sistema = SystemMessage(
        content=(
            "Eres un analista de datos educativos. "
            "no inventes nada que no tengas la información"
            "Se requiere que analices los datos de la manera más cercana a un experto en analítica de datos y estadísitica"
        )
    )

    usuario = HumanMessage(
        content=(
            "Tienes información que muestra el valor de la matrícula para diferentes instituciones educativas en diferentes períodos de tiempo"
            "Para cada registro cuentas con un diccionario que primero tiene el nombre de la ies y el origrama: nombre_ies_programa"
            "Tambien cuentas con el sector que define si es oficial o privada. "
            "Y los datos están en una lista llamada serie, que para cada período tiene el valor de la matrícula."
            "Datos:\n"
            f"{datos_json_str}\n\n"
            "A partir de estos datos, por favor:\n"
            "1. Encuentra si hay una tendencia en los valores de las matrículas respecto al tiempo. "
            "2. Encuentra si hay una tendencia entre las universidades privadas y oficiales respecto a la evolución de la matrícula\n"
            "3. Haz un análisis de oportunidad para este programa en una universidad privada, de acuerdo con las tendencias observadas."
            
        )
    )
    if(len(state.analisis_valor_matricula_tiempo)>10):
        print('Los datos ya se encontraban analizados. Se lee el análisis previo')
        respuesta=state.analisis_valor_matricula_tiempo
    else:
        respuesta = llm.invoke([sistema, usuario]).content
        print('Se cargó el análisis desde el LLM')

    return {
        "analisis_valor_matricula_tiempo": respuesta
    }


def nodo_analizar_programas_por_departamento_municipio(
    state: AgentState) -> Dict[str, Any]:
    print('\nAgente: análisis de número de programas por departamento y municipio')
    registros = state.snies["programas_por_departamento_municipio"]
    # Los pasamos a JSON “bonito” para que el LLM lo lea bien
    datos_json_str = json.dumps(registros, ensure_ascii=False, indent=2)

    sistema = SystemMessage(
        content=(
            "Eres un analista de datos educativos. "
            "no inventes nada que no tengas la información"
            "Se requiere que analices los datos de la manera más cercana a un experto en analítica de datos y estadísitica"
        )
    )

    usuario = HumanMessage(
        content=(
            "Tienes un conjunto de datos que muestra la cantidad de programas académicos en cada municipio y departamento de Colombia. "
            "Por favor realiza un análisis que permita entender cuáles departamentos tienen la mayor cantidad de programas"
            "y la distribución de programas en las ciudades del pais. Debes analizar si existe alguna correlación entre la cantidad"
            "de programas y la vocación económica de los departamentos y ciudades. "
            "Datos:\n"
            f"{datos_json_str}\n\n"
            "A partir de estos datos, por favor:\n"
            "1. Encuentra una relación entre la cantidad de programas y las vocaciones de las regiones."
            "2. Considera que el programa lo abriríamos en Medellín. Determina si la apertura del mismo tendría una buena oportunidad de acuerdo con la cantidad de programas en la región."
            "3. Haz un análisis de oportunidad para este programa en una universidad privada, de acuerdo con las tendencias observadas."
            
        )
    )
    if(len(state.analisis_programas_municipios)>10):
        print('Ya se cuenta con una respuesta previa. No se hace consulta al LLM')
        respuesta=state.analisis_programas_municipios
    else:
        respuesta = llm.invoke([sistema, usuario]).content
        print('Se consulta al LLM')

    return {
        "analisis_programas_municipios": respuesta
    }

def nodo_analizar_num_estudiantes_tiempo(
    state: AgentState) -> Dict[str, Any]:
    print('\nAgente: análisis de número de estudiantes en el tiempo en los programas')
    datos_plot = state.snies["num_estudiantes_tiempo"]
    # Los pasamos a JSON “bonito” para que el LLM lo lea bien
    datos_json_str = json.dumps(datos_plot, ensure_ascii=False, indent=2)

    sistema = SystemMessage(
        content=(
            "Eres un analista de datos educativos. "
            "no inventes nada que no tengas la información"
            "Se requiere que analices los datos de la manera más cercana a un experto en analítica de datos y estadísitica"
        )
    )

    usuario = HumanMessage(
        content=(
            "Tienes un conjunto de datos que dividen la información en tres grupos: todos los sectores, Universidades oficiales y universidades privadas. "
            "En cada uno de estos grupos se tiene un campo de periodos que indica el semestre y el año. 2001-1 indica el primer semestre del 2001"
            "En otro campo aparecen los diferentes procesos de los estudiantes que representan: los estudiantes ADMITIDOS (que se aceptan al programa)"
            "Los GRADUADOS, que terminaron el programa, los INSCRITOS que manifestaron su interés y se inscribieron. "
            "Finalmemente los MATRICULADOS, que efectivamente cumplieron el proceso de matrículas, así como los NUEVOS que son los que se aceptaron y matricularon como nuevos en cada período"
            "Datos:\n"
            f"{datos_json_str}\n\n"
            "A partir de estos datos, por favor:\n"
            "1. Encuentra relaciones en la variación de los estudiantes en todos los sectores, los oficiales y los privados. Determina si existen variaciones importantes entre unos y otros."
            "2. Analiza para todos estos tipos de sectores, si existe una diferencia entre los estudiantes inscritos y los que realmente se matricularon"
            "Existe un criterio muy importante y es determinar la capacidad de absorción del proceso de admisiones. Si se inscriben pocos, hay poco interes. "
            "Pero si se inscriben muchos pero se matriculan pocos, es porque algo más impide que completen su inscripción"
            "3. Haz un análisis de oportunidad para este programa en una universidad privada, de acuerdo con las tendencias observadas."
            
        )
    )
    if(len(state.analisis_numero_de_estudiantes)>10):
        print('Ya los datos se habían calculado previamente')
        respuesta=state.analisis_numero_de_estudiantes
    else:
        respuesta = llm.invoke([sistema, usuario]).content
        print('El análisis se procesa desde el LLM')

    return {
        "analisis_numero_de_estudiantes": respuesta
    }

