from __future__ import annotations
from matplotlib.pyplot import text
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from estado import AgentState, Nivel
import json
import re
from typing import Any, Dict, List, Optional, TypedDict
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import numpy as np
import time

def fetch_url(url: str, timeout_s: int = 20) -> str:
    """Descarga el HTML de una URL (para scraping). Devuelve texto HTML."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
    }
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    html=r.text
    soup = BeautifulSoup(html, "html.parser")

    # Limpieza básica: quitar scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text(" ").split())
    text = text[:20000]  # recorte defensivo
    return text

class QueryPlan(BaseModel):
    queries: List[str] = Field(..., description="Consultas de búsqueda enfocadas en reviews confiables.")

def build_query_agent(state: AgentState) -> Dict[str, Any]:
    print('\nAgente: Generación de consultas de búsqueda para información detallada del programa académico')
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    progs=state.informacion_programas_nacionales or []
    revisar=0
    for idx, prg in enumerate(progs):
        if prg.iteraciones < 1 or len(prg.queries) == 0:
            revisar=idx
            break
    prg=progs[revisar]
    print(f"Generando consultas para el programa: {prg.Programa} de la institución {prg.Institucion}")
    system=f"""
Encontrar solo URLs que contengan información detallada y estructurada sobre el programa,
para poder extraer los siguientes datos:
- Descripción del programa
- Perfil del egresado
- Plan de estudios o malla curricular

Criterios de búsqueda:
- Prioriza páginas dentro del dominio oficial de la institución que ofrece el programa
- Incluye páginas institucionales del programa, facultad o escuela
- Evita resultados genéricos, noticias, rankings o blogs externos
- Prefiere páginas HTML (no PDFs, salvo que sean planes de estudio oficiales)
"""
    prompt=f"""Eres un agente de búsqueda especializado en encontrar información detallada sobre programas académicos en Colombia. 
    Tu tarea es generar consultas de búsqueda efectivas para encontrar páginas web oficiales que contengan información 
    relevante sobre el programa "{prg}" Considera que ya se pueden haber realizado búsquedas previas y debes enfocarte en encontrar información 
    más específica y detallada e información que pueda estar faltando. El énfasis es lograr completar la información necesaria desde 
    URL oficiales del programa o de la universidad correspondiente. 
    Tu objetivo es construir 4 queries que se van a usar para buscar en la web información detallada sobre el programa académico.
"""
    time.sleep(0.5)
    plan = llm.with_structured_output(QueryPlan).invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt)
    ])
    plan = QueryPlan.model_validate(plan.model_dump())
    #print('Salida del llm: ', plan)
    updated_prog = prg.model_copy(update={"queries": list(plan.queries), "iteraciones": prg.iteraciones + 1})
    updated_list=list(progs)
    updated_list[idx]=updated_prog
    #print(updated_list)
    return {'informacion_programas_nacionales': updated_list}

def decide_iterate(state: AgentState) -> str:
    # Determina si alguno de los programas nacionales tiene una iteración 0 o no tiene queries por revisar. En este caso, coloca el target_index al programa que debe completarse. Si todos los programas tienen iteración mayor a 0 y tienen queries, entonces se decide terminar.
    progs=state.informacion_programas_nacionales or []
    for idx, prg in enumerate(progs):
        if prg.iteraciones < 1 or len(prg.queries) == 0:
            return "iterate"
    return "finish"