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
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import numpy as np
import time
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple, Optional, TypedDict

from openai import OpenAI


def _ensure_https(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def _domain_from_url(url: str) -> str:
    u = _ensure_https(url)
    if not u:
        return ""
    return urlparse(u).netloc


def _state_get(state: Any, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _state_set(state: Any, key: str, value):
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


def openai_web_search_program_page(
    query: str,
    allowed_domains: List[str],
    max_sources: int = 8,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Hace búsqueda web con OpenAI y devuelve:
    - sources: [{"url","title"}]
    - best_url: str (elegida por el modelo)
    - extracted: {"descripcion","perfil","plan_de_estudios","plan_urls": [...]}
    Todo en JSON (dict).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system = (
        "Eres un asistente que busca SOLO en sitios oficiales permitidos. "
        "Debes seleccionar la URL oficial más relevante del programa académico "
        "para extraer luego (scraping). "
        "Luego, con base únicamente en esa(s) fuente(s), redacta una extracción breve y estructurada "
        "de: descripcion, perfil y plan de estudios. "
        "Si no encuentras plan de estudios en la misma página, incluye URLs alternativas del mismo dominio "
        "donde aparezca (pensum/malla/plan de estudios). "
        "NO inventes información: si no está en las fuentes, deja el campo como null."
    )

    # Le pedimos salida JSON estricta.
    user = f"""
Consulta: {query}

Devuelve SOLO JSON con este esquema:
{{
  "best_url": "https://...",
  "supporting_urls": ["https://...","https://..."],
  "extracted": {{
    "descripcion": "string|null",
    "perfil": "string|null",
    "plan_de_estudios": "string|null",
    "plan_urls": ["https://..."]
  }},
  "notes": "string"
}}

Reglas:
- best_url y supporting_urls deben ser del dominio permitido.
- extracted.* debe estar sustentado en las URLs.
- plan_urls debe incluir páginas del plan/pensum/malla si existen.
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=[{
            "type": "web_search",
            "filters": {"allowed_domains": allowed_domains}
        }],
        tool_choice="auto",
        include=["web_search_call.action.sources"],
    )

    # 1) Extraer sources (URLs) desde la llamada de web_search
    sources = []
    try:
        for out in resp.output:
            if getattr(out, "type", None) == "web_search_call":
                action = getattr(out, "action", None)
                if action and getattr(action, "sources", None):
                    for s in action.sources:
                        url = getattr(s, "url", None) or (s.get("url") if isinstance(s, dict) else None)
                        title = getattr(s, "title", None) or (s.get("title") if isinstance(s, dict) else None)
                        if url:
                            sources.append({"url": url, "title": title})
    except Exception:
        pass

    # 2) Parsear el JSON final del modelo
    text = (getattr(resp, "output_text", "") or "").strip()
    #print("DEBUG: Respuesta cruda del modelo:", text)
    data = {}
    try:
        data = json.loads(text)
    except Exception:
        # fallback: si no devolvió JSON perfecto, al menos devuelve las fuentes
        data = {
            "best_url": "",
            "supporting_urls": [s["url"] for s in sources[:max_sources]],
            "extracted": {"descripcion": None, "perfil": None, "plan_de_estudios": None, "plan_urls": []},
            "notes": "El modelo no devolvió JSON parseable; revisa resp.output_text.",
        }

    # 3) Enforce dominios + dedupe
    allowed = set([d.lower() for d in allowed_domains if d])
    def ok_domain(u: str) -> bool:
        try:
            host = urlparse(_ensure_https(u)).netloc.lower()
            return any(host == d or host.endswith("." + d) for d in allowed)
        except Exception:
            return False

    # Normaliza/filtra URLs
    best_url = data.get("best_url") or ""
    if best_url and not ok_domain(best_url):
        best_url = ""

    supporting = [u for u in (data.get("supporting_urls") or []) if u and ok_domain(u)]
    plan_urls = [u for u in ((data.get("extracted") or {}).get("plan_urls") or []) if u and ok_domain(u)]

    # Si el modelo no eligió best_url, intenta tomar la primera fuente
    if not best_url and sources:
        for s in sources:
            if ok_domain(s["url"]):
                best_url = s["url"]
                break

    # Dedupe
    def dedupe(lst):
        seen = set()
        out = []
        for u in lst:
            u = _ensure_https(u)
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        return out

    supporting = dedupe(supporting)[:max_sources]
    plan_urls = dedupe(plan_urls)[:max_sources]

    # Adjunta sources crudas también (útil para debug)
    return {
        "best_url": _ensure_https(best_url),
        "supporting_urls": supporting,
        "extracted": {
            "descripcion": (data.get("extracted") or {}).get("descripcion"),
            "perfil": (data.get("extracted") or {}).get("perfil"),
            "plan_de_estudios": (data.get("extracted") or {}).get("plan_de_estudios"),
            "plan_urls": plan_urls,
        },
        "notes": data.get("notes", ""),
        "sources": sources[:max_sources],
    }


def programas_nacionales_openai_node(state: Any) -> Dict[str, Any]:
    """
    Nodo para tu grafo:
    - Lee state.target_index y state.informacion_programas_nacionales
    - Busca en dominio oficial (del campo URL) la página del programa
    - Actualiza URL_programa + (opcional) Descripcion/Perfil/Plan_de_estudios y logs
    """
    idx = int(state.target_index)
    programas = state.informacion_programas_nacionales

    if not isinstance(programas, list) or not programas:
        return state
    if idx < 0 or idx >= len(programas):
        return {'target_index': -1}  # señal de que terminamos

    item = programas[idx]

    programa = item.Programa
    print('Buscando información del programa:', programa, 'de la institución:', item.Institucion)
    institucion = item.Institucion
    municipio = item.Municipio
    base_url = item.URL

    domain = _domain_from_url(base_url)
    if not domain:
        # fallback muy básico si no hay URL
        # (puedes poner un mapping por institución)
        domain = "unal.edu.co"

    # Query bien dirigida para páginas de programa
    query = (
        f'site:{domain} "{programa}" '
        f'("plan de estudios" OR "malla curricular" OR pensum OR perfil OR descripción OR descripcion) '
        f'{municipio}'
    )

    result = openai_web_search_program_page(
        query=query,
        allowed_domains=[domain],
        max_sources=8,
        model="gpt-4.1",
    )

    # Actualiza: URL para scraping (lo principal)
    item.URL_programa = result.get("best_url") or item.URL_programa

    # Opcional: si quieres que el agente también te deje “pre-extraído”
    extracted = result.get("extracted") or {}
    item.Descripcion = extracted.get("descripcion") or item.Descripcion
    item.Perfil = extracted.get("perfil") or item.Perfil
    # Plan_de_estudios en tu schema es lista: aquí guardo URLs del plan (mejor para scraping)
    plan_urls = extracted.get("plan_urls") or []
    if plan_urls:
        item.Plan_de_estudios = plan_urls

    # logs
    item.iteraciones = int(item.iteraciones or 0) + 1
    prev_q = item.queries or []
    prev_q.append({
        "provider": "openai_web_search",
        "query": query,
        "allowed_domain": domain,
        "best_url": result.get("best_url"),
        "supporting_urls": result.get("supporting_urls"),
        "notes": result.get("notes"),
    })
    item.queries = prev_q

    programas[idx] = item
    #state.informacion_programas_nacionales = programas
    ind=-1
    for i, prg in enumerate(programas): 
        if prg.iteraciones==0:
            ind = i
            break
    #print('Índice del siguiente programa a procesar a continuación:', ind, '\n')
    return { 'informacion_programas_nacionales': programas, 'target_index': ind }

def decide_iterate(state: Any) -> str:
    idx=int(state.target_index)
    #print('Función decide_iterate: target_index=', idx, '\n')
    if idx < 0:
        return "terminar"
    else:
        return "iterar"