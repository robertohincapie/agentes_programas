import os
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from enum import Enum

class Nivel(str, Enum):
    pregrado = "pregrado"
    tecnica = "tecnica"
    tecnologia = "tecnologia"
    especializacion = "especializacion"
    maestria = "maestria"
    doctorado = "doctorado"

class programa_nacional(BaseModel):
    Programa: str
    Institucion: str
    Municipio: str
    URL: Optional[str]
    URL_programa: Optional[str]
    Descripcion: Optional[str]
    Perfil: Optional[str]
    Plan_de_estudios: Optional[List[str]] = Field(default_factory=list)
    iteraciones: int = 0
    queries: Optional[List[str]] = Field(default_factory=list, description="Consultas de búsqueda enfocadas en reviews confiables.")

class AgentState(BaseModel):
    nombre: str
    nivel: Nivel
    descripcion: str
    requerido: List[str] #Palabras claves para buscar el programa en el listado de programas existentes
    snies: Optional[Dict[str, Any]] = None
    analisis_num_programas_instituciones_tiempo: Optional[str] = ""
    analisis_dispersion_matricula_vs_estudiantes: Optional[str] = ""
    analisis_valor_matricula_tiempo: Optional[str] = ""
    analisis_programas_municipios: Optional[str] = ""
    analisis_numero_de_estudiantes: Optional[str] = ""
    informacion_programas_nacionales: Optional[List[programa_nacional]] = None
    target_index: Optional[int] = None #Campo que determina el programa que se está analizando en el nodo de búsqueda web
