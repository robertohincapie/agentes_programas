from typing import Any, List, Dict
import unicodedata
import json
import os
import seaborn as sns
from estado import AgentState, Nivel, programa_nacional
import markdown
from datetime import datetime

def nodo_creador_reporte(state: AgentState) -> Dict[str, Any]:
    print('\nAgente creador de reporte tipo web')
    nombre = state.nombre
    nivel = state.nivel
    descripcion = state.descripcion
    codigos = state.codigos
    fecha = datetime.now()
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio","julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]

    fecha = f"{fecha.day} de {meses[fecha.month - 1]} de {fecha.year}"
    num_programas = len(state.informacion_programas_nacionales) if state.informacion_programas_nacionales else 0
    cad=f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reporte: {nombre}</title>

  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 0;
      background: #f5f7fa;
      color: #1f2937;
      line-height: 1.6;
    }}

    .container {{
      max-width: 900px;
      margin: 40px auto;
      padding: 40px;
      background: white;
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    }}

    h1 {{
      margin-top: 0;
      font-size: 32px;
      border-bottom: 3px solid #e5e7eb;
      padding-bottom: 12px;
    }}

    h2 {{
      margin-top: 50px;
      font-size: 22px;
      border-left: 5px solid #3b82f6;
      padding-left: 12px;
    }}

    ul {{
      padding-left: 22px;
    }}

    li {{
      margin-bottom: 8px;
    }}

    .section {{
      margin-top: 40px;
    }}

    .section img {{
      width: 100%;
      border-radius: 10px;
      margin: 18px 0;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}

    .analysis {{
      background: #f9fafb;
      padding: 18px;
      border-radius: 10px;
      border: 1px solid #e5e7eb;
    }}

    .footer {{
      margin-top: 60px;
      font-size: 13px;
      color: #6b7280;
      text-align: center;
    }}
  </style>
</head>
<body>

  <div class="container">

    <!-- TÍTULO -->
    <h1>{nombre}</h1>
    <!-- DESCRIPCIÓN -->
    <p>{descripcion}</p>

    <!-- INFORMACIÓN BÁSICA -->
    <ul>
      <li><strong>Fecha de generación:</strong> {fecha}</li>
      <li><strong>Total de programas analizados:</strong> {num_programas}</li>
    </ul>
    """
    info=[
        ("Análisis de número de programas e instituciones en el tiempo", state.analisis_num_programas_instituciones_tiempo,['num_programas_instituciones_tiempo.png']),
        ("Análisis de matrícula vs estudiantes", state.analisis_dispersion_matricula_vs_estudiantes, ['dispersión_estudiantes_matricula.png']),
        ("Análisis de valor matrícula en el tiempo", state.analisis_valor_matricula_tiempo, ['valor_matriculas_por_periodo.png']),
        ("Análisis de programas por municipios", state.analisis_programas_municipios,['programas_por_departamento_municipio.png']),
        ("Análisis del número de estudiantes en el tiempo", state.analisis_numero_de_estudiantes, ['num_estudiantes_tiempo_Universidades_Oficiales.png', 'num_estudiantes_tiempo_Universidades_Privadas.png', 'num_estudiantes_tiempo_Todos_los_sectores.png'])
    ]
    for titulo, analisis, imagenes in info:
        imgs="".join([f'<img src="{img}" alt="{titulo}">' for img in imagenes])
        analisis=markdown.markdown(analisis, extensions=["extra"])   # incluye soporte de tablas)
        cad+=f"""
<div class="section">
      <h2>{titulo}</h2>

      {imgs}

      <div class="analysis">
        <p>
        {analisis}
        </p>
      </div>
    </div>
"""
    cad+=f"""
    <!-- PIE -->
    <div class="footer">
      Reporte generado automáticamente · Sistema de análisis académico
    </div>

  </div>

</body>
</html>
"""
    with open("./salida/reporte.html", "w", encoding="utf-8") as f:
        f.write(cad)
    return {}
