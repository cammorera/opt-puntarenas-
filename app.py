"""
Florida Bebidas — Distribución Nacional CVRP
Provincia: Puntarenas  ·  Curso II-1122 · Clase 13
Streamlit App con Clarke-Wright Savings Algorithm
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import permutations
import math

# ─────────────────────────────────────────────────────────────
# DATOS PUNTARENAS
# ─────────────────────────────────────────────────────────────
CANTONES = {
    0:  "CD Puntarenas",
    1:  "Puntarenas",
    2:  "Esparza",
    3:  "Buenos Aires",
    4:  "Montes de Oro",
    5:  "Osa",
    6:  "Quepos",
    7:  "Golfito",
    8:  "Coto Brus",
    9:  "Parrita",
    10: "Corredores",
    11: "Garabito",
    12: "Monteverde",
    13: "Puerto Jiménez",
}

DEMAND = {
    1: 107, 2: 27,  3: 37,  4: 12,
    5: 28,  6: 24,  7: 33,  8: 35,
    9: 16,  10: 39, 11: 20, 12: 4, 13: 8,
}

DEMAND_DETAIL = {
    1:  (53,27,27), 2:  (13,7,7),  3:  (19,9,9),  4:  (6,3,3),
    5:  (14,7,7),   6:  (12,6,6),  7:  (17,8,8),  8:  (17,9,9),
    9:  (8,4,4),    10: (19,10,10),11: (10,5,5),   12: (2,1,1),
    13: (4,2,2),
}

DIST_MATRIX_RAW = [
    #  0    1    2    3    4    5    6    7    8    9   10   11   12   13
    [  0,   0,  25, 244,  27, 243, 124, 307, 307,  99, 332,  60,  47, 303],  # 0
    [  0,   0,  25, 244,  27, 243, 124, 307, 307,  99, 332,  60,  47, 303],  # 1
    [ 25,  25,   0, 224,  19, 226, 109, 290, 287,  85, 314,  55,  49, 288],  # 2
    [244, 244, 224,   0, 240,  41, 124,  80,  63, 150,  95, 196, 267,  92],  # 3
    [ 27,  27,  19, 240,   0, 244, 127, 308, 303, 103, 331,  73,  30, 305],  # 4
    [243, 243, 226,  41, 244,   0, 119,  64,  79, 145,  90, 189, 272,  64],  # 5
    [124, 124, 109, 124, 127, 119,   0, 183, 186,  26, 208,  72, 156, 179],  # 6
    [307, 307, 290,  80, 308,  64, 183,   0,  54, 208,  31, 253, 336,  25],  # 7
    [307, 307, 287,  63, 303,  79, 186,  54,   0, 212,  46, 259, 330,  78],  # 8
    [ 99,  99,  85, 150, 103, 145,  26, 208, 212,   0, 234,  47, 133, 204],  # 9
    [332, 332, 314,  95, 331,  90, 208,  31,  46, 234,   0, 279, 359,  52],  # 10
    [ 60,  60,  55, 196,  73, 189,  72, 253, 259,  47, 279,   0, 102, 246],  # 11
    [ 47,  47,  49, 267,  30, 272, 156, 336, 330, 133, 359, 102,   0, 335],  # 12
    [303, 303, 288,  92, 305,  64, 179,  25,  78, 204,  52, 246, 335,   0],  # 13
]

DIST = np.array(DIST_MATRIX_RAW)

# Coordenadas aproximadas para visualización (lat, lon)
COORDS = {
    0:  (9.975, -84.831),   # CD Puntarenas (Río Segundo Alajuela aprox)
    1:  (9.978, -84.836),   # Puntarenas ciudad
    2:  (9.990, -84.664),   # Esparza
    3:  (9.164, -83.327),   # Buenos Aires
    4:  (10.080,-84.623),   # Montes de Oro
    5:  (9.143, -83.727),   # Osa
    6:  (9.432, -84.162),   # Quepos
    7:  (8.650, -83.180),   # Golfito
    8:  (8.810, -82.960),   # Coto Brus
    9:  (9.516, -84.293),   # Parrita
    10: (8.618, -83.052),   # Corredores
    11: (9.671, -84.628),   # Garabito
    12: (10.348,-84.821),   # Monteverde
    13: (8.533, -83.299),   # Puerto Jiménez
}

CAPACITY = 24

COLORS = [
    "#E63946","#2A9D8F","#E9C46A","#264653","#F4A261",
    "#A8DADC","#457B9D","#6A0572","#F77F00","#4CAF50",
    "#FF6B6B","#C77DFF","#06D6A0","#FFB703","#023E8A",
    "#90BE6D","#F9C74F",
]


# ─────────────────────────────────────────────────────────────
# ALGORITMO CLARKE-WRIGHT
# ─────────────────────────────────────────────────────────────
def clarke_wright(capacity=24):
    """Clarke-Wright Savings Algorithm para CVRP."""
    clients = list(DEMAND.keys())  # 1..13
    depot = 0

    # Calcular savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in clients:
        for j in clients:
            if i < j:
                s = DIST[depot][i] + DIST[depot][j] - DIST[i][j]
                savings.append((s, i, j))
    savings.sort(reverse=True)

    # Rutas iniciales: cada cliente en su propia ruta
    routes = {c: [depot, c, depot] for c in clients}
    route_load = {c: DEMAND[c] for c in clients}
    client_route = {c: c for c in clients}  # cliente -> ruta_id

    def route_end(route_id):
        """Último cliente antes del depot de retorno."""
        r = routes[route_id]
        return r[-2]

    def route_start(route_id):
        """Primer cliente después del depot."""
        r = routes[route_id]
        return r[1]

    for s, i, j in savings:
        ri = client_route.get(i)
        rj = client_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        if route_load[ri] + route_load[rj] > capacity:
            continue
        # Merge: i debe ser el final de su ruta, j el inicio de la suya
        if route_end(ri) == i and route_start(rj) == j:
            new_route = routes[ri][:-1] + routes[rj][1:]
            new_id = ri
            routes[new_id] = new_route
            route_load[new_id] += route_load[rj]
            del routes[rj]
            del route_load[rj]
            for c in new_route[1:-1]:
                client_route[c] = new_id
        elif route_end(rj) == j and route_start(ri) == i:
            new_route = routes[rj][:-1] + routes[ri][1:]
            new_id = rj
            routes[new_id] = new_route
            route_load[new_id] += route_load[ri]
            del routes[ri]
            del route_load[ri]
            for c in new_route[1:-1]:
                client_route[c] = new_id

    return list(routes.values()), list(route_load.values())


def route_distance(route):
    total = 0
    for k in range(len(route) - 1):
        total += DIST[route[k]][route[k+1]]
    return total


def two_opt(route):
    """Mejora 2-opt dentro de una ruta."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(new_route) < route_distance(best):
                    best = new_route
                    improved = True
    return best


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Florida Bebidas · CVRP Puntarenas",
    page_icon="🍺",
    layout="wide",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.main { background-color: #0d1117; }
.block-container { padding-top: 2rem; }
.kpi-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.kpi-val { font-size: 2.2rem; font-weight: 800; color: #E63946; font-family: 'Syne', sans-serif; }
.kpi-lbl { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: .08em; margin-top: .3rem; }
.route-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="padding:2rem 0 1rem 0;">
  <div style="color:#E63946;font-size:.8rem;font-family:'Syne',sans-serif;letter-spacing:.2em;font-weight:700;">
    FLORIDA BEBIDAS · FIFCO · II-1122
  </div>
  <h1 style="margin:0;font-size:2.4rem;color:#edf2f4;">
    🍺 Distribución CVRP — <span style="color:#E63946;">Puntarenas</span>
  </h1>
  <p style="color:#8892b0;margin-top:.4rem;">
    Clarke-Wright Savings + 2-opt · 13 cantones · Capacidad 24 pallets/camión
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parámetros")
    capacity = st.slider("Capacidad por camión (pallets)", 12, 48, 24, step=4)
    apply_2opt = st.checkbox("Aplicar mejora 2-opt", value=True)
    show_demand = st.checkbox("Mostrar demanda en mapa", value=True)

    st.divider()
    st.markdown("### 📦 Demanda por cantón")
    df_demand = pd.DataFrame([
        {"Nodo": k, "Cantón": CANTONES[k],
         "Imperial": DEMAND_DETAIL[k][0],
         "Pilsen": DEMAND_DETAIL[k][1],
         "Tropical": DEMAND_DETAIL[k][2],
         "Total": DEMAND[k]}
        for k in DEMAND
    ])
    st.dataframe(df_demand.set_index("Nodo"), use_container_width=True, height=400)
    st.markdown(f"**Total:** {sum(DEMAND.values())} pallets/sem")

# ── Resolver ─────────────────────────────────────────────────
routes_raw, loads_raw = clarke_wright(capacity)

if apply_2opt:
    routes_opt = [two_opt(r) for r in routes_raw]
else:
    routes_opt = routes_raw

routes_final = routes_opt
loads_final  = [sum(DEMAND.get(n,0) for n in r[1:-1]) for r in routes_final]
dists_final  = [route_distance(r) for r in routes_final]
total_dist   = sum(dists_final)
num_trucks   = len(routes_final)

# ── KPIs ─────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
kpis = [
    (f"{total_dist:,.0f} km", "Distancia total"),
    (str(num_trucks), "Camiones utilizados"),
    (f"{sum(DEMAND.values())} pallets", "Demanda semanal"),
    (f"{(sum(loads_final)/(num_trucks*capacity)*100):.1f}%", "Utilización flota"),
]
for col,(val,lbl) in zip([c1,c2,c3,c4],kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Mapa de rutas ─────────────────────────────────────────────
st.markdown("### 🗺️ Mapa de Rutas")

fig_map = go.Figure()

# Arcos de rutas
for idx, route in enumerate(routes_final):
    color = COLORS[idx % len(COLORS)]
    lons, lats = [], []
    for node in route:
        lat, lon = COORDS[node]
        lats.append(lat); lons.append(lon)
    fig_map.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode="lines",
        line=dict(width=4, color=color),
        name=f"Ruta {idx+1} ({loads_final[idx]}/{capacity}p · {dists_final[idx]:.0f}km)",
        hoverinfo="skip",
    ))

# Nodos
for node_id, (lat, lon) in COORDS.items():
    is_depot = node_id == 0
    d = DEMAND.get(node_id, 0)
    size = 18 if is_depot else (8 + d // 5)
    color = "#C1121F" if is_depot else "#1B6CA8"
    text = f"<b>{CANTONES[node_id]}</b><br>Demanda: {d} pallets" if not is_depot else "<b>CD Puntarenas</b><br>Depósito"
    fig_map.add_trace(go.Scattermapbox(
        lat=[lat], lon=[lon],
        mode="markers+text" if is_depot else "markers",
        marker=dict(size=size, color=color),
        text=["🏭"] if is_depot else [],
        textposition="top center",
        hovertext=text,
        hoverinfo="text",
        showlegend=is_depot,
        name="CD Puntarenas (depósito)" if is_depot else "",
    ))

fig_map.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=9.3, lon=-84.0),
        zoom=7.2,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=520,
    legend=dict(
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#cbd5e0",
        borderwidth=1,
        font=dict(color="#1a202c", size=11),
    ),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
)
st.plotly_chart(fig_map, use_container_width=True)

# ── Detalle de rutas ─────────────────────────────────────────
st.markdown("### 🚛 Detalle de Rutas")

cols = st.columns(2)
for idx, (route, load, dist_km) in enumerate(zip(routes_final, loads_final, dists_final)):
    with cols[idx % 2]:
        color = COLORS[idx % len(COLORS)]
        stops = " → ".join(CANTONES[n] for n in route)
        utilization = load / capacity * 100
        st.markdown(f"""
        <div style="background:#1a1f2e;border:1px solid {color}40;border-left:4px solid {color};
                    border-radius:10px;padding:1rem;margin-bottom:.8rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-family:'Syne',sans-serif;font-weight:700;color:{color};font-size:1rem;">
                    Camión {idx+1}
                </span>
                <span style="color:#8892b0;font-size:.8rem;">{dist_km:.0f} km</span>
            </div>
            <div style="color:#edf2f4;font-size:.82rem;margin:.5rem 0;line-height:1.6;">
                {stops}
            </div>
            <div style="display:flex;gap:1rem;font-size:.78rem;">
                <span style="color:#2A9D8F;">📦 {load}/{capacity} pallets</span>
                <span style="color:#E9C46A;">⚡ {utilization:.0f}% utilización</span>
            </div>
            <div style="background:#0d1117;border-radius:4px;height:6px;margin-top:.6rem;">
                <div style="background:{color};width:{utilization}%;height:6px;border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Tabla resumen ─────────────────────────────────────────────
st.markdown("### 📊 Resumen por Ruta")

summary_data = []
for idx, (route, load, dist_km) in enumerate(zip(routes_final, loads_final, dists_final)):
    clients_in_route = [CANTONES[n] for n in route if n != 0]
    summary_data.append({
        "Camión": idx + 1,
        "Cantones atendidos": len(clients_in_route),
        "Paradas": ", ".join(clients_in_route),
        "Pallets": f"{load}/{capacity}",
        "Utilización": f"{load/capacity*100:.0f}%",
        "Distancia (km)": int(dist_km),
    })

df_summary = pd.DataFrame(summary_data)
st.dataframe(df_summary.set_index("Camión"), use_container_width=True)

# ── Gráfico de utilización ────────────────────────────────────
st.markdown("### 📈 Utilización y Distancia por Camión")

col_a, col_b = st.columns(2)
with col_a:
    fig_util = go.Figure(go.Bar(
        x=[f"C{i+1}" for i in range(len(routes_final))],
        y=[l/capacity*100 for l in loads_final],
        marker_color=COLORS[:len(routes_final)],
        text=[f"{l/capacity*100:.0f}%" for l in loads_final],
        textposition="outside",
    ))
    fig_util.update_layout(
        title="Utilización por camión (%)",
        yaxis=dict(range=[0, 110], ticksuffix="%", color="#8892b0"),
        xaxis_color="#8892b0",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#edf2f4"),
        height=320,
        margin=dict(t=40,b=20,l=20,r=20),
        shapes=[dict(type="line", y0=100, y1=100, x0=-0.5,
                     x1=len(routes_final)-0.5,
                     line=dict(color="#E63946", dash="dash", width=2))],
    )
    st.plotly_chart(fig_util, use_container_width=True)

with col_b:
    fig_dist = go.Figure(go.Bar(
        x=[f"C{i+1}" for i in range(len(routes_final))],
        y=dists_final,
        marker_color=COLORS[:len(routes_final)],
        text=[f"{d:.0f}" for d in dists_final],
        textposition="outside",
    ))
    fig_dist.update_layout(
        title="Distancia recorrida por camión (km)",
        yaxis_color="#8892b0",
        xaxis_color="#8892b0",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#edf2f4"),
        height=320,
        margin=dict(t=40,b=20,l=20,r=20),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ── Matriz de distancias ─────────────────────────────────────
with st.expander("🔢 Matriz de Distancias (km)"):
    labels = [f"{k}-{CANTONES[k][:8]}" for k in range(14)]
    df_dist = pd.DataFrame(DIST_MATRIX_RAW, index=labels, columns=labels)
    fig_heat = px.imshow(
        df_dist,
        color_continuous_scale="YlOrRd",
        title="Matriz de distancias por carretera (km)",
        height=500,
    )
    fig_heat.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#edf2f4"),
        coloraxis_colorbar=dict(tickfont=dict(color="#edf2f4")),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Modelo AMPL ──────────────────────────────────────────────
with st.expander("📄 Código AMPL del Modelo"):
    st.code(open("cvrp_puntarenas.mod").read() if False else """
# AMPL Model — CVRP Florida Bebidas / Puntarenas

set NODES;
set CLIENTS = NODES diff {0};

param demand {CLIENTS} >= 0;
param dist   {NODES, NODES} >= 0;
param Q      := 24;
param max_vehicles := ceil(sum {i in CLIENTS} demand[i] / Q);
set VEHICLES := 1..max_vehicles;

var x {NODES, NODES, VEHICLES} binary;
var u {CLIENTS, VEHICLES}       >= 0;

minimize TotalDist:
    sum {k in VEHICLES, i in NODES, j in NODES: i != j}
        dist[i,j] * x[i,j,k];

subject to VisitOnce {j in CLIENTS}:
    sum {k in VEHICLES, i in NODES: i != j} x[i,j,k] = 1;

subject to FlowBalance {k in VEHICLES, h in NODES}:
    sum {i in NODES: i != h} x[i,h,k] =
    sum {j in NODES: j != h} x[h,j,k];

subject to DepotOut {k in VEHICLES}:
    sum {j in CLIENTS} x[0,j,k] <= 1;

subject to Capacity {k in VEHICLES}:
    sum {i in CLIENTS, j in NODES: i != j} demand[i] * x[i,j,k] <= Q;

subject to Subtour {k in VEHICLES, i in CLIENTS, j in CLIENTS: i != j}:
    u[i,k] - u[j,k] + Q * x[i,j,k] <= Q - demand[j];
    """, language="ampl")

st.markdown("""
---
<div style="text-align:center;color:#4a5568;font-size:.8rem;padding:1rem 0;">
  Florida Bebidas (FIFCO) · Diseño de Rutas CVRP · Provincia Puntarenas · Curso II-1122
</div>
""", unsafe_allow_html=True)
