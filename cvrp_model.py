"""
============================================================
CVRP - Florida Bebidas / Puntarenas
Modelo MIP en Python usando PuLP
Curso: Diseño de rutas de distribución (CVRP) · II-1122
============================================================
Instalar: pip install pulp
Ejecutar:  python cvrp_model.py
"""

import pulp
import math

# ─── DATOS ──────────────────────────────────────────────────
CANTONES = {
    0: "CD Puntarenas", 1: "Puntarenas",    2: "Esparza",
    3: "Buenos Aires",  4: "Montes de Oro", 5: "Osa",
    6: "Quepos",        7: "Golfito",        8: "Coto Brus",
    9: "Parrita",      10: "Corredores",    11: "Garabito",
   12: "Monteverde",   13: "Puerto Jiménez",
}

DEMAND = {
    1: 107, 2: 27,  3: 37,  4: 12,
    5: 28,  6: 24,  7: 33,  8: 35,
    9: 16, 10: 39, 11: 20, 12:  4, 13: 8,
}

DIST_RAW = [
    #  0    1    2    3    4    5    6    7    8    9   10   11   12   13
    [  0,   0,  25, 244,  27, 243, 124, 307, 307,  99, 332,  60,  47, 303],
    [  0,   0,  25, 244,  27, 243, 124, 307, 307,  99, 332,  60,  47, 303],
    [ 25,  25,   0, 224,  19, 226, 109, 290, 287,  85, 314,  55,  49, 288],
    [244, 244, 224,   0, 240,  41, 124,  80,  63, 150,  95, 196, 267,  92],
    [ 27,  27,  19, 240,   0, 244, 127, 308, 303, 103, 331,  73,  30, 305],
    [243, 243, 226,  41, 244,   0, 119,  64,  79, 145,  90, 189, 272,  64],
    [124, 124, 109, 124, 127, 119,   0, 183, 186,  26, 208,  72, 156, 179],
    [307, 307, 290,  80, 308,  64, 183,   0,  54, 208,  31, 253, 336,  25],
    [307, 307, 287,  63, 303,  79, 186,  54,   0, 212,  46, 259, 330,  78],
    [ 99,  99,  85, 150, 103, 145,  26, 208, 212,   0, 234,  47, 133, 204],
    [332, 332, 314,  95, 331,  90, 208,  31,  46, 234,   0, 279, 359,  52],
    [ 60,  60,  55, 196,  73, 189,  72, 253, 259,  47, 279,   0, 102, 246],
    [ 47,  47,  49, 267,  30, 272, 156, 336, 330, 133, 359, 102,   0, 335],
    [303, 303, 288,  92, 305,  64, 179,  25,  78, 204,  52, 246, 335,   0],
]

NODES    = list(range(14))
CLIENTS  = list(range(1, 14))
CAPACITY = 24
DEPOT    = 0
dist     = {(i, j): DIST_RAW[i][j] for i in NODES for j in NODES}

K = math.ceil(sum(DEMAND.values()) / CAPACITY)   # número máximo de vehículos
VEHICLES = list(range(1, K + 1))

# ─── MODELO PuLP ────────────────────────────────────────────
model = pulp.LpProblem("CVRP_Puntarenas", pulp.LpMinimize)

# Variables
x = pulp.LpVariable.dicts(
    "x",
    [(i, j, k) for i in NODES for j in NODES for k in VEHICLES if i != j],
    cat="Binary",
)
u = pulp.LpVariable.dicts(
    "u",
    [(i, k) for i in CLIENTS for k in VEHICLES],
    lowBound=0, cat="Continuous",
)

# Objetivo: minimizar distancia total
model += pulp.lpSum(
    dist[i, j] * x[i, j, k]
    for i in NODES for j in NODES for k in VEHICLES if i != j
)

# (1) Cada cliente visitado exactamente una vez
for j in CLIENTS:
    model += pulp.lpSum(
        x[i, j, k] for i in NODES for k in VEHICLES if i != j
    ) == 1

# (2) Balance de flujo
for k in VEHICLES:
    for h in NODES:
        model += (
            pulp.lpSum(x[i, h, k] for i in NODES if i != h) ==
            pulp.lpSum(x[h, j, k] for j in NODES if j != h)
        )

# (3) Cada vehículo sale del depósito como máximo una vez
for k in VEHICLES:
    model += pulp.lpSum(x[DEPOT, j, k] for j in CLIENTS) <= 1

# (4) Capacidad
for k in VEHICLES:
    model += pulp.lpSum(
        DEMAND[i] * x[i, j, k]
        for i in CLIENTS for j in NODES if i != j
    ) <= CAPACITY

# (5) Eliminación de subtours (MTZ)
for k in VEHICLES:
    for i in CLIENTS:
        for j in CLIENTS:
            if i != j:
                model += (
                    u[i, k] - u[j, k] + CAPACITY * x[i, j, k]
                    <= CAPACITY - DEMAND[j]
                )

# (6) Límites de carga acumulada
for k in VEHICLES:
    for i in CLIENTS:
        model += u[i, k] >= DEMAND[i]
        model += u[i, k] <= CAPACITY

# ─── RESOLVER ───────────────────────────────────────────────
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=300, gapRel=0.05)
status = model.solve(solver)

# ─── RESULTADOS ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SOLUCIÓN CVRP — FLORIDA BEBIDAS / PUNTARENAS")
print("=" * 55)
print(f"  Estado    : {pulp.LpStatus[status]}")
print(f"  Distancia : {pulp.value(model.objective):,.1f} km")
print("=" * 55)

for k in VEHICLES:
    arcs = [
        (i, j) for i in NODES for j in NODES
        if i != j and pulp.value(x[i, j, k]) and pulp.value(x[i, j, k]) > 0.5
    ]
    if not arcs:
        continue

    # Reconstruir ruta ordenada desde el depósito
    route = [DEPOT]
    while True:
        current = route[-1]
        nxt = next((j for (i, j) in arcs if i == current), None)
        if nxt is None or nxt == DEPOT:
            route.append(DEPOT)
            break
        route.append(nxt)

    load     = sum(DEMAND.get(n, 0) for n in route)
    dist_km  = sum(dist[route[t], route[t+1]] for t in range(len(route)-1))
    names    = " → ".join(CANTONES[n] for n in route)
    print(f"\n  Camión {k}  |  {load}/{CAPACITY} pallets  |  {dist_km} km")
    print(f"  {names}")

print("\n" + "=" * 55)
