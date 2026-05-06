#!/bin/bash
# Ejecuta todos los reactores con la misma semilla y guarda los plots por separado.

GAMMA=0.9
SEED=42

REACTORS=(
    "Reactors/R0.json"
    "Reactors/R1.json"
    "Reactors/R2.json"
    "Reactors/R3.json"
    "Reactors/R4.json"
    "Reactors/R5.json"
    "Reactors/R6.json"
)

for reactor in "${REACTORS[@]}"; do
    echo "=============================="
    echo "Ejecutando: $reactor"
    echo "=============================="
    python main.py --input-reactor "$reactor" --gamma $GAMMA --random-seed $SEED
    echo ""
done

echo "Todos los reactores procesados. Plots en la carpeta: plots/"
