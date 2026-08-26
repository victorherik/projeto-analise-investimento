# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. IMPORTAÇÃO DOS DADOS
# ============================================================

df = pd.read_csv("investimento_vendas.csv")

print("\nPrimeiras linhas do dataset:")
print(df.head())


# ============================================================
# 2. DEFINIÇÃO DAS VARIÁVEIS
# ============================================================

X = df[['investimento_publicidade_mil_reais']]
y = df['vendas_mil_unidades']


# ============================================================
# 3. VISUALIZAÇÃO DOS DADOS
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    color='steelblue'
)

plt.xlabel('Investimento em Publicidade (mil R$)')
plt.ylabel('Vendas (mil unidades)')
plt.title('Investimento em Publicidade vs Vendas')
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# 4. DIVISÃO DOS DADOS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. TREINAMENTO DO MODELO
# ============================================================

modelo = LinearRegression()

modelo.fit(X_train, y_train)


# ============================================================
# 6. COEFICIENTE E INTERCEPTO
# ============================================================

print("\nResultados do modelo:")

print(
    f"Coeficiente (β1): {modelo.coef_[0]:.4f}"
)

print(
    f"Intercepto (β0): {modelo.intercept_:.4f}"
)


# ============================================================
# 7. PREVISÕES
# ============================================================

y_pred = modelo.predict(X_test)


# ============================================================
# 8. MÉTRICAS
# ============================================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("\nMétricas do modelo:")

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# ============================================================
# 9. GRÁFICO DA REGRESSÃO
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    color='steelblue',
    label='Dados reais'
)

X_sorted = X.sort_values(
    by=X.columns[0]
)

plt.plot(
    X_sorted,
    modelo.predict(X_sorted),
    color='red',
    linewidth=2,
    label='Reta de regressão'
)

plt.xlabel(
    'Investimento em Publicidade (mil R$)'
)

plt.ylabel(
    'Vendas (mil unidades)'
)

plt.title(
    'Regressão Linear Simples'
)

plt.legend()

plt.grid(alpha=0.3)

plt.show()


# ============================================================
# 10. PREVISÃO PARA NOVO INVESTIMENTO
# ============================================================

novo_investimento = pd.DataFrame({
    'investimento_publicidade_mil_reais': [25]
})

previsao = modelo.predict(
    novo_investimento
)

print(
    f"\nPara um investimento de R$ 25 mil, "
    f"a previsão de vendas é: "
    f"{previsao[0]:.2f} mil unidades"
)


# ============================================================
# 11. GRÁFICO FINAL
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6),
    dpi=120
)


# Pontos reais
ax.scatter(
    X,
    y,
    color='#2E86AB',
    s=70,
    alpha=0.75,
    edgecolors='white',
    linewidth=0.8,
    label='Dados reais',
    zorder=3
)


# Reta de regressão
ax.plot(
    X_sorted,
    modelo.predict(X_sorted),
    color='#F24236',
    linewidth=3,
    label='Reta de regressão',
    zorder=4
)


# Área de erro
ax.fill_between(
    X_sorted[X_sorted.columns[0]],
    modelo.predict(X_sorted) - rmse,
    modelo.predict(X_sorted) + rmse,
    color='#F24236',
    alpha=0.08,
    zorder=1,
    label='Faixa de erro (±RMSE)'
)


# Títulos
ax.set_title(
    'Investimento em Publicidade vs Vendas',
    fontsize=16,
    fontweight='bold',
    pad=20
)

ax.set_xlabel(
    'Investimento em Publicidade (mil R$)',
    fontsize=12,
    labelpad=10
)

ax.set_ylabel(
    'Vendas (mil unidades)',
    fontsize=12,
    labelpad=10
)


# R²
ax.text(
    0.03,
    0.95,
    f'R² = {r2:.3f}',
    transform=ax.transAxes,
    fontsize=13,
    fontweight='bold',
    color='#F24236',
    bbox=dict(
        boxstyle='round,pad=0.4',
        facecolor='white',
        edgecolor='#F24236',
        alpha=0.9
    ),
    verticalalignment='top'
)


# Legenda
ax.legend(
    frameon=True,
    fontsize=10,
    loc='lower right'
)


# Grade
ax.grid(
    True,
    alpha=0.25,
    linestyle='--'
)


# Remover bordas superiores e direitas
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)


plt.tight_layout()


# Salvar gráfico
plt.savefig(
    'regressao_investimento_vendas.png',
    dpi=300,
    bbox_inches='tight'
)


plt.show()