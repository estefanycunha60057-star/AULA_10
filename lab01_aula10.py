import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

try:
    PASTA = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PASTA = os.getcwd()

ARQUIVO_IMAGEM = os.path.join(PASTA, "superficie_decisao.png")


def criar_sistema(formato_critica="triangular", metodo="centroid"):
    impacto = ctrl.Antecedent(np.arange(0, 101, 1), "impacto")
    explorabilidade = ctrl.Antecedent(np.arange(0, 11, 1), "explorabilidade")
    urgencia = ctrl.Consequent(np.arange(0, 101, 1), "urgencia")

    impacto["baixo"] = fuzz.trapmf(
        impacto.universe, [0, 0, 20, 40]
    )
    impacto["medio"] = fuzz.trimf(
        impacto.universe, [20, 50, 80]
    )
    impacto["alto"] = fuzz.trapmf(
        impacto.universe, [60, 80, 100, 100]
    )

    explorabilidade["baixa"] = fuzz.trapmf(
        explorabilidade.universe, [0, 0, 2, 4]
    )
    explorabilidade["moderada"] = fuzz.trimf(
        explorabilidade.universe, [2, 5, 8]
    )
    explorabilidade["alta"] = fuzz.trapmf(
        explorabilidade.universe, [6, 8, 10, 10]
    )

    urgencia["baixa"] = fuzz.trapmf(
        urgencia.universe, [0, 0, 15, 35]
    )
    urgencia["media"] = fuzz.trimf(
        urgencia.universe, [20, 40, 60]
    )
    urgencia["alta"] = fuzz.trimf(
        urgencia.universe, [45, 65, 85]
    )

    if formato_critica == "triangular":
        urgencia["critica"] = fuzz.trimf(
            urgencia.universe, [70, 85, 100]
        )
    else:
        urgencia["critica"] = fuzz.trapmf(
            urgencia.universe, [70, 85, 100, 100]
        )

    urgencia.defuzzify_method = metodo

    regras = [
        ctrl.Rule(
            impacto["baixo"] & explorabilidade["baixa"],
            urgencia["baixa"]
        ),
        ctrl.Rule(
            impacto["baixo"] & explorabilidade["moderada"],
            urgencia["baixa"]
        ),
        ctrl.Rule(
            impacto["baixo"] & explorabilidade["alta"],
            urgencia["media"]
        ),
        ctrl.Rule(
            impacto["medio"] & explorabilidade["baixa"],
            urgencia["media"]
        ),
        ctrl.Rule(
            impacto["medio"] & explorabilidade["moderada"],
            urgencia["alta"]
        ),
        ctrl.Rule(
            impacto["medio"] & explorabilidade["alta"],
            urgencia["alta"]
        ),
        ctrl.Rule(
            impacto["alto"] & explorabilidade["baixa"],
            urgencia["critica"]
        ),
        ctrl.Rule(
            impacto["alto"] & explorabilidade["moderada"],
            urgencia["critica"]
        ),
        ctrl.Rule(
            impacto["alto"] & explorabilidade["alta"],
            urgencia["critica"]
        )
    ]

    sistema = ctrl.ControlSystem(regras)

    return impacto, explorabilidade, urgencia, sistema


def calcular_saida(impacto_valor, explorabilidade_valor,
                   formato_critica="triangular", metodo="centroid"):
    impacto, explorabilidade, urgencia, sistema = criar_sistema(
        formato_critica, metodo
    )

    simulacao = ctrl.ControlSystemSimulation(sistema)

    simulacao.input["impacto"] = impacto_valor
    simulacao.input["explorabilidade"] = explorabilidade_valor

    simulacao.compute()

    return simulacao.output["urgencia"]


def comparar_abordagens():
    testes = [
        ("Mínimo", 0, 0),
        ("Intermediário", 50, 5),
        ("Máximo", 100, 10)
    ]

    configuracoes = [
        ("Triangular + Centroide", "triangular", "centroid"),
        ("Trapézio + Centroide", "trapezoidal", "centroid"),
        ("Triangular + MOM", "triangular", "mom"),
        ("Trapézio + MOM", "trapezoidal", "mom"),
        ("Trapézio + LOM", "trapezoidal", "lom")
    ]

    print("\nCOMPARAÇÃO DOS MÉTODOS DE DEFUZZIFICAÇÃO")
    print("-" * 100)
    print(
        f"{'Teste':<16} {'Impacto':>8} {'Expl.':>8} "
        + "".join(f"{nome:>22}" for nome, _, _ in configuracoes)
    )
    print("-" * 100)

    for nome_teste, impacto_valor, explorabilidade_valor in testes:
        valores = []

        for _, formato, metodo in configuracoes:
            resultado = calcular_saida(
                impacto_valor,
                explorabilidade_valor,
                formato,
                metodo
            )
            valores.append(resultado)

        print(
            f"{nome_teste:<16} {impacto_valor:>8} "
            f"{explorabilidade_valor:>8} "
            + "".join(f"{valor:>22.2f}" for valor in valores)
        )

    print("-" * 100)
    print("MOM = Mean of Maximum")
    print("LOM = Largest of Maximum")


def gerar_superficie():
    impacto, explorabilidade, urgencia, sistema = criar_sistema(
        formato_critica="trapezoidal",
        metodo="centroid"
    )

    simulacao = ctrl.ControlSystemSimulation(sistema)

    valores_impacto = np.linspace(0, 100, 51)
    valores_explorabilidade = np.linspace(0, 10, 41)

    X, Y = np.meshgrid(
        valores_impacto,
        valores_explorabilidade
    )

    Z = np.zeros_like(X, dtype=float)

    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):
            simulacao.input["impacto"] = float(X[i, j])
            simulacao.input["explorabilidade"] = float(Y[i, j])
            simulacao.compute()
            Z[i, j] = simulacao.output["urgencia"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    superficie = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        edgecolor="none",
        antialiased=True
    )

    ax.set_title("Superfície de Decisão Fuzzy - Mamdani")
    ax.set_xlabel("Impacto (%)")
    ax.set_ylabel("Explorabilidade")
    ax.set_zlabel("Urgência (%)")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 100)

    fig.colorbar(
        superficie,
        ax=ax,
        shrink=0.65,
        aspect=15,
        label="Urgência (%)"
    )

    plt.tight_layout()
    plt.savefig(ARQUIVO_IMAGEM, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"\nImagem salva em: {ARQUIVO_IMAGEM}")


def visualizar_funcoes():
    impacto, explorabilidade, urgencia, sistema = criar_sistema(
        formato_critica="trapezoidal",
        metodo="centroid"
    )

    fig, eixos = plt.subplots(1, 3, figsize=(16, 4))

    for nome in impacto.terms:
        eixos[0].plot(
            impacto.universe,
            impacto[nome].mf,
            label=nome
        )

    eixos[0].set_title("Funções de Impacto")
    eixos[0].set_xlabel("Impacto (%)")
    eixos[0].set_ylabel("Pertinência")
    eixos[0].legend()
    eixos[0].grid(True, alpha=0.3)

    for nome in explorabilidade.terms:
        eixos[1].plot(
            explorabilidade.universe,
            explorabilidade[nome].mf,
            label=nome
        )

    eixos[1].set_title("Funções de Explorabilidade")
    eixos[1].set_xlabel("Explorabilidade")
    eixos[1].set_ylabel("Pertinência")
    eixos[1].legend()
    eixos[1].grid(True, alpha=0.3)

    for nome in urgencia.terms:
        eixos[2].plot(
            urgencia.universe,
            urgencia[nome].mf,
            label=nome
        )

    eixos[2].set_title("Funções de Urgência")
    eixos[2].set_xlabel("Urgência (%)")
    eixos[2].set_ylabel("Pertinência")
    eixos[2].legend()
    eixos[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    comparar_abordagens()
    visualizar_funcoes()
    gerar_superficie()
