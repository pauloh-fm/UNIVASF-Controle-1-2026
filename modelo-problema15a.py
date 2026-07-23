"""
Módulo modelo-problema15a.py — Núcleo de Cálculos Matemáticos Puros & Simulação
Capítulo 3, Problema 15(a) - Modelagem no Domínio do Tempo (Norman Nise)

Utiliza SymPy e Python Control para obter a Função de Transferência G(s) = C*(sI - A)^(-1)*B + D,
analisar os polos, estabilidade e gerar a resposta temporal ao degrau.
"""

import os
import matplotlib
try:
    matplotlib.use('QtAgg')
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import control as ct


def obter_matrizes():
    """
    Retorna as matrizes A, B, C e D do Problema 15(a) do Nise (6ª Edição).
    """
    A = np.array([
        [0.0,  1.0,  5.0,  0.0],
        [0.0,  0.0,  1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0],
        [-7.0, -9.0, -2.0, -3.0]
    ], dtype=float)

    B = np.array([
        [0.0],
        [5.0],
        [8.0],
        [2.0]
    ], dtype=float)

    C = np.array([[1.0, 3.0, 6.0, 6.0]], dtype=float)
    D = np.array([[0.0]], dtype=float)

    return A, B, C, D


def deduzir_simbolico_sympy():
    """
    Dedução analítica passo a passo via SymPy:
    G(s) = C * (sI - A)^(-1) * B + D
    """
    s = sp.symbols('s')

    A_mat = sp.Matrix([
        [0,  1,  5,  0],
        [0,  0,  1,  0],
        [0,  0,  0,  1],
        [-7, -9, -2, -3]
    ])

    B_mat = sp.Matrix([
        [0],
        [5],
        [8],
        [2]
    ])

    C_mat = sp.Matrix([[1, 3, 6, 6]])
    D_mat = sp.Matrix([[0]])

    I = sp.eye(4)
    sI_minus_A = s * I - A_mat
    sI_minus_A_inv = sI_minus_A.inv()

    G_mat = C_mat * sI_minus_A_inv * B_mat + D_mat
    G_raw = G_mat[0]

    G_simbolica = sp.cancel(sp.simplify(G_raw))
    numerador, denominador = sp.fraction(G_simbolica)

    return G_simbolica, sp.expand(numerador), sp.expand(denominador), sI_minus_A, sI_minus_A.det()


def analisar_estabilidade(A):
    """
    Calcula autovalores (polos) de A e avalia estabilidade.
    """
    polos = np.linalg.eigvals(A)
    estavel = all(p.real < 0 for p in polos)
    return polos, estavel


def gerar_grafico_resposta(sys, caminho_imagem="resposta_degrau_15a.png"):
    """
    Gera gráfico Matplotlib da resposta temporal do sistema.
    """
    t = np.linspace(0, 10, 1000)
    t_out, y_out = ct.step_response(sys, T=t)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.plot(t_out, y_out, color='#1e3a8a', linewidth=2, label='y(t) - Resposta ao Degrau')
    ax.axhline(0, color='#64748b', linestyle='--', linewidth=1)
    ax.axhline(1, color='#10b981', linestyle=':', linewidth=1.5, label='Entrada Degrau Unitário r(t)=1')

    ax.set_title('Capítulo 3, Problema 15(a) — Resposta Temporal ao Degrau Unitário', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Tempo t (segundos)', fontsize=10)
    ax.set_ylabel('Saída y(t)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(caminho_imagem, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico salvo com sucesso em: {os.path.abspath(caminho_imagem)}")
    print("🚀 Abrindo janela do gráfico Matplotlib...")
    plt.show()
    plt.close()


if __name__ == "__main__":
    A, B, C, D = obter_matrizes()
    print("==================================================")
    print("  PROBLEMA 15(a) - CAPÍTULO 3 (NORMAN NISE)")
    print("==================================================")
    
    G_sym, num, den, sI_A, det_sI_A = deduzir_simbolico_sympy()
    print("\n1. Matriz (sI - A):")
    sp.pprint(sI_A)
    
    print("\n2. Determinante det(sI - A):")
    sp.pprint(det_sI_A)

    print("\n3. Função de Transferência G(s) Dedução Simbólica:")
    print(f"G(s) = ({num}) / ({den})")

    polos, estavel = analisar_estabilidade(A)
    print("\n4. Polos do Sistema (Autovalores de A):")
    for i, p in enumerate(polos, 1):
        print(f"   Polo {i}: {p.real:.4f} + {p.imag:.4f}j")
    
    print(f"\n5. Estabilidade: {'ESTÁVEL' if estavel else 'INSTÁVEL'}")

    sys = ct.ss(A, B, C, D)
    sys_tf = ct.tf(sys)
    print("\n6. Sistema em Função de Transferência (Python Control):")
    print(sys_tf)

    gerar_grafico_resposta(sys)
