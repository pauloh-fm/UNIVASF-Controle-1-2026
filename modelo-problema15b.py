"""
Módulo modelo-problema15b.py — Núcleo de Cálculos Matemáticos Puros & Geração do Gráfico Matplotlib
Capítulo 3, Problema 15(b) - Modelagem no Domínio do Tempo (Norman Nise)

Este arquivo contém a lógica matemática, representação no espaço de estados,
dedução algébrica via SymPy, cálculo de polos, simulação numérica e geração do gráfico Matplotlib.
Pode ser executado de forma totalmente autônoma.
"""

import os
import numpy as np
import sympy as sp
import control as ct
import matplotlib.pyplot as plt


def obter_matrizes():
    """
    Retorna as matrizes A, B, C e D do Problema 15(b) como arrays NumPy.
    
    A: (5x5) Matriz de dinâmica interna do sistema
    B: (5x1) Vetor de entrada
    C: (1x5) Vetor de saída
    D: (1x1) Transmissão direta
    """
    A = np.array([
        [ 3.0,  1.0,  0.0,  4.0, -2.0],
        [-3.0,  5.0, -5.0,  2.0, -1.0],
        [ 0.0,  1.0, -1.0,  2.0,  8.0],
        [-7.0,  6.0, -3.0, -4.0,  0.0],
        [-6.0,  0.0,  4.0, -3.0,  1.0]
    ], dtype=float)

    B = np.array([
        [2.0],
        [7.0],
        [8.0],
        [5.0],
        [4.0]
    ], dtype=float)

    C = np.array([[1.0, -2.0, -9.0, 7.0, 6.0]], dtype=float)
    D = np.array([[0.0]], dtype=float)

    # Validação de dimensões
    assert A.shape == (5, 5), f"Dimensão de A incorreta: {A.shape}"
    assert B.shape == (5, 1), f"Dimensão de B incorreta: {B.shape}"
    assert C.shape == (1, 5), f"Dimensão de C incorreta: {C.shape}"
    assert D.shape == (1, 1), f"Dimensão de D incorreta: {D.shape}"

    return A, B, C, D


def criar_sistema_control():
    """
    Cria e retorna o objeto de sistema LTI contínuo (StateSpace) usando Python Control.
    """
    A, B, C, D = obter_matrizes()
    return ct.ss(A, B, C, D)


def obter_funcao_transferencia_simbolica():
    """
    Dedução Simbólica em Laplace:
    Calcula G(s) = C * (s*I - A)^(-1) * B + D utilizando a biblioteca SymPy.

    Retorna:
        G_simbolica: Expressão simplificada de G(s)
        num_exp: Polinômio do numerador expandido
        den_exp: Polinômio do denominador expandido
    """
    s = sp.symbols('s')

    A_mat = sp.Matrix([
        [ 3,  1,  0,  4, -2],
        [-3,  5, -5,  2, -1],
        [ 0,  1, -1,  2,  8],
        [-7,  6, -3, -4,  0],
        [-6,  0,  4, -3,  1]
    ])

    B_mat = sp.Matrix([
        [2],
        [7],
        [8],
        [5],
        [4]
    ])

    C_mat = sp.Matrix([[1, -2, -9, 7, 6]])
    D_mat = sp.Matrix([[0]])

    I = sp.eye(A_mat.rows)

    # Fórmula: G(s) = C * (sI - A)^(-1) * B + D
    sI_minus_A = s * I - A_mat
    sI_minus_A_inv = sI_minus_A.inv()

    G_matriz = C_mat * sI_minus_A_inv * B_mat + D_mat
    G_raw = G_matriz[0]

    # Simplificação e fracionamento de termos
    G_simbolica = sp.cancel(sp.simplify(G_raw))

    numerador, denominador = sp.fraction(G_simbolica)
    num_exp = sp.expand(numerador)
    den_exp = sp.expand(denominador)

    return G_simbolica, num_exp, den_exp


def obter_polos_e_estabilidade():
    """
    Calcula os polos numéricos do sistema (autovalores de A) e classifica a estabilidade.
    """
    A, _, _, _ = obter_matrizes()
    polos = np.linalg.eigvals(A)
    
    polos_ordenados = sorted(polos, key=lambda p: (p.real, p.imag))
    tem_polo_instavel = any(p.real > 1e-6 for p in polos_ordenados)
    estabilidade = "instável" if tem_polo_instavel else "estável"

    justificativa = (
        "Existe pelo menos um polo com parte real positiva no semiplano direito (RHP), "
        "o que gera uma resposta temporal com crescimento exponencial indeterminado."
        if tem_polo_instavel else
        "Todos os polos possuem parte real estritamente negativa (semiplano esquerdo LHP)."
    )

    return polos_ordenados, estabilidade, justificativa


def simular_degrau(sistema, tempo_final=2.0, quantidade_pontos=1000):
    """
    Simula a resposta temporal do sistema LTI a uma entrada degrau unitário r(t) = 1(t).

    Retorna:
        tempo: Vetor de amostragem temporal (s)
        saida: Resposta y(t)
    """
    t_span = np.linspace(0, tempo_final, quantidade_pontos)
    resposta = ct.step_response(sistema, timepts=t_span)
    
    tempo = resposta.time
    saida = np.squeeze(resposta.outputs)
    return tempo, saida


def gerar_grafico_matplotlib(tempo, saida, caminho_imagem=None):
    """
    Gera e salva a imagem PNG da resposta ao degrau utilizando Matplotlib.
    """
    if caminho_imagem is None:
        caminho_imagem = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resposta_degrau_15b.png")

    diretorio_destino = os.path.dirname(caminho_imagem)
    if diretorio_destino and not os.path.exists(diretorio_destino):
        os.makedirs(diretorio_destino, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor='white')
    ax.set_facecolor('white')
    
    ax.plot(tempo, saida, color='#dc2626', linewidth=2, label='y(t) - Saída do Sistema (Instável)')
    ax.axhline(0, color='#64748b', linestyle='--', linewidth=1)
    ax.axhline(1, color='#059669', linestyle=':', linewidth=1.5, label='r(t) = 1 - Entrada Degrau')

    ax.set_title('Capítulo 3, Problema 15(b) — Resposta ao Degrau Unitário (Python Control / Matplotlib)', fontsize=12, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Tempo (s)', fontsize=10, color='#0f172a')
    ax.set_ylabel('Saída y(t)', fontsize=10, color='#0f172a')
    ax.tick_params(colors='#0f172a', labelsize=9)
    ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax.legend(loc='upper left', fontsize=9, facecolor='#f8fafc', edgecolor='#cbd5e1')
    
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')

    plt.tight_layout()
    plt.savefig(caminho_imagem, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Gráfico Matplotlib salvo em: {os.path.abspath(caminho_imagem)}")


if __name__ == "__main__":
    A, B, C, D = obter_matrizes()
    print("=== MATRIZES DO ESPAÇO DE ESTADOS (15b) ===")
    print("A =\n", A)
    print("B =\n", B)
    print("C =\n", C)
    print("D =\n", D)

    G, num, den = obter_funcao_transferencia_simbolica()
    print("\n=== FUNÇÃO DE TRANSFERÊNCIA SIMBÓLICA G(s) ===")
    print("G(s) =", G)
    print("Numerador:", num)
    print("Denominador:", den)

    polos, est, just = obter_polos_e_estabilidade()
    print("\n=== ANÁLISE DE POLOS E ESTABILIDADE ===")
    for idx, p in enumerate(polos, 1):
        sign = "+" if p.imag >= 0 else "-"
        print(f"Polo {idx}: {p.real:.6f} {sign} {abs(p.imag):.6f}j")
    print(f"Estabilidade: {est}")
    print(f"Justificativa: {just}")

    sys_ctrl = criar_sistema_control()
    t_val, y_val = simular_degrau(sys_ctrl)
    gerar_grafico_matplotlib(t_val, y_val)
