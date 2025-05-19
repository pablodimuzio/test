import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, diff, sympify, solveset, Poly, S
import io

# --- Title ---
st.title("Explorateur de fractale de Newton interactive")

# --- Inputs fonction et dérivée ---
st.sidebar.header("Fonction")
f_input = st.sidebar.text_input("Entrez f(z)", value="z**3 - 1")
d_input = st.sidebar.text_input("Entrez f'(z) (laisser vide pour dériver)", value="")

# --- Paramètres de la visualisation ---
st.sidebar.header("Paramètres de la vue")
width = st.sidebar.slider("Largeur (px)", 200, 1000, 600)
height = st.sidebar.slider("Hauteur (px)", 200, 1000, 600)
max_iter = st.sidebar.slider("Itérations max", 10, 200, 50)
tolerance = st.sidebar.slider("Tolérance", 1e-8, 1e-3, 1e-8, format="%.0e")
zoom = st.sidebar.slider("Zoom", 0.1, 5.0, 2.0)
x_center = st.sidebar.slider("Centre X", -5.0, 5.0, 0.0)
y_center = st.sidebar.slider("Centre Y", -5.0, 5.0, 0.0)

# --- Création d'expression symbolique ---
z = symbols('z')
try:
    expr = sympify(f_input)
except Exception as e:
    st.error(f"Expression invalide: {e}")
    st.stop()

# dérivée automatique si non fournie
if d_input.strip() == "":
    d_expr = diff(expr, z)
else:
    try:
        d_expr = sympify(d_input)
    except Exception as e:
        st.error(f"Dérivée invalide: {e}")
        st.stop()

# conversion en fonctions numpy
f = lambdify(z, expr, 'numpy')
df = lambdify(z, d_expr, 'numpy')

# tentative de calcul des racines
try:
    poly = Poly(expr, z)
    coeffs = poly.all_coeffs()
    roots = np.roots([complex(c) for c in coeffs])
except Exception:
    sol = solveset(expr, z, domain=S.Complexes)
    roots = np.array([complex(r.evalf()) for r in sol])

# --- Génération fractale ---
x_min, x_max = x_center - zoom, x_center + zoom
y_min, y_max = y_center - zoom, y_center + zoom
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
output = np.zeros(Z.shape, dtype=int)
iterations = np.zeros(Z.shape, dtype=int)

for i in range(max_iter):
    Z = Z - f(Z) / df(Z)
    for j, r in enumerate(roots):
        mask = (np.abs(Z - r) < tolerance) & (output == 0)
        output[mask] = j + 1
        iterations[mask] = i

# --- Color mapping ---
colors = np.zeros((height, width, 3))
colormap = np.random.RandomState(0).rand(len(roots), 3)
for i, color in enumerate(colormap):
    mask = output == i + 1
    for c in range(3):
        colors[..., c][mask] = color[c] * (1 - iterations[mask] / max_iter)

# --- Affichage ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(colors, extent=(x_min, x_max, y_min, y_max))
ax.axis('off')

st.pyplot(fig)

# --- Exportation PNG ---
buf = io.BytesIO()
fig.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
st.download_button("Télécharger l'image", data=buf, file_name="newton_fractal.png", mime="image/png")
