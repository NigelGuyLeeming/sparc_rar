# ============================================================
# 0. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# ============================================================
# DATA RETENTION REPORTING
# ============================================================

def report_stage(df, label, initial_count=None):
    """
    Print a clean summary of how many rows remain at a given stage.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The current dataframe.
    label : str
        Name of the stage (e.g. 'After mask', 'After PCA prep').
    initial_count : int or None
        If provided, prints how many rows were removed since that count.
    """
    current = len(df)
    print("------------------------------------------------------------")
    print(f" {label}")
    print("------------------------------------------------------------")
    print(f"Rows at this stage: {current}")
    if initial_count is not None:
        print(f"Rows removed since previous stage: {initial_count - current}")
    print("------------------------------------------------------------\n")
    
    return current  # so you can pass it to the next stage

# ============================================================
# 1. LOAD + CLEAN DATA
# ============================================================

df = pd.read_csv("sparc_rarplus.csv")
initial_raw = len(pd.read_csv("sparc_rarplus.csv"))


# Ensure ID is a string
df["ID"] = df["ID"].astype(str)

# Columns we expect to be numeric
float_cols = [
    "Distance","Galactocentric Radius","Vobs","e_Vobs","Vgas","Vdisk","Vbul",
    "SBdisk","SBbul","Radius","gobs","gbar","L10(gobs)","L10(gbar)",
    "tdyn","Luminosity","Mass"
]

for col in float_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Unified mask: positive, finite, non-missing
mask = (
    df["gbar"].notna() &
    df["gobs"].notna() &
    df["tdyn"].notna() &
    df["Vobs"].notna() &
    df["Radius"].notna() &
    (df["gbar"] > 0) &
    (df["gobs"] > 0) &
    (df["tdyn"] > 0) &
    (df["Vobs"] > 0) &
    (df["Radius"] > 0)
)

df = df[mask].copy()
count_after_mask = report_stage(df, "After positivity + non-missing mask", initial_raw)



# Log coordinates for both spaces
df["lgbar"] = np.log10(df["gbar"])
df["lgobs"] = np.log10(df["gobs"])
df["ltdyn"] = np.log10(df["tdyn"])

# New coordinate system logs
df["lVobs"] = np.log10(df["Vobs"])
df["lR"]    = np.log10(df["Radius"])

galaxies = df["ID"].unique()

# ============================================================
# 2. SIDE-BY-SIDE PLOTS: ORIGINAL (carrot) vs NEW (fin)
# ============================================================

fig = plt.figure(figsize=(14, 6))

# Left: original coordinates (carrot)
ax1 = fig.add_subplot(121, projection="3d")
cmap = plt.cm.get_cmap("turbo", len(galaxies))

for i, gal in enumerate(galaxies):
    sub = df[df["ID"] == gal]
    ax1.scatter(
        sub["lgbar"], sub["lgobs"], sub["ltdyn"],
        s=8, alpha=0.6, color=cmap(i)
    )

ax1.set_xlabel(r"$\log g_{\rm bar}$")
ax1.set_ylabel(r"$\log g_{\rm obs}$")
ax1.set_zlabel(r"$\log t_{\rm dyn}$")
ax1.set_title("Original coordinate space")

# Right: new coordinates (fin)
ax2 = fig.add_subplot(122, projection="3d")

for i, gal in enumerate(galaxies):
    sub = df[df["ID"] == gal]
    ax2.scatter(
        sub["lgbar"], sub["lVobs"], sub["lR"],
        s=8, alpha=0.6, color=cmap(i)
    )

ax2.set_xlabel(r"$\log g_{\rm bar}$")
ax2.set_ylabel(r"$\log V_{\rm obs}$")
ax2.set_zlabel(r"$\log R$")
ax2.set_title("New coordinate space")

plt.tight_layout()
plt.show()

# ============================================================
# 3. PREPARE 3D COORDINATES IN NEW SPACE FOR PCA
# ============================================================

coords3d = np.vstack([
    df["lgbar"].values,
    df["lVobs"].values,
    df["lR"].values
]).T  # shape (N, 3)

print("coords3d shape:", coords3d.shape)

# ============================================================
# 4. PCA FLATTENING TO (u, v)
# ============================================================

pca = PCA(n_components=2)
uv = pca.fit_transform(coords3d)

u = uv[:, 0]
v = uv[:, 1]

# Report retention (no rows removed, but keeps the chain clean)
count_after_pca = report_stage(df, "After PCA flattening", count_after_mask)

print("Explained variance ratios:", pca.explained_variance_ratio_)
print("PCA components (rows = PCs):")
print(pca.components_)

# ============================================================
# 5. PLOT THE FIN IN THE UV-PLANE
# ============================================================

plt.figure(figsize=(7, 6))
plt.scatter(u, v, s=4, alpha=0.5, c="black")
plt.xlabel("u (1st PCA component)")
plt.ylabel("v (2nd PCA component)")
plt.title("SPARC fin in PCA-flattened (u, v) plane")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. DATA LOADING
# ============================================================

# Raw data cloud in uv-plane
uv_uv = np.column_stack([u, v])   # shape (N,2)


# ============================================================
# 2. BÉZIER GEOMETRY CORE (XYZ COORDINATES)
# ============================================================

def bezier_xyz(P0_xyz, P1_xyz, P2_xyz, P3_xyz, t):
    """Evaluate cubic Bézier curve at parameter array t."""
    t = t[:, None]
    return ((1 - t)**3) * P0_xyz + \
           (3 * (1 - t)**2 * t) * P1_xyz + \
           (3 * (1 - t) * t**2) * P2_xyz + \
           (t**3) * P3_xyz

def bezier_derivative_xyz(P0_xyz, P1_xyz, P2_xyz, P3_xyz, t):
    """Evaluate derivative of cubic Bézier curve at parameter array t."""
    t = t[:, None]
    return 3*(1 - t)**2 * (P1_xyz - P0_xyz) + \
           6*(1 - t)*t   * (P2_xyz - P1_xyz) + \
           3*t**2        * (P3_xyz - P2_xyz)

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0. LOAD DATA (uv-plane)
# ============================================================

uv_uv = np.column_stack([u, v])   # raw data cloud


# ============================================================
# 1. BASE LINE + NORMAL (for marching)
# ============================================================

P0_base_uv = np.array([2.0, 2.0])
P1_base_uv = np.array([0.0, 0.0])

d_uv = P1_base_uv - P0_base_uv
n_uv = np.array([d_uv[1], -d_uv[0]])
n_uv = n_uv / np.linalg.norm(n_uv)


# ============================================================
# 2. DISTANCE TO INFINITE LINE
# ============================================================

def distance_to_line_uv(points_uv, P0_uv, d_uv):
    """Distance of each point to infinite line through P0 with direction d."""
    v = points_uv - P0_uv
    t = (v @ d_uv) / np.dot(d_uv, d_uv)
    proj = P0_uv + t[:, None] * d_uv
    return np.linalg.norm(points_uv - proj, axis=1)


# ============================================================
# 3. NEAREST POINT WITHIN TOLERANCE
# ============================================================

def nearest_within_tol_uv(points_uv, P0_uv, d_uv, tol):
    """Return nearest point to P0 on the line, within tolerance."""
    dist = distance_to_line_uv(points_uv, P0_uv, d_uv)
    mask = dist < tol
    if not np.any(mask):
        return None
    candidates = points_uv[mask]
    d_to_P0 = np.linalg.norm(candidates - P0_uv, axis=1)
    return candidates[np.argmin(d_to_P0)]


# ============================================================
# 4. MARCH AND COLLECT SUPPORT POINTS
# ============================================================

def march_and_collect_uv(points_uv, P0_uv, d_uv, n_uv, tol=0.1, step=0.1, steps=200):
    """March along normal direction and collect nearest support points."""
    support = []
    P = P0_uv.copy()

    for _ in range(steps):
        nearest = nearest_within_tol_uv(points_uv, P, d_uv, tol)
        if nearest is not None:
            if len(support) == 0 or np.linalg.norm(nearest - support[-1]) > 1e-6:
                support.append(nearest)
        P = P + step * n_uv

    return np.array(support)


support_f_uv = march_and_collect_uv(uv_uv, P0_base_uv, d_uv,  n_uv)
support_b_uv = march_and_collect_uv(uv_uv, P0_base_uv, d_uv, -n_uv)
support_uv   = np.vstack([support_f_uv, support_b_uv])


# ============================================================
# 5. SORT SUPPORT POINTS ALONG MARCHING DIRECTION
# ============================================================

proj = (support_uv - P0_base_uv) @ n_uv
support_sorted_uv = support_uv[np.argsort(proj)]


# ============================================================
# 6. TRIM ENDS
# ============================================================

support_clean_uv = support_sorted_uv[4:-4]


# ============================================================
# 7. FIT CUBIC BÉZIER TO SUPPORT POINTS
# ============================================================

def fit_bezier_to_support_uv(support_uv):
    """Fit cubic Bézier to ordered support points."""
    diffs = np.diff(support_uv, axis=0)
    seg_len = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    t = s / s[-1]

    P0 = support_uv[0]
    P3 = support_uv[-1]

    B0 = (1 - t)**3
    B1 = 3 * (1 - t)**2 * t
    B2 = 3 * (1 - t) * t**2
    B3 = t**3

    A = np.column_stack([B1, B2])
    rhs = support_uv - (B0[:, None] * P0 + B3[:, None] * P3)

    sol_x, _, _, _ = np.linalg.lstsq(A, rhs[:, 0], rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, rhs[:, 1], rcond=None)

    P1 = np.array([sol_x[0], sol_y[0]])
    P2 = np.array([sol_x[1], sol_y[1]])

    return P0, P1, P2, P3


P0_fit_uv, P1_fit_uv, P2_fit_uv, P3_fit_uv = fit_bezier_to_support_uv(support_clean_uv)


# ============================================================
# 8. BÉZIER EVALUATION (uv-plane)
# ============================================================

def bezier_uv(P0, P1, P2, P3, t):
    t = t[:, None]
    B0 = (1 - t)**3
    B1 = 3 * (1 - t)**2 * t
    B2 = 3 * (1 - t) * t**2
    B3 = t**3
    return B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3


t_plot = np.linspace(0, 1, 400)
curve_uv = bezier_uv(P0_fit_uv, P1_fit_uv, P2_fit_uv, P3_fit_uv, t_plot)


# ============================================================
# 9. EXTEND ENDS
# ============================================================

def extend_ends_uv(P0, P1, P2, P3, amount=1.0):
    """Extend Bézier endpoints along tangent directions."""
    # start tangent
    T0 = P1 - P0
    if np.linalg.norm(T0) < 1e-6: T0 = P2 - P0
    if np.linalg.norm(T0) < 1e-6: T0 = P3 - P0
    T0 = T0 / np.linalg.norm(T0)

    # end tangent
    T1 = P3 - P2
    if np.linalg.norm(T1) < 1e-6: T1 = P3 - P1
    if np.linalg.norm(T1) < 1e-6: T1 = P3 - P0
    T1 = T1 / np.linalg.norm(T1)

    return P0 - amount*T0, P1, P2, P3 + amount*T1


P0_ext_uv, P1_ext_uv, P2_ext_uv, P3_ext_uv = extend_ends_uv(
    P0_fit_uv, P1_fit_uv, P2_fit_uv, P3_fit_uv
)


# ============================================================
# 10. SLIDE CURVE OUTWARD UNTIL SAFE
# ============================================================

def offset_curve_uv(P0, P1, P2, P3, uv_uv, step=0.01, max_iter=200):
    """Slide Bézier outward until all data lie on one side."""
    centroid = uv_uv.mean(axis=0)

    for _ in range(max_iter):
        C = bezier_uv(P0, P1, P2, P3, t_plot)

        # tangent via finite difference
        eps = 1e-4
        C_plus  = bezier_uv(P0, P1, P2, P3, t_plot + eps)
        C_minus = bezier_uv(P0, P1, P2, P3, t_plot - eps)
        dC = C_plus - C_minus
        dC = dC / np.linalg.norm(dC, axis=1, keepdims=True)

        # normals
        nC = np.stack([dC[:,1], -dC[:,0]], axis=1)

        # ensure outward
        for j in range(len(t_plot)):
            if (centroid - C[j]) @ nC[j] > 0:
                nC[j] = -nC[j]

        # signed distances
        signed = np.array([(uv_uv - C[j]) @ nC[j] for j in range(len(t_plot))])
        if np.max(signed) <= 0:
            return P0, P1, P2, P3

        # slide outward
        shift = step * np.mean(nC, axis=0)
        P0 += shift; P1 += shift; P2 += shift; P3 += shift

    return P0, P1, P2, P3


P0_final_uv, P1_final_uv, P2_final_uv, P3_final_uv = offset_curve_uv(
    P0_ext_uv, P1_ext_uv, P2_ext_uv, P3_ext_uv, uv_uv
)

curve_final_uv = bezier_uv(P0_final_uv, P1_final_uv, P2_final_uv, P3_final_uv, t_plot)
# ============================================================
# 3. LOAD FINAL CONTROL POINTS AND FLIP BÉZIER
# ============================================================

P0_xyz = np.asarray(P0_final_uv)
P1_xyz = np.asarray(P1_final_uv)
P2_xyz = np.asarray(P2_final_uv)
P3_xyz = np.asarray(P3_final_uv)

# Flip parameterisation: t → 1 - t
P0_xyz, P1_xyz, P2_xyz, P3_xyz = (
    P3_xyz,
    P2_xyz,
    P1_xyz,
    P0_xyz
)

print("Flipped Bézier control points:")
for name, P in zip(["P0","P1","P2","P3"], [P0_xyz, P1_xyz, P2_xyz, P3_xyz]):
    print(f"  {name} = ({P[0]:.4f}, {P[1]:.4f})")


# ============================================================
# 4. SAMPLE CURVE AND DERIVATIVE
# ============================================================

t_curve = np.linspace(0, 1, 2000)
C_xyz  = bezier_xyz(P0_xyz, P1_xyz, P2_xyz, P3_xyz, t_curve)
C1_xyz = bezier_derivative_xyz(P0_xyz, P1_xyz, P2_xyz, P3_xyz, t_curve)


# ============================================================
# 5. TANGENT + NORMAL AT INDEX
# ============================================================

def tangent_normal_at_xyz(i):
    """Return unit tangent and unit normal at curve index i."""
    T = C1_xyz[i]
    T = T / np.linalg.norm(T)
    N = np.array([T[1], -T[0]])   # rotate tangent 90° CCW
    return T, N


# ============================================================
# 6. SIDE-OF-LINE TEST (CROSS PRODUCT)
# ============================================================

def count_sides_uv(P_xyz, N_xyz, uv_uv):
    """Count how many uv points lie on each side of the normal line."""
    x0, y0 = P_xyz
    dx, dy = N_xyz
    pos = neg = zero = 0
    for (x, y) in uv_uv:
        side = (x - x0)*dy - (y - y0)*dx
        if side > 0:
            pos += 1
        elif side < 0:
            neg += 1
        else:
            zero += 1
    return pos, neg, zero


# ============================================================
# 7. FORWARD SWEEP TO FIND ORIGIN
# ============================================================

origin_index = None
cross_index  = None

for i in range(len(C_xyz)):
    P_i = C_xyz[i]
    T_i, N_i = tangent_normal_at_xyz(i)
    pos, neg, zero = count_sides_uv(P_i, N_i, uv_uv)

    if pos > 0 and neg > 0:
        cross_index  = i
        origin_index = i - 1   # last clean supporting normal
        break

# Extract origin + axes
origin_xyz = C_xyz[origin_index]
e1_xyz, e2_xyz = tangent_normal_at_xyz(origin_index)

print("\n--- Coordinate system definition ---")
print(f"Origin: ({origin_xyz[0]:.6f}, {origin_xyz[1]:.6f})")
print(f"Tangent e1: ({e1_xyz[0]:.6f}, {e1_xyz[1]:.6f})")
print(f"Normal  e2: ({e2_xyz[0]:.6f}, {e2_xyz[1]:.6f})")


# ============================================================
# 8. COORDINATE TRANSFORM (XYZ → XR)
# ============================================================

def to_xr(P_xyz, origin_xyz, e1_xyz, e2_xyz):
    """Transform world coordinate P_xyz into fin coordinates (x,r)."""
    d = P_xyz - origin_xyz
    x = np.dot(d, e1_xyz)
    r = np.dot(d, e2_xyz)
    return np.array([x, r])


# ============================================================
# 9. PLOTTING UTILITIES (SEPARATE FROM GEOMETRY)
# ============================================================



# --- Missing function restored ---
def normal_at_xyz(i):
    """Return unit tangent and unit normal at curve index i."""
    T = C1_xyz[i]
    T = T / np.linalg.norm(T)
    N = np.array([T[1], -T[0]])   # rotate tangent 90° CCW
    return T, N    

P_origin = C_xyz[origin_index]
T_origin, N_origin = normal_at_xyz(origin_index)

print("\n--- Coordinate system definition ---")
print("Origin (on Bézier):")
print("  x =", P_origin[0])
print("  y =", P_origin[1])

print("\nTangent axis at origin (e1):")
print("  Tx =", T_origin[0])
print("  Ty =", T_origin[1])

print("\nNormal axis at origin (e2):")
print("  Nx =", N_origin[0])
print("  Ny =", N_origin[1])

# ============================================================
# 8. Plot data, curve, and origin normal
# ============================================================

def draw_normal_xyz(P_xyz, N_xyz, color='red', lw=2, label=""):
    """Draw a normal line through P_xyz in world coordinates."""
    L = 4
    x0, y0 = P_xyz
    dx, dy = N_xyz
    x1, y1 = x0 + L*dx, y0 + L*dy
    x2, y2 = x0 - dx/4.0, y0 - dy/4.0
    plt.plot([x1, x2], [y1, y2], color=color, lw=lw, label=label)


# --- Plotting ---
plt.figure(figsize=(6,6))
plt.scatter(uv_uv[:,0], uv_uv[:,1], s=4, alpha=0.6)
plt.plot(C_xyz[:,0], C_xyz[:,1], 'k-', lw=2, label="Flipped Bézier")

# origin normal
draw_normal_xyz(origin_xyz, e2_xyz, color='red', lw=2, label="Normal")
plt.scatter([origin_xyz[0]], [origin_xyz[1]], color='red', s=40, label="Origin")

## crossing normal (optional)
#P_cross_xyz = C_xyz[cross_index]
#_, N_cross_xyz = normal_at_xyz(cross_index)
#draw_normal_xyz(P_cross_xyz, N_cross_xyz, color='orange', lw=1)
#plt.scatter([P_cross_xyz[0]], [P_cross_xyz[1]], color='orange', s=30, label="Crossing")

plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# INTRINSIC COORDINATES (x, r) FROM BÉZIER
# ============================================================

# 1. Arc-length along the curve → x(t)

diffs = np.diff(C_xyz, axis=0)
seg_len = np.sqrt((diffs**2).sum(axis=1))
s_curve = np.concatenate([[0.0], np.cumsum(seg_len)])  # same length as t_curve
L_total = s_curve[-1]

# Normalise if you want x in [0,1], else keep physical arc-length
x_of_t = s_curve  # or s_curve / L_total

# 2. For each data point, find nearest point on the curve

N_pts = uv_uv.shape[0]
x_vals = np.zeros(N_pts)
r_vals = np.zeros(N_pts)

for i, P in enumerate(uv_uv):
    # nearest sample on the curve
    d2 = np.sum((C_xyz - P)**2, axis=1)
    j = np.argmin(d2)

    # intrinsic x from arc-length
    x_vals[i] = x_of_t[j]

    # tangent + normal at that index
    T, N = tangent_normal_at_xyz(j)

    # signed normal distance → r
    r_vals[i] = (P - C_xyz[j]) @ N

# 3. Quick xr-plot

plt.figure(figsize=(7,6))
plt.scatter(x_vals, r_vals, s=3, alpha=0.4, color="black")
plt.xlabel("x (arc-length along Bézier)")
plt.ylabel("r (signed normal distance)")
plt.title("Intrinsic coordinates (x, r)")
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# COLOUR-CODED XR-PLOT BY GALAXY (using df["ID"])
# ============================================================

galaxy_labels = df["ID"].values
unique_galaxies = np.unique(galaxy_labels)

# A stable colour map with one colour per galaxy
#cmap = plt.cm.get_cmap("tab20", len(unique_galaxies))
cmap = plt.cm.get_cmap("turbo", len(unique_galaxies))

colour_lookup = {g: cmap(i) for i, g in enumerate(unique_galaxies)}

colours = np.array([colour_lookup[g] for g in galaxy_labels])

plt.figure(figsize=(8,6))
plt.scatter(x_vals, r_vals, s=6, c=colours, alpha=0.85)
plt.xlabel("x (arc-length along Bézier)")
plt.ylabel("r (signed normal distance)")
plt.title("Intrinsic coordinates (x, r) coloured by galaxy")
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# LIFT BÉZIER AXIS BACK INTO 3D: (lgbar, lVobs, lR)
# ============================================================

# 1. Original 3D data cloud in the physical RAR space
X3d = df[["lgbar", "lVobs", "lR"]].values

# 2. Lift the uv Bézier back into this 3D space using inverse PCA
C_3d = C_xyz @ pca.components_ + pca.mean_

# 3. Colour by galaxy (Turbo)
galaxy_labels = df["ID"].values
unique_galaxies = np.unique(galaxy_labels)
cmap = plt.cm.get_cmap("turbo", len(unique_galaxies))
colour_lookup = {g: cmap(i) for i, g in enumerate(unique_galaxies)}
colours = np.array([colour_lookup[g] for g in galaxy_labels])

# 4. 3D plot
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X3d[:,0], X3d[:,1], X3d[:,2],
    s=6, c=colours, alpha=0.7
)

ax.plot(
    C_3d[:,0], C_3d[:,1], C_3d[:,2],
    color="black", linewidth=3, label="Intrinsic axis"
)

ax.set_xlabel("log g_bar")
ax.set_ylabel("log V_obs")
ax.set_zlabel("log R")
ax.set_title("RAR fin in 3D with intrinsic Bézier axis")
ax.legend()

plt.show()

# ============================================================
# CROSS-SECTIONS PERPENDICULAR TO THE 3D AXIS (FRACTIONAL)
# ============================================================

X3d = df[["lgbar", "lVobs", "lR"]].values  # same basis as PCA

# 1. Tangent along the 3D axis
diffs_3d = np.diff(C_3d, axis=0)
T_3d = diffs_3d / np.linalg.norm(diffs_3d, axis=1, keepdims=True)
T_3d = np.vstack([T_3d[0], T_3d])  # pad

# 2. Orthonormal frame (T, N1, N2)
N1_3d = np.zeros_like(T_3d)
N2_3d = np.zeros_like(T_3d)
ref = np.array([0.0, 0.0, 1.0])

for i in range(len(T_3d)):
    t = T_3d[i]
    if np.abs(np.dot(t, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    n1 = np.cross(t, ref); n1 /= np.linalg.norm(n1)
    n2 = np.cross(t, n1);  n2 /= np.linalg.norm(n2)
    N1_3d[i] = n1
    N2_3d[i] = n2

# 3. Choose sections by fraction along the axis
section_fracs = [0.25] #, 0.5, 0.75]  # 25%, 50%, 75%
section_indices = [int(f * (len(C_3d) - 1)) for f in section_fracs]

# 4. Slab half-thickness along the axis (tune this)
slab_half_thickness = 0.05  # try 0.05, 0.1, etc.

for idx, frac in zip(section_indices, section_fracs):
    P0 = C_3d[idx]
    t  = T_3d[idx]
    n1 = N1_3d[idx]
    n2 = N2_3d[idx]

    V = X3d - P0
    s = V @ t  # longitudinal coordinate

    slab_mask = np.abs(s) < slab_half_thickness
    V_slab = V[slab_mask]

    u_cs = V_slab @ n1
    v_cs = V_slab @ n2

    # Stack into 2D array
    coords = np.column_stack([u_cs, v_cs])

    # PCA in the cross-section plane to find the long axis
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]          # direction of max spread
    orthogonal = np.array([-principal[1], principal[0]]) # 90° rotated

    # Rotate into (u_flat, v_flat) where u_flat is the long axis
    u_flat = coords @ principal
    v_flat = coords @ orthogonal

    # Note, although the code uses u_flat and v_flat, it was written 
    # before we reviewed our entire coordinate system, which had become unwieldy
    # through many transformations

    # u_flat, v_flat is (u,w) in PCA space

    plt.figure(figsize=(6,6))
    plt.scatter(u_flat, v_flat, s=6, alpha=0.6)
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, color="gray", linewidth=1)
    plt.gca().set_aspect("equal", "box")
    plt.title("Cross-section at 25% (flattened)")
    plt.xlabel("u")
    plt.ylabel("w")
    plt.grid(True, alpha=0.3)
    plt.show()




# ============================================================
# 0. Inputs
# ============================================================

P = coords3d          # (N, 3) galaxy points in 3D# 
S = C_3d              # (M, 3) NEW intrinsic axis in 3D (lifted Bézier)

# ============================================================
# 1. Frenet frame along S
# ============================================================

# Tangent
dS = np.gradient(S, axis=0)
T = dS / np.linalg.norm(dS, axis=1, keepdims=True)

# Normal (Frenet)
ddS = np.gradient(T, axis=0)
Nvec = ddS / np.linalg.norm(ddS, axis=1, keepdims=True)

# Binormal
B = np.cross(T, Nvec)
B = B / np.linalg.norm(B, axis=1, keepdims=True)

# ============================================================
# 2. Nearest spine point for each galaxy
# ============================================================

dist = np.linalg.norm(P[:, None, :] - S[None, :, :], axis=2)
spine_index = np.argmin(dist, axis=1)

# ============================================================
# 3. Arc-length parameter x_s along S in [0,1]
# ============================================================

dS_seg = np.diff(S, axis=0)
segment_lengths = np.linalg.norm(dS_seg, axis=1)
arc = np.concatenate([[0], np.cumsum(segment_lengths)])
x_s = arc / arc[-1]     # (M,)

# Each galaxy inherits the x of its nearest spine point
x = x_s[spine_index]

# ============================================================
# 4. Binormal displacement of each galaxy
# ============================================================

d = np.einsum('ij,ij->i', P - S[spine_index], B[spine_index])

# ============================================================
# 5. Bin along x (same logic as before)
# ============================================================

n_bins  = 40
overlap = 0.3
x_min, x_max = 0.0, 1.0
bin_width = (x_max - x_min) / n_bins
step = bin_width * (1 - overlap)

centers = np.arange(x_min + bin_width/2, x_max, step)
mean_d  = np.full_like(centers, np.nan, dtype=float)
N_bin   = np.zeros_like(centers, dtype=int)

N_valid = 3

for k, xc in enumerate(centers):
    mask = (x >= xc - bin_width/2) & (x < xc + bin_width/2)
    N_bin[k] = mask.sum()
    if N_bin[k] >= N_valid:
        mean_d[k] = d[mask].mean()

# ============================================================
# 6. Normalise for colour
# ============================================================

valid = ~np.isnan(mean_d)
if np.any(valid):
    D = np.percentile(np.abs(mean_d[valid]), 95)
    c = np.clip(mean_d / D, -1, 1)
else:
    c = np.zeros_like(mean_d)

c = np.nan_to_num(c, nan=0.0)

# High-confidence bins
N_conf = 20
valid_bins = N_bin >= N_conf



# ============================================================
# 0. Inputs
# ============================================================

P = coords3d      # (N,3) galaxy coordinates in 3D
S = C_3d          # (M,3) NEW intrinsic axis in 3D

# ============================================================
# 1. Compute arc-length parameter x_s along the new axis
# ============================================================

dS = np.diff(S, axis=0)
seg_len = np.linalg.norm(dS, axis=1)
arc = np.concatenate([[0], np.cumsum(seg_len)])
x_s = arc / arc[-1]     # (M,) in [0,1]

# Each galaxy inherits x by nearest-axis projection
dist = np.linalg.norm(P[:,None,:] - S[None,:,:], axis=2)
spine_index = np.argmin(dist, axis=1)
x = x_s[spine_index]

# ============================================================
# 2. Define 7 alternating teal/gold bands
# ============================================================

# Teal and gold RGB values
teal = (0/255, 128/255, 150/255)
gold = (220/255, 170/255, 0/255)

# Seven equal-width bands in x
edges = np.linspace(0, 1, 8)
bands = list(zip(edges[:-1], edges[1:]))

# Alternate colours: teal, gold, teal, ...
band_colors = [teal if i % 2 == 0 else gold for i in range(7)]

# ============================================================
# 3. Plot
# ============================================================

fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111, projection='3d')

# Background cloud
ax.scatter(
    P[:,0], P[:,1], P[:,2],
    c="lightgrey", s=5, alpha=0.15
)

# Banded points
for (x0, x1), col in zip(bands, band_colors):
    sel = (x >= x0) & (x <= x1)
    ax.scatter(
        P[sel,0], P[sel,1], P[sel,2],
        c=[col], s=15,
        label=f"{x0:.2f} ≤ x ≤ {x1:.2f}"
    )

# New intrinsic axis
ax.plot(
    S[:,0], S[:,1], S[:,2],
    color='black', linewidth=3.0, label='Intrinsic axis'
)

ax.set_xlabel("lgbar")
ax.set_ylabel("lVobs")
ax.set_zlabel("lR")
ax.legend()
plt.tight_layout()
plt.show()


###==============================================================
### BINORMAL DISPLACEMENT 
# or 
# cross section through the airfoil whose nose is the leading edge
###===============================================================

from scipy.signal import savgol_filter

# x_fit and y_fit are your sorted high-confidence bins
order = np.argsort(centers[valid_bins])
x_fit = centers[valid_bins][order]
y_fit = mean_d[valid_bins][order]

# Apply a gentle Savitzky–Golay smoothing
# window_length must be odd; adjust 11 → 9 or 13 if needed
yy = savgol_filter(y_fit, window_length=13, polyorder=1)
# uncomment this following line for a closer fit to the data
#yy = savgol_filter(y_fit, window_length=13, polyorder=2)

plt.figure(figsize=(10,4))

# Scatter points
plt.scatter(
    x_fit, y_fit,
    c=c[valid_bins][order],
    cmap='bwr', s=80, edgecolors='none'
)

# Light, dashed, modest trend line
plt.plot(
    x_fit, yy,
    color='black',
    linewidth=1.2,
    linestyle='--',
    alpha=0.6
)

plt.axhline(0, color='grey', linewidth=1)
plt.xlabel("x")
plt.ylabel("n (mean binormal displacement)")
plt.title("Mean binormal displacement in n  vs x (light trend)")
plt.tight_layout()
plt.show()

# ============================================================
# BREADTH PROFILE ALONG THE INTRINSIC AXIS
# ============================================================

X3d = df[["lgbar", "lVobs", "lR"]].values

# Tangent T_3d, N1_3d, N2_3d already computed earlier

slab_half_thickness = 0.05  # tune as needed
breadth = []

for idx in range(len(C_3d)):
    P0 = C_3d[idx]
    t  = T_3d[idx]
    n1 = N1_3d[idx]
    n2 = N2_3d[idx]

    # Vector from axis point to all data
    V = X3d - P0
    s = V @ t

    # Slab selection
    mask = np.abs(s) < slab_half_thickness
    V_slab = V[mask]

    if len(V_slab) < 10:
        breadth.append(np.nan)
        continue

    # Project into cross-section plane
    u_cs = V_slab @ n1
    v_cs = V_slab @ n2
    coords = np.column_stack([u_cs, v_cs])

    # Flatten: PCA in-plane
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]

    # Project onto major axis
    u_flat = coords @ principal

    # Breadth = half-width (e.g. 90% span)
    lo, hi = np.percentile(u_flat, [5, 95])
    breadth.append((hi - lo) / 2)

breadth = np.array(breadth)

from scipy.signal import savgol_filter
# Smooth the jagged breadth curve directly
# breadth is raw jagged array, same length as C_3d
# x_s is intrinsic coordinate (arc-length)

valid = ~np.isnan(breadth)
x_clean = x_s[valid]
b_clean = breadth[valid]

# Sort by x
order = np.argsort(x_clean)
x_clean = x_clean[order]
b_clean = b_clean[order]

# Smooth the jagged breadth curve
import numpy as np
from scipy.optimize import curve_fit

def hill(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))

# x_clean, b_clean already sorted and NaN-free
p0 = [b_clean.max(), x_clean[np.argmax(b_clean)], 0.1]  # sensible initial guess

params, _ = curve_fit(hill, x_clean, b_clean, p0=p0)

A, mu, sigma = params
yy = hill(x_clean, A, mu, sigma)

plt.figure(figsize=(8,5))

plt.plot(x_clean, b_clean, color="gray", linewidth=1, alpha=0.4)
plt.plot(x_clean, yy, color="orange", linewidth=2.5, linestyle='--')

plt.xlabel("x (arc-length along Bézier)")
plt.ylabel("breadth(x)")
plt.title("Breadth profile along intrinsic axis")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Extents in lgbar,lvobs and lR
import numpy as np

cols = ['lgbar', 'lVobs', 'lR']

print("SPARC data ranges in (lgbar, lVobs, lR):")
for c in cols:
    cmin = df[c].min()
    cmax = df[c].max()
    span = cmax - cmin
    print(f"  {c:6s}: min = {cmin:.3f}, max = {cmax:.3f}, span = {span:.3f} dex")

import numpy as np
import pandas as pd

# your seven equal bands
bands = np.array([
    0.0,
    0.142857143,
    0.285714286,
    0.428571429,
    0.571428571,
    0.714285714,
    0.857142857,
    1.0
])

# x = your arc-length values, e.g. x = df['x'].values
# (paste or load your x-values here)

labels = range(1, 8)
df = pd.DataFrame({'x': x})
df['band'] = pd.cut(df['x'], bins=bands, labels=labels, include_lowest=True)

counts = df['band'].value_counts().sort_index()
perc = 100 * counts / counts.sum()

table = pd.DataFrame({'Count': counts, 'Percent': perc.round(1)})
print(table)