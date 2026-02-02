import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import sys
import scipy.io
from scipy.integrate import simpson
from scipy.special import kv, hyperu
from mpmath import mp, meijerg, exp
import scipy.special
from scipy.integrate import romb
from fractions import Fraction
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mp.dps = 25

# -----------------------
def whittakerW(kappa, mu, z):
    return np.exp(-z / 2) * z**(mu + 0.5) * hyperu(mu - kappa + 0.5, 2 * mu + 1, z)

def get_f_alpha(alpha, z):
    z = np.asarray(z)
    if alpha == 0.5:
        return (1 / (2 * np.sqrt(np.pi))) * z**(-1.5) * np.exp(-1 / (4 * z))
    elif alpha == 1/3:
        return (1 / (3 * np.pi)) * z**(-1.5) * kv(1/3, 2 / np.sqrt(27 * z))
    elif alpha == 2/3:
        y = 4 / (27 * z**2)
        return (np.sqrt(3 / np.pi)) * z**(-1) * np.exp(-y / 2) * whittakerW(0.5, 1/6, y)
    else:
        print("error")
        sys.exit(1)

def extract_ctrw_times(file_list):
    times = []
    for fname in file_list:
        match = re.search(r'(\d+(?:\.\d+)?)\.mat$', fname)
        if match:
            times.append(float(match.group(1)))
        else:
            times.append(np.nan)
    return np.array(times)


# -----------------------
save_dir = "p2_num_0.5"
os.makedirs(save_dir, exist_ok=True)
ctrw_dir = 'ctrw/ctrw_0.5'
alpha = 1/2
frac = Fraction(alpha).limit_denominator()
step = 0.05
x_vals = np.arange(-2.8, 3.2, step)
t_base = np.arange(0.001, 200.0, step)
t_special = np.array([0.1,1.0,2.0,7.0])
t_vals = np.unique(np.concatenate((t_base, t_special)))

s_log = np.logspace(np.log10(0.001), np.log10(0.1), 30)
s_linear = np.arange(0.1, 8.0, step*2)
s_vals = np.unique(np.concatenate((s_log, s_linear)))

X, S = np.meshgrid(x_vals, s_vals)
S2, T = np.meshgrid(s_vals, t_vals)
U_pred = np.load('num_p1.npy')

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(S, X, U_pred, cmap='viridis')
ax1.set_xlabel('s', fontsize=18, labelpad=12)
ax1.set_ylabel('x', fontsize=18, labelpad=12)
ax1.set_zlabel(r'$P_1(x,s)$', fontsize=18, labelpad=12)
ax1.tick_params(axis='both', which='major', labelsize=12)  
ax1.zaxis.set_tick_params(labelsize=12)   
ax1.set_title(r'$P_1(x,s)$', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'p1_check_3D.pdf'))
plt.close()

# -----------------------
# n(s,t)
X_st = T * S2**(-1 / alpha)       
f_alpha = get_f_alpha(alpha, X_st)
n_vals = (T / alpha) * S2**(-1 / alpha - 1) * f_alpha

# -----------------------
# P_alpha(x, t)
P_alpha_vals = np.zeros((len(t_vals), len(x_vals))) 
for i, t in enumerate(t_vals):
    for j, x1 in enumerate(x_vals):
        n_st = n_vals[i, :] 
        P1_s = U_pred[:, j] 
        integrand = n_st * P1_s
        P_alpha_vals[i, j] = np.trapz(integrand, s_vals)

plt.figure(2)
t_targets = list(t_special)
colors = ['r', 'g', 'b', 'm']  
stride = 4  
ctrw_files = sorted([f for f in os.listdir(ctrw_dir) if f.endswith('.mat')])
ctrw_times = extract_ctrw_times(ctrw_files)
for t_val, color in zip(t_targets, colors):
    if t_val==0.1:
        stride = 5
    elif t_val==1.0:
        stride = 6
    else:
        stride = 8
    idx = np.argmin(np.abs(t_vals - t_val))
    closest_t = t_vals[idx]
    pred_vals = P_alpha_vals[idx, :].flatten()
    mask = (x_vals >= -10) & (x_vals <= 10)
    plt.plot(x_vals[mask], pred_vals[mask], color=color, label=rf'SNN $t={closest_t:.1f}$')
    
    # CTRW 
    ctrw_idx = np.argmin(np.abs(ctrw_times - t_val))
    ctrw_file = ctrw_files[ctrw_idx]
    ctrw_data = scipy.io.loadmat(os.path.join(ctrw_dir, ctrw_file))
    centers = ctrw_data['centers'].flatten()
    pdf_hist = ctrw_data['pdf_hist'].flatten()
    
    mask2 = (centers >= -10) & (centers <= 10)
    centers_selected = centers[mask2][::stride]
    pdf_selected = pdf_hist[mask2][::stride]
    plt.plot(centers_selected, pdf_selected, marker='o', markersize=4, linestyle='none', color=color, label=rf'MC $t={ctrw_times[ctrw_idx]:.1f}$')

plt.xlabel('x', fontsize=18)
plt.ylabel(r'$P(x,t)$', fontsize=18)
plt.title(rf'$P(x,t)$, $\alpha={frac.numerator}/{frac.denominator}$'.format(alpha), fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'p_alpha_fitted_t.pdf'))
plt.close()


# -----------------------
S1_vals = []
for i, t in enumerate(s_vals):
    P1_xt = U_pred[i, :]
    S_t = simpson(P1_xt, x=x_vals)
    S1_vals.append(S_t)
S1_vals = np.array(S1_vals)

S2_vals = []
for i, t in enumerate(t_vals):
    P_xt = P_alpha_vals[i, :]
    S_t = simpson(P_xt, x=x_vals)
    S2_vals.append(S_t)
S2_vals = np.array(S2_vals)
min_idx = np.argmin(S2_vals)

plt.figure(3)
plt.loglog(s_vals, S1_vals, 'r', label=r'SNN $S(t),\alpha=1$')
plt.loglog(t_vals, S2_vals, 'g', label=r'SNN $S(t),\alpha=1/2$')

S50 = interp1d(t_vals, S2_vals)(50)
t_ref = t_vals[(t_vals >= 20) & (t_vals <= 200)]
plt.loglog(t_ref, 0.8*S50 * (t_ref/50)**-0.5, 'k--', label=r'$t^{-1/2}$')

# -----------------------
ctrw_dir2 = 'ctrw'
ctrw_files = sorted([f for f in os.listdir(ctrw_dir2) if f.endswith('.mat')])
for fname in ctrw_files:
    mat_data = scipy.io.loadmat(os.path.join(ctrw_dir2, fname))
    
    if fname.split('_')[-1].startswith('1'):
        t_mc = mat_data['t1'].flatten()
        S1_mc = mat_data['S1'].flatten()
        color = 'r'
        label = r'MC $S(t),\alpha=1$'
        n_points = 70
    elif fname.split('_')[-1].startswith('2'):
        t_mc = mat_data['t'].flatten()
        S_mc = mat_data['S'].flatten()
        S1_mc = S_mc  
        color = 'g'
        label = r'MC $S(t),\alpha=1/2$'
        n_points = 40
    else:
        continue
    sort_idx = np.argsort(t_mc)
    t_mc = t_mc[sort_idx]
    S1_mc = S1_mc[sort_idx]
    mask = t_mc > 0
    t_mc = t_mc[mask]
    S1_mc = S1_mc[mask]
    log_t_min = np.log10(t_mc[0])
    log_t_max = np.log10(t_mc[-1])
    t_selected = 10**np.linspace(log_t_min, log_t_max, n_points)
    S_selected = np.interp(np.log10(t_selected), np.log10(t_mc), S1_mc)
    plt.loglog(t_selected, S_selected, marker='o', markersize=4, linestyle='none', color=color, label=label)
plt.xlim(10**-1.0, 250)
plt.ylim(10**-2.7, 2)
plt.xlabel('t', fontsize=18)
plt.ylabel(r'$S(t)$', fontsize=18)

# ================= 
ax = plt.gca()
ax_in = inset_axes(ax, width="33%", height="33%", loc='lower right',bbox_to_anchor=(0.0, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=1.2)
ax_in.set_xlim(0.1, 11)
ax_in.semilogx(s_vals, S1_vals, 'r', linewidth=1)
for fname in ctrw_files:
    if fname.split('_')[-1].startswith('1'):
        mat_data = scipy.io.loadmat(os.path.join(ctrw_dir2, fname))
        t_mc = mat_data['t1'].flatten()
        S1_mc = mat_data['S1'].flatten()
        idx = np.argsort(t_mc)
        t_mc, S1_mc = t_mc[idx], S1_mc[idx]
        mask = t_mc > 0
        t_mc = t_mc[mask]
        S1_mc = S1_mc[mask]
        n_points = 50
        log_t_min = np.log10(t_mc[0])
        log_t_max = np.log10(t_mc[-1])
        t_selected = 10**np.linspace(log_t_min, log_t_max, n_points)
        S_selected = np.interp(np.log10(t_selected), np.log10(t_mc), S1_mc)
        ax_in.semilogx(t_selected, S_selected, 'o', color='r', markersize=3)
        break
ax_in.set_yticks([0.0, 0.5, 1.0])
ax_in.set_yticklabels(['0.0', '0.5', '1.0'])
ax_in.set_title(r'$\alpha=1$', fontsize=10)
ax_in.tick_params(axis='both', which='major', labelsize=9, length=3)
ax_in.tick_params(axis='both', which='minor', labelsize=9, length=2)
ax_in.grid(False)

ax.legend(loc='lower left', fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(os.path.join(save_dir, 'survival_log.pdf'))
plt.close()



