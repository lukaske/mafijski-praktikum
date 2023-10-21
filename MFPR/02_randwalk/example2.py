import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
import markdown
import datetime
import scipy.stats as stats
import scipy.optimize as opt
import scipy.interpolate as interp

# Function and constants defitition for Juptyer notebook (rerun for hot reload of imports)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['text.usetex'] = True
plt.rcParams["grid.linestyle"] = "dashed"
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')

# Silence warnings, ups...
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def centerHTML(html, text=""):
    text = markdown.markdown(text)
    return '<div style="display: flex; align-items: center; flex-direction: column;">' + html + f'</div><div style="display: flex; align-items: center; flex-direction: column;padding-top: 15px;"><small style="max-width: 600px">{text}</small></div>'


def generateL(mu, m, n, seed):
    a = mu - 1
    np.random.seed(seed)
    return (np.random.pareto(a, n) + 1) * m


def generatePhi(n=10, seed=10001):
    np.random.seed(seed)
    return np.random.uniform(0, 2 * np.pi, n)


def generateWalk(mu=4, m=2, n=10, seed1=10001, seed2=10002):
    L = generateL(mu, m, n, seed1)
    phi = generatePhi(n, seed2)
    x = np.cumsum(L * np.cos(phi))
    y = np.cumsum(L * np.sin(phi))
    return x, y


def plot_with_gradient_color(ax, x, y, colors):
    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], 'o-', color=colors[i], markersize=1, linewidth=1)


def generate_config(elements, mu_low=2.25, mu_high=3.75):
    steps = [10, 100, 1000, 10000]
    return {
        "mu": np.flip(np.round(np.linspace(mu_low, mu_high, elements), 2)),
        "seed": np.linspace(13401, 16302, elements, dtype=int),
        "n": [steps] * elements,
        "t_f": [max(steps)] * elements,
        "t_w": [np.inf] * elements,
        "sigma^2": [np.inf] * elements,
        "gamma": [np.inf] * elements,
        "tip_f": ["?"] * elements,
        "tip_w": ["?"] * elements,
    }


def simulate_with_config_row(config_row, num_of_simulations=300):
    mu = config_row["mu"]
    initial_seed = config_row["seed"]
    n = np.max(config_row["n"])
    np_int = np.iinfo(np.int32)
    new_seed = initial_seed + datetime.datetime.now().microsecond
    np.random.seed(new_seed)
    seeds = np.random.randint(0, np_int.max, 2 * num_of_simulations + 1)
    df_rows = []
    for i in range(num_of_simulations):
        seed1 = int(seeds[i])
        seed2 = int(seeds[2 * i + 1])
        x, y = generateWalk(mu, seed1=seed1, seed2=seed2, n=n)
        mus = np.full_like(x, mu)
        seeds1 = np.full_like(x, seed1)
        seeds2 = np.full_like(x, seed2)
        t = np.arange(0, len(x))
        df_rows.append(np.column_stack((mus, x, y, seeds1, seeds2, t)))
    return np.vstack(df_rows)


def simulate_with_config(config, num_of_simulations=300):
    df_rows = []
    for i in range(len(config.index)):
        df_rows.append(simulate_with_config_row(config.iloc[i], num_of_simulations))
    df = pd.DataFrame(np.vstack(df_rows), columns=["mu", "x", "y", "seed1", "seed2", "t"])
    return df


def distribute_over_t(x_y_data, type):
    if type == "f":
        pass
    elif type == "w":
        pass
    else:
        raise ValueError("type must be either f or w")


def gamma_flight(mu):
    if mu > 3:
        return 1
    elif 3 > mu > 1:
        return 2 / (mu - 1)
    else:
        return 0


def gamma_stick(mu):
    nu = 1.9
    if mu > 3:
        return nu - 1
    elif 3 > mu > 1:
        return 2 + nu - mu
    else:
        return 0


def gamma_walk(mu):
    if mu > 3:
        return 1
    elif 3 > mu > 2:
        return 4 - mu
    elif 2 > mu > 1:
        return 2
    else:
        return 0


def lin_f(x, k, n):
    return k * x + n


def interpolate_df(arr):
    points = arr[:, 0:2]
    shifted_x = np.roll(points, -1, axis=0)
    velocity = np.linalg.norm(shifted_x - points, axis=1)[:-1].round().astype(int)
    t = np.arange(0, np.sum(velocity), 1)
    x_vectors = (shifted_x - points)[:-1, 0] / velocity
    y_vectors = (shifted_x - points)[:-1, 1] / velocity
    norm_x = np.cumsum(np.repeat(x_vectors, velocity))
    norm_y = np.cumsum(np.repeat(y_vectors, velocity))
    return np.column_stack((norm_x, norm_y, t))

# Generate pandas dataframe config for random walks
df = pd.DataFrame(generate_config(4))
df.to_parquet("data/config.parquet")
try:
    simulations_df = pd.read_parquet("data/simulations.parquet")
except Exception as e:
    print('Warning: Could not find simulations.parquet, generating new dataset...')
    simulations_df = simulate_with_config(df, 300)
    simulations_df.to_parquet("data/simulations.parquet")
    print('Done! Rerun this cell to get rid of this message.')
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(6,6))

df_indexed = simulations_df.set_index(['mu', 't'])
unique_mu = df_indexed.index.get_level_values(0).unique()
mu = unique_mu[0]
unique_sims = df_indexed.loc[mu].index.get_level_values(0).unique()

bins = [np.arange(-400, 400, 10), np.arange(-400, 400, 10)]
_, _, _, img = ax.hist2d(np.zeros((300)), np.zeros((300)), bins=bins)
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.colorbar(img, ax=ax, label="Stevilo delcev")
ax.set_title(f'Levyjevi poleti - 300 delcev, $\mu = {3.75}$ , $t = {1300 + 5}$ s')

plt.show()
1/0
def animate(i) :
    j = (i - 1) * 5
    if i == 0:
        h, _, _, img1 = plt.hist2d(np.zeros((300)), np.zeros((300)), bins=bins)
    else:
        simulation = df_indexed.loc[(mu, j)]
        sim = simulation.to_numpy()
        h, _, _, img1 = plt.hist2d(sim[:, 0], sim[:, 1], bins=bins)
    ax.set_title(f'Levyjev polet za 300 delcev, $\mu$ = {mu} , $t$ = {j}')
    img.set_array(h)

anim = animation.FuncAnimation(fig, animate, frames=500, interval = 20, blit = False)
writervideo = animation.FFMpegWriter(fps=60)
anim.save(f'./media/disperzija-{mu}.mp4', writer=writervideo)
