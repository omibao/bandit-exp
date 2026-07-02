"""
实验一：UCB1、Thompson 采样、KL-UCB 与 epsilon-greedy 在 5 臂伯努利 Bandit 上的对比
"""
import numpy as np
from scipy.special import rel_entr

np.random.seed(42)

K = 5
T = 5000
N_RUNS = 50
TRUE_MEANS = np.array([0.60, 0.55, 0.50, 0.45, 0.40])
BEST_ARM = np.argmax(TRUE_MEANS)
BEST_MEAN = TRUE_MEANS[BEST_ARM]


def kl_ucb_index(success, total, t, c=3):
    """KL-UCB 置信上界：二分搜索求解 max q s.t. n * KL(μ_hat, q) <= f(t)"""
    mu_hat = success / total if total > 0 else 0.0
    f = np.log(t) + c * np.log(np.log(t)) if t > 2 else np.log(t) + c * np.log(np.log(3))
    threshold = f / total if total > 0 else 0.0

    lo, hi = mu_hat, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if mu_hat < 1e-12 or mu_hat > 1 - 1e-12 or abs(mid - mu_hat) < 1e-12 or mid > 1 - 1e-12:
            kl = 0.0
        else:
            kl = rel_entr(mu_hat, mid) + rel_entr(1 - mu_hat, 1 - mid)
        kl = max(kl, 0.0)
        if kl <= threshold:
            lo = mid
        else:
            hi = mid
    return lo


def run_ucb1():
    counts = np.zeros(K)
    rewards = np.zeros(K)
    regret = np.zeros(T)
    optimal_choices = np.zeros(T)
    for t in range(T):
        ucb = np.ones(K) * np.inf
        for i in range(K):
            if counts[i] > 0:
                mu_hat = rewards[i] / counts[i]
                ucb[i] = mu_hat + np.sqrt(2 * np.log(t + 1) / counts[i])
        arm = np.argmax(ucb)
        reward = np.random.binomial(1, TRUE_MEANS[arm])
        counts[arm] += 1
        rewards[arm] += reward
        regret[t] = BEST_MEAN - TRUE_MEANS[arm]
        optimal_choices[t] = 1 if arm == BEST_ARM else 0
    return np.cumsum(regret), optimal_choices


def run_thompson():
    alphas = np.ones(K)
    betas = np.ones(K)
    regret = np.zeros(T)
    optimal_choices = np.zeros(T)
    for t in range(T):
        samples = np.random.beta(alphas, betas)
        arm = np.argmax(samples)
        reward = np.random.binomial(1, TRUE_MEANS[arm])
        alphas[arm] += reward
        betas[arm] += 1 - reward
        regret[t] = BEST_MEAN - TRUE_MEANS[arm]
        optimal_choices[t] = 1 if arm == BEST_ARM else 0
    return np.cumsum(regret), optimal_choices


def run_kl_ucb():
    counts = np.zeros(K)
    successes = np.zeros(K)
    regret = np.zeros(T)
    optimal_choices = np.zeros(T)
    for t in range(T):
        ucb = np.ones(K) * np.inf
        for i in range(K):
            if counts[i] > 0:
                ucb[i] = kl_ucb_index(successes[i], counts[i], t + 1)
        arm = np.argmax(ucb)
        reward = np.random.binomial(1, TRUE_MEANS[arm])
        counts[arm] += 1
        successes[arm] += reward
        regret[t] = BEST_MEAN - TRUE_MEANS[arm]
        optimal_choices[t] = 1 if arm == BEST_ARM else 0
    return np.cumsum(regret), optimal_choices


def run_eps_greedy(eps):
    counts = np.zeros(K)
    rewards = np.zeros(K)
    regret = np.zeros(T)
    optimal_choices = np.zeros(T)
    for t in range(T):
        if np.random.random() < eps:
            arm = np.random.randint(K)
        else:
            mus = np.array([rewards[i] / counts[i] if counts[i] > 0 else np.inf for i in range(K)])
            arm = np.argmax(mus)
        reward = np.random.binomial(1, TRUE_MEANS[arm])
        counts[arm] += 1
        rewards[arm] += reward
        regret[t] = BEST_MEAN - TRUE_MEANS[arm]
        optimal_choices[t] = 1 if arm == BEST_ARM else 0
    return np.cumsum(regret), optimal_choices


def run_experiment(algo_fn, **kwargs):
    final_regrets = np.zeros(N_RUNS)
    opt_rates = np.zeros(N_RUNS)
    for run in range(N_RUNS):
        np.random.seed(1000 + run)
        regret, opt = algo_fn(**kwargs)
        final_regrets[run] = regret[-1]
        opt_rates[run] = opt[-1000:].mean()
    return final_regrets, opt_rates


if __name__ == "__main__":
    print("实验一：5 臂伯努利 Bandit，T=5000，50 次独立运行")
    print(f"真实均值: {TRUE_MEANS}")
    print(f"最优臂: {BEST_ARM}, 最优均值: {BEST_MEAN}")
    print()

    algorithms = [
        ("KL-UCB", run_kl_ucb),
        ("Thompson 采样", run_thompson),
        ("UCB1", run_ucb1),
        ("epsilon-greedy (eps=0.05)", lambda: run_eps_greedy(0.05)),
        ("epsilon-greedy (eps=0.10)", lambda: run_eps_greedy(0.10)),
    ]

    for name, fn in algorithms:
        print(f"运行 {name} ...")
        regs, opts = run_experiment(fn)
        print(f"  {name}: 累积遗憾 = {regs.mean():.1f} ± {regs.std():.1f}, "
              f"后1000轮最优臂选择率 = {opts.mean()*100:.1f}%")
