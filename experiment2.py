"""
实验二：EXP3 在奖励分布突变下的适应性
前 T/2=5000 轮臂 0 最优（0.9 vs 0.5），后 T/2=5000 轮臂 1 变为最优（臂 0→0.1）
"""
import numpy as np

np.random.seed(42)

K = 5
T = 10000
SHIFT = 5000
N_RUNS = 50


def true_mean_at(arm, t):
    """返回第 t 轮臂 arm 的真实期望奖励（用于遗憾计算，避免二重采样）"""
    if t < SHIFT:
        means = np.ones(K) * 0.5
        means[0] = 0.9
    else:
        means = np.ones(K) * 0.5
        means[0] = 0.1
        means[1] = 0.9
    return means[arm]


def generate_reward(arm, t):
    return np.random.binomial(1, true_mean_at(arm, t))


def best_mean(t):
    return 0.9


def run_exp3():
    eta = np.sqrt(np.log(K) / (T * K))
    gamma = min(0.5, np.sqrt(K * np.log(K) / T))
    weights = np.ones(K)
    regret = np.zeros(T)
    for t in range(T):
        p = (1 - gamma) * weights / weights.sum() + gamma / K
        arm = np.random.choice(K, p=p)
        reward = generate_reward(arm, t)
        g_hat = np.zeros(K)
        g_hat[arm] = reward / p[arm]
        weights = weights * np.exp(eta * g_hat)
        regret[t] = best_mean(t) - true_mean_at(arm, t)
    return np.cumsum(regret)


def run_ucb1():
    counts = np.zeros(K)
    rewards = np.zeros(K)
    regret = np.zeros(T)
    for t in range(T):
        ucb = np.ones(K) * np.inf
        for i in range(K):
            if counts[i] > 0:
                mu_hat = rewards[i] / counts[i]
                ucb[i] = mu_hat + np.sqrt(2 * np.log(t + 1) / counts[i])
        arm = np.argmax(ucb)
        reward = generate_reward(arm, t)
        counts[arm] += 1
        rewards[arm] += reward
        regret[t] = best_mean(t) - true_mean_at(arm, t)
    return np.cumsum(regret)


def run_discounted_ucb(gamma=0.995):
    """置信半径使用 Discount UCB 标准公式：sqrt(2 * log(1/(1-gamma)) / N_i^gamma)"""
    disc_rewards = np.zeros(K)
    disc_counts = np.zeros(K)
    regret = np.zeros(T)
    conf_radius = np.sqrt(2 * np.log(1.0 / (1.0 - gamma)))
    for t in range(T):
        ucb = np.ones(K) * np.inf
        for i in range(K):
            if disc_counts[i] > 1e-8:
                mu_hat = disc_rewards[i] / disc_counts[i]
                ucb[i] = mu_hat + conf_radius / np.sqrt(disc_counts[i])
        arm = np.argmax(ucb)
        reward = generate_reward(arm, t)
        disc_rewards = disc_rewards * gamma
        disc_counts = disc_counts * gamma
        disc_rewards[arm] += reward
        disc_counts[arm] += 1
        regret[t] = best_mean(t) - true_mean_at(arm, t)
    return np.cumsum(regret)


def run_eps_greedy(eps=0.05):
    counts = np.zeros(K)
    rewards = np.zeros(K)
    regret = np.zeros(T)
    for t in range(T):
        if np.random.random() < eps:
            arm = np.random.randint(K)
        else:
            mus = np.array([rewards[i] / counts[i] if counts[i] > 0 else np.inf for i in range(K)])
            arm = np.argmax(mus)
        reward = generate_reward(arm, t)
        counts[arm] += 1
        rewards[arm] += reward
        regret[t] = best_mean(t) - true_mean_at(arm, t)
    return np.cumsum(regret)


if __name__ == "__main__":
    print("实验二：奖励分布在 T/2=5000 处突变，K=5, 50 次运行")
    print(f"前{SHIFT}轮：臂 0 最优(0.9)，其余 0.5")
    print(f"后{SHIFT}轮：臂 1 最优(0.9)，臂 0→0.1，其余 0.5")
    print()

    algorithms = [
        ("EXP3", run_exp3),
        ("UCB1", run_ucb1),
        ("Discounted UCB (gamma=0.995)", run_discounted_ucb),
        ("epsilon-greedy (eps=0.05)", run_eps_greedy),
    ]

    for name, fn in algorithms:
        print(f"运行 {name} ...")
        before = np.zeros(N_RUNS)
        after = np.zeros(N_RUNS)
        for run in range(N_RUNS):
            np.random.seed(2000 + run)
            cum_reg = fn()
            before[run] = cum_reg[SHIFT - 1]
            after[run] = cum_reg[-1] - cum_reg[SHIFT - 1]
        print(f"  {name}: 前{SHIFT}轮遗憾 = {before.mean():.1f} ± {before.std():.1f}, "
              f"后{SHIFT}轮遗憾 = {after.mean():.1f} ± {after.std():.1f}")
