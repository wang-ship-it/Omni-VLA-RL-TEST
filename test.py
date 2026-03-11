import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# =========================
# 超简单环境：猜 0 或 1
# =========================
class GuessEnv:

    def reset(self):
        self.target = random.randint(0, 1)
        return torch.tensor([self.target], dtype=torch.float32)

    def step(self, action):
        reward = 1.0 if action == self.target else 0.0
        return reward


# =========================
# Policy 网络
# =========================
class PolicyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 采样 rollout
# =========================
def rollout(policy, env, n):

    states = []
    actions = []
    rewards = []
    logps = []

    for _ in range(n):

        s = env.reset()

        logits = policy(s)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)

        a = dist.sample()
        logp = dist.log_prob(a)

        r = env.step(a.item())

        states.append(s)
        actions.append(a)
        rewards.append(r)
        logps.append(logp)

    return states, actions, rewards, logps


# =========================
# PPO
# =========================
def train_ppo():

    print("\n===== PPO =====")

    env = GuessEnv()
    policy = PolicyNet()
    opt = optim.Adam(policy.parameters(), lr=3e-3)

    for epoch in range(100):

        states, actions, rewards, logps_old = rollout(
            policy, env, 64
        )

        rewards = torch.tensor(rewards)

        # baseline = mean
        adv = rewards - rewards.mean()

        loss = 0

        for s, a, A, old_logp in zip(
            states, actions, adv, logps_old
        ):

            logits = policy(s)
            probs = F.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            logp = dist.log_prob(a)

            ratio = torch.exp(logp - old_logp.detach())

            clip = torch.clamp(ratio, 0.8, 1.2)

            loss -= torch.min(ratio * A, clip * A)

        loss /= len(states)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | "
                f"Reward {rewards.mean():.3f}"
            )


# =========================
# GRPO
# =========================
def train_grpo():

    print("\n===== GRPO =====")

    env = GuessEnv()
    policy = PolicyNet()
    opt = optim.Adam(policy.parameters(), lr=3e-3)

    GROUP = 8
    BATCH = 64

    for epoch in range(100):

        all_loss = 0
        all_adv = []
        all_rewards = []

        for _ in range(BATCH // GROUP):

            states, actions, rewards, logps_old = rollout(
                policy, env, GROUP
            )

            rewards = torch.tensor(rewards)
            baseline = rewards.mean()

            adv = rewards - baseline

            all_rewards.extend(rewards.tolist())

            for s, a, A, old_logp in zip(
                states, actions, adv, logps_old
            ):

                logits = policy(s)
                probs = F.softmax(logits, dim=-1)

                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(a)

                ratio = torch.exp(logp - old_logp.detach())

                clip = torch.clamp(ratio, 0.8, 1.2)

                all_loss -= torch.min(
                    ratio * A, clip * A
                )

                all_adv.append(A.item())

        loss = all_loss / BATCH

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | "
                f"Reward {sum(all_rewards)/len(all_rewards):.3f} | "
                f"MeanAdv {sum(all_adv)/len(all_adv):.3f} | "
                f"MeanAdvAbs {sum(abs(a) for a in all_adv)/len(all_adv):.3f}"
            )


# =========================
# GSPO（序列版，这里用 episode 模拟）
# =========================
def train_gspo():

    print("\n===== GSPO =====")

    env = GuessEnv()
    policy = PolicyNet()
    opt = optim.Adam(policy.parameters(), lr=3e-3)

    EPISODE = 5
    BATCH = 64

    for epoch in range(100):

        traj_logps = []
        traj_rewards = []

        for _ in range(BATCH):

            logp_sum = 0
            r_sum = 0

            for _ in range(EPISODE):

                s = env.reset()

                logits = policy(s)
                probs = F.softmax(logits, dim=-1)

                dist = torch.distributions.Categorical(probs)

                a = dist.sample()
                logp = dist.log_prob(a)

                r = env.step(a.item())

                logp_sum += logp
                r_sum += r

            traj_logps.append(logp_sum)
            traj_rewards.append(r_sum)

        rewards = torch.tensor(traj_rewards)

        baseline = rewards.mean()
        adv = rewards - baseline

        loss = 0

        for logp, A in zip(traj_logps, adv):

            loss -= logp * A

        loss /= BATCH

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | "
                f"TrajReward {rewards.mean():.3f}"
            )


# =========================
# 主入口
# =========================
if __name__ == "__main__":

    train_ppo()
    train_grpo()
    train_gspo()
