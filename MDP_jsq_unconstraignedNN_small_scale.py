import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================
# Global parameters (modifiable)
# ==========================

N = 2        # Number of VMs (can be changed later)
K = 4        # Number of containers per VM (can be changed later)
C_vm = 1     # VM queue capacity
C_cont = 1   # Container queue capacity

beta = 0.95      # Discount factor
mu_vm = 1.0      # Service rate at VM
mu_cont = 1.0    # Service rate at containers

# Simulation parameters
simulation_time = 10000
burn_in = 1000
lambda_min = 1
lambda_max = 10

# NN training weights:
w_reg = 0.2
w_cls = 1.0
w_adv = 1.0
w_margin = 0.5

# ==========================
# State space construction
# ==========================

def generate_states(N, K, C_vm, C_cont):
    """
    Generate the full global state space as tuples of occupancies.
    For capacity=1 everywhere, each position is 0 or 1, so state is a bit-vector.
    State ordering:
      (q1_vm, q1_c1, ..., q1_cK, q2_vm, q2_c1, ..., q2_cK, ..., qN_vm, qN_c1, ..., qN_cK)
    """
    per_vm_dim = 1 + K  # VM queue + K containers
    total_positions = N * per_vm_dim

    states = []
    for idx in range(2 ** total_positions):
        bits = [(idx >> b) & 1 for b in range(total_positions)]
        states.append(tuple(bits))
    state_to_index = {s: i for i, s in enumerate(states)}
    return states, state_to_index

# ==========================
# Transition and cost
# ==========================

def Lambda_tot(lambda_rate):
    """
    Total uniformization rate for arrivals.
    For now we simply use lambda_rate directly.
    """
    return lambda_rate

def departure_rate(state, N, K, mu_vm, mu_cont, C_cont=1):
    """
    Compute the total CTMC event rate leaving the state (excluding arrivals).

    Model-consistent dynamics:
      - VM buffer has NO independent completion. A buffer job only 'promotes' to a container
        at rate mu_vm *if* the buffer is occupied AND at least one container is free.
      - Each busy container completes service at rate mu_cont.

    This must match the simulation dynamics.
    """
    rate = 0.0
    per_vm_dim = 1 + K
    for i in range(N):
        base = i * per_vm_dim
        q_vm = state[base]
        q_conts = state[base + 1: base + 1 + K]

        # Promotion only if buffer occupied AND there exists a free container
        if q_vm > 0 and any(q < C_cont for q in q_conts):
            rate += mu_vm

        # Service completions from busy containers
        for q in q_conts:
            if q > 0:
                rate += mu_cont
    return rate

def next_states_probs(state, action, states, state_to_index,
                      lambda_rate, N, K, C_vm, C_cont, mu_vm, mu_cont):
    """
    Compute one-step transition probabilities from a given state for a given action
    using uniformization of the underlying CTMC.

    Events (must match simulation):
      - Arrival at rate lambda_rate: route to chosen VM; try VM buffer then local containers else block.
      - Promotion at rate mu_vm per eligible VM: eligible if buffer occupied and at least one free container.
      - Service completion at rate mu_cont per busy container.

    Action: the VM chosen for the incoming job (1..N).
    """
    per_vm_dim = 1 + K
    idx_self = state_to_index[state]
    probs = np.zeros(len(states), dtype=float)

    # Uniformization rate
    Lambda = Lambda_tot(lambda_rate)

    # Build explicit departure-event list with rates (robust when mu's differ)
    events = []  # list of (rate, kind, vm_idx, k_idx_or_None)
    for i in range(N):
        base = i * per_vm_dim
        q_vm = state[base]
        q_conts = state[base + 1: base + 1 + K]

        # Promotion event (VM -> container)
        if q_vm > 0 and any(q < C_cont for q in q_conts):
            events.append((mu_vm, "promo", i, None))

        # Container service completions
        for k in range(K):
            if state[base + 1 + k] > 0:
                events.append((mu_cont, "serv", i, k))

    dep_rate = sum(r for r, *_ in events)
    Q_rate = Lambda + dep_rate
    if Q_rate == 0.0:
        probs[idx_self] = 1.0
        return probs

    # Start with self-loop
    probs[idx_self] = 1.0

    # --------------------------
    # Arrival event
    # --------------------------
    p_arr = Lambda / Q_rate
    if p_arr > 0:
        chosen_vm_idx = action - 1
        base = chosen_vm_idx * per_vm_dim
        q_vm = state[base]
        q_conts = list(state[base + 1:base + 1 + K])

        if q_vm < C_vm:
            new_state_list = list(state)
            new_state_list[base] = q_vm + 1
            new_state = tuple(new_state_list)
            probs[state_to_index[new_state]] += p_arr
            probs[idx_self] -= p_arr
        else:
            placed = False
            for k in range(K):
                if q_conts[k] < C_cont:
                    new_state_list = list(state)
                    new_state_list[base + 1 + k] = q_conts[k] + 1
                    new_state = tuple(new_state_list)
                    probs[state_to_index[new_state]] += p_arr
                    probs[idx_self] -= p_arr
                    placed = True
                    break
            # If not placed -> blocked (self-loop)

    # --------------------------
    # Promotion + service events
    # --------------------------
    if dep_rate > 0:
        for rate, kind, i, k in events:
            p = rate / Q_rate
            new_state_list = list(state)
            base_i = i * per_vm_dim

            if kind == "promo":
                # Move one job from VM buffer to the first free container on that VM
                for kk in range(K):
                    if new_state_list[base_i + 1 + kk] < C_cont:
                        new_state_list[base_i] -= 1
                        new_state_list[base_i + 1 + kk] += 1
                        break
            else:  # "serv"
                new_state_list[base_i + 1 + k] = 0

            new_state = tuple(new_state_list)
            probs[state_to_index[new_state]] += p
            probs[idx_self] -= p

    # Numerical safety
    probs = np.maximum(probs, 0.0)
    s = probs.sum()
    if s > 0:
        probs /= s
    else:
        probs[idx_self] = 1.0
    return probs

def cost(state, action, N, K, C_vm, C_cont,
         holding_vm=1.0, holding_cont=1.0, block_cost=10.0):
    """
    One-step cost for (state, action).
    Holding cost proportional to number of jobs.
    Additional block cost if chosen VM is full at arrival.
    """
    per_vm_dim = 1 + K
    # Holding cost
    total_jobs_vm = 0
    total_jobs_cont = 0
    for i in range(N):
        base = i * per_vm_dim
        total_jobs_vm += state[base]
        total_jobs_cont += sum(state[base + 1:base + 1 + K])
    cost_holding = holding_vm * total_jobs_vm + holding_cont * total_jobs_cont

    # Blocking cost if chosen VM is full
    chosen_vm = action
    chosen_vm_idx = chosen_vm - 1
    base = chosen_vm_idx * per_vm_dim
    q_vm = state[base]
    q_conts = state[base + 1:base + 1 + K]
    block = 1 if (q_vm >= C_vm and sum(q_conts) >= K * C_cont) else 0

    return cost_holding + block_cost * block

# ==========================
# Policy iteration and advantage
# ==========================

def policy_iteration(states, state_to_index, actions,
                     lambda_rate, N, K, C_vm, C_cont,
                     beta, mu_vm, mu_cont,
                     tol=1e-8, max_iter=1000):
    """
    Policy iteration to compute optimal policy and value function
    for a given lambda_rate.
    Returns:
        policy (array of ints),
        V_opt (value vector),
        Q (dict: action -> Q-values array)
    """
    n_states = len(states)

    # Precompute transition matrices and one-step cost for each action
    P = {a: np.zeros((n_states, n_states)) for a in actions}
    c = {a: np.zeros(n_states) for a in actions}

    for i, s in enumerate(states):
        for a in actions:
            P[a][i, :] = next_states_probs(s, a, states, state_to_index,
                                           lambda_rate, N, K, C_vm, C_cont,
                                           mu_vm, mu_cont)
            c[a][i] = cost(s, a, N, K, C_vm, C_cont)

    # Initialize policy (e.g., always choose action 1)
    policy = np.ones(n_states, dtype=int)

    # Policy iteration
    policy_stable = False
    V = np.zeros(n_states)
    it = 0
    while not policy_stable and it < max_iter:
        it += 1
        # Policy evaluation by iterative method
        while True:
            V_new = np.zeros(n_states)
            for i, s in enumerate(states):
                a = policy[i]
                V_new[i] = c[a][i] + beta * P[a][i, :].dot(V)
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new

        # Policy improvement
        policy_stable = True
        for i, s in enumerate(states):
            action_values = []
            for a in actions:
                q_val = c[a][i] + beta * P[a][i, :].dot(V)
                action_values.append(q_val)
            best_a = actions[int(np.argmin(action_values))]
            if best_a != policy[i]:
                policy_stable = False
                policy[i] = best_a

    V_opt = V.copy()

    # Compute Q-values under optimal policy
    Q = {a: np.zeros(n_states) for a in actions}
    for i, s in enumerate(states):
        for a in actions:
            Q[a][i] = c[a][i] + beta * P[a][i, :].dot(V_opt)

    return policy, V_opt, Q

def build_advantage_dataframe(states, state_to_index, policy, Q, N, K):
    """
    Build a DataFrame with:
    - state components
    - per-VM local backlogs (for VM1 and VM2)
    - total backlog S_tot
    - optimal action vs JSQ action
    - Q-values and advantage between actions
    """
    per_vm_dim = 1 + K
    actions = sorted(Q.keys())
    # For now we assume actions = {1,2} and analyze first two VMs
    assert actions == [1, 2], "This helper assumes actions {1,2} for two VMs."

    rows = []
    for i, s in enumerate(states):
        # VM1
        q1_vm = s[0]
        q1_conts = s[1:1 + K]
        total1 = q1_vm + sum(q1_conts)
        # VM2
        offset2 = per_vm_dim
        q2_vm = s[offset2]
        q2_conts = s[offset2 + 1:offset2 + 1 + K]
        total2 = q2_vm + sum(q2_conts)

        S_tot = total1 + total2  # we ignore extra VMs when computing S_tot here

        # JSQ decision
        if total1 < total2:
            jsq_action = 1
        elif total2 < total1:
            jsq_action = 2
        else:
            jsq_action = 1  # tie-breaking

        rows.append({
            "state": s,
            "q1_vm": q1_vm,
            "q1_c1": q1_conts[0] if K > 0 else 0,
            "q1_c2": q1_conts[1] if K > 1 else 0,
            "q2_vm": q2_vm,
            "q2_c1": q2_conts[0] if K > 0 else 0,
            "q2_c2": q2_conts[1] if K > 1 else 0,
            "total1": total1,
            "total2": total2,
            "S_tot": S_tot,
            "opt_action": policy[i],
            "jsq_action": jsq_action,
            "Q1": Q[1][i],
            "Q2": Q[2][i],
            "adv_1_vs_2": Q[1][i] - Q[2][i]
        })

    df_adv = pd.DataFrame(rows)
    df_adv["jsq_agree"] = (df_adv["opt_action"] == df_adv["jsq_action"])
    return df_adv

def compute_jsq_agreement(df_adv):
    """
    Compute R(S) = P(opt_action == JSQ_action | S_tot = S)
    from the advantage DataFrame.
    We no longer derive tau here; tau is computed globally over lambda.
    """
    R = df_adv.groupby("S_tot")["jsq_agree"].mean().reset_index()
    R.columns = ["S_tot", "R_S"]
    return R


# ==========================
# Neural network (teacher-forced) for toy MDP
# ==========================

def build_multi_lambda_dataset(lambda_list,
                               state_weights_by_lambda=None,  # dict: lambda -> w_states (len(states),)
                               N=N, K=K, C_vm=C_vm, C_cont=C_cont,
                               beta=beta, mu_vm=mu_vm, mu_cont=mu_cont):
    """
    Build a joint dataset over several lambda values.

    Features:
      [state_bits..., lambda_norm, total1_norm, total2_norm, S_norm]
    Targets:
      teacher Q-values (from DP): Y = [Q1, Q2]
    Weights:
      if state_weights_by_lambda is provided, use those; otherwise use teacher stationary weights.
    """
    states, state_to_index = generate_states(N, K, C_vm, C_cont)

    per_vm_dim = 1 + K
    cap_per_vm = C_vm + K * C_cont
    cap_total = N * cap_per_vm
    lambda_max_local = max(lambda_list)

    all_X, all_Y, all_opt_actions, all_lambda, all_weights = [], [], [], [], []

    for lambda_rate in lambda_list:
        actions = [1, 2]

        # DP teacher
        policy_opt, V_opt, Q = policy_iteration(
            states, state_to_index, actions,
            lambda_rate, N, K, C_vm, C_cont,
            beta, mu_vm, mu_cont
        )

        # Choose weights
        if state_weights_by_lambda is not None and lambda_rate in state_weights_by_lambda:
            w_states = np.array(state_weights_by_lambda[lambda_rate], dtype=float)
            # normalize to mean 1 for stability (same convention as before)
            w_states = w_states / (w_states.mean() if w_states.mean() > 0 else 1.0)
        else:
            # fallback to teacher stationary weights
            policy_dict_teacher = {states[i]: policy_opt[i] for i in range(len(states))}
            w_states = estimate_state_weights_for_lambda(
                lambda_rate, policy_dict_teacher, states, state_to_index,
                N, K, C_vm, C_cont, mu_vm, mu_cont,
                simulation_time=simulation_time, burn_in=burn_in
            )

        lambda_norm = lambda_rate / lambda_max_local

        for i, s in enumerate(states):
            state_bits = list(s)

            # VM totals and S
            base1 = 0
            q1_vm = s[base1]
            q1_conts = s[base1 + 1: base1 + 1 + K]
            total1 = q1_vm + sum(q1_conts)

            base2 = per_vm_dim
            q2_vm = s[base2]
            q2_conts = s[base2 + 1: base2 + 1 + K]
            total2 = q2_vm + sum(q2_conts)

            S_tot = total1 + total2

            total1_norm = total1 / cap_per_vm
            total2_norm = total2 / cap_per_vm
            S_norm = S_tot / cap_total

            feat = state_bits + [lambda_norm, total1_norm, total2_norm, S_norm]

            all_X.append(feat)
            all_Y.append([Q[1][i], Q[2][i]])
            all_opt_actions.append(policy_opt[i])
            all_lambda.append(lambda_rate)
            all_weights.append(w_states[i])

    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.float32)
    opt_actions = np.array(all_opt_actions, dtype=np.int64)
    lambda_arr = np.array(all_lambda, dtype=np.int64)
    sample_weights = np.array(all_weights, dtype=np.float32)

    return X, Y, lambda_arr, states, opt_actions, sample_weights





class MonotoneQNet(nn.Module):
    """
    Feedforward Q-network (unconstrained weights).
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(MonotoneQNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 2)

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x


def clamp_nonnegative_weights(model):
    """No-op (kept for backward compatibility)."""
    return


def train_multi_lambda_nn(lambda_list,
                          hidden_dim=64,
                          num_epochs=500,
                          lr=5e-3,
                          w_reg=0.5,
                          w_cls=1.0,
                          verbose=True):
    """
    Train a single MonotoneQNet on multiple lambda values.

    Loss = w_reg * (weighted MSE on Q-values)
         + w_cls * (weighted cross-entropy on optimal action),
    where sample weights come from the approximate stationary distribution.
    """
    (X_np, Y_np, lambda_np,
     states, opt_actions_np, sample_weights_np) = build_multi_lambda_dataset(lambda_list)

    n_samples, input_dim = X_np.shape

    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)
    actions_t = torch.from_numpy(opt_actions_np - 1)   # {1,2} -> {0,1}
    weights_t = torch.from_numpy(sample_weights_np)

    model = MonotoneQNet(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mse = nn.MSELoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X)                         # (n_samples, 2)

        # ----- Advantage-based losses -----
        A_hat = pred[:, 0] - pred[:, 1]
        A_teacher = Y[:, 0] - Y[:, 1]   # since Y stores [Q1,Q2]

        # 1) Robust regression on advantage (Huber / SmoothL1)
        huber = nn.SmoothL1Loss(reduction="none")
        adv_reg_per_sample = huber(A_hat, A_teacher)
        loss_adv_reg = (weights_t * adv_reg_per_sample).mean()

        # 2) Margin loss on the sign (prevents flips near A=0)
        # y = -1 if optimal action is 1, +1 if optimal action is 2
        y = torch.where(actions_t == 0, -torch.ones_like(A_hat), torch.ones_like(A_hat))
        margin = 0.25  # start small; tune 0.1–1.0
        margin_per_sample = torch.relu(margin - y * A_hat)
        loss_margin = (weights_t * margin_per_sample).mean()


        # regression loss per sample
        mse_per_sample = mse(pred, Y).mean(dim=1)     # (n_samples,)
        loss_reg = (weights_t * mse_per_sample).mean()

        # classification loss per sample on optimal action
        logits = -pred                             # lower Q = better action
        ce_per_sample = ce(logits, actions_t)      # (n_samples,)
        loss_cls = (weights_t * ce_per_sample).mean()

        loss = w_reg * loss_reg + w_cls * loss_cls + w_adv * loss_adv_reg + w_margin * loss_margin
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            with torch.no_grad():
                pred_full = model(X)
                a_hat = torch.argmin(pred_full, dim=1)  # {0,1}
                acc = (a_hat == actions_t).float().mean().item()
                total_loss = loss.item()
            print(f"[multi-lambda] Epoch {epoch+1}/{num_epochs} "
                  f"Loss={total_loss:.6f}, Train accuracy={acc:.4f}")

    return model, states


def train_multi_lambda_nn_from_dataset(X_np, Y_np, opt_actions_np, sample_weights_np,
                                       hidden_dim=64,
                                       num_epochs=300,
                                       lr=2e-3,
                                       w_reg=0.5,
                                       w_cls=1.0,
                                       verbose=True,
                                       init_model=None):
    """
    Same loss as train_multi_lambda_nn, but takes a prepared dataset.
    If init_model is provided, fine-tunes it (keeps weights); else trains from scratch.
    """
    n_samples, input_dim = X_np.shape

    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)
    actions_t = torch.from_numpy(opt_actions_np - 1)   # {1,2}->{0,1}
    weights_t = torch.from_numpy(sample_weights_np)

    model = init_model if init_model is not None else MonotoneQNet(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mse = nn.MSELoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X)

        # ----- Advantage-based losses -----
        A_hat = pred[:, 0] - pred[:, 1]
        A_teacher = Y[:, 0] - Y[:, 1]   # since Y stores [Q1,Q2]

        # 1) Robust regression on advantage (Huber / SmoothL1)
        huber = nn.SmoothL1Loss(reduction="none")
        adv_reg_per_sample = huber(A_hat, A_teacher)
        loss_adv_reg = (weights_t * adv_reg_per_sample).mean()

        # 2) Margin loss on the sign (prevents flips near A=0)
        # y = -1 if optimal action is 1, +1 if optimal action is 2
        y = torch.where(actions_t == 0, -torch.ones_like(A_hat), torch.ones_like(A_hat))
        margin = 0.25  # start small; tune 0.1–1.0
        margin_per_sample = torch.relu(margin - y * A_hat)
        loss_margin = (weights_t * margin_per_sample).mean()


        mse_per_sample = mse(pred, Y).mean(dim=1)
        loss_reg = (weights_t * mse_per_sample).mean()

        logits = -pred
        ce_per_sample = ce(logits, actions_t)
        loss_cls = (weights_t * ce_per_sample).mean()

        loss = w_reg * loss_reg + w_cls * loss_cls + w_adv * loss_adv_reg + w_margin * loss_margin
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            with torch.no_grad():
                a_hat = torch.argmin(model(X), dim=1)
                acc = (a_hat == actions_t).float().mean().item()
            print(f"[dagger-ft] Epoch {epoch+1}/{num_epochs} Loss={loss.item():.6f}, Train acc={acc:.4f}")

    return model




def build_nn_policy_for_lambda_multi(model, states, lambda_rate, lambda_max):
    """
    Use a multi-lambda model to build policy for a specific lambda_rate.
    lambda_max is the max of the lambda_list used in training (for normalization).

    IMPORTANT: feature construction must match build_multi_lambda_dataset:
        feat = state_bits + [lambda_norm, total1_norm, total2_norm, S_norm]
    """
    lambda_norm = lambda_rate / lambda_max

    per_vm_dim = 1 + K
    cap_per_vm = C_vm + K * C_cont
    cap_total = N * cap_per_vm

    X_list = []
    for s in states:
        state_bits = list(s)

        # --- per-VM totals and global backlog S ---
        # VM1
        base1 = 0
        q1_vm = s[base1]
        q1_conts = s[base1 + 1: base1 + 1 + K]
        total1 = q1_vm + sum(q1_conts)

        # VM2
        base2 = per_vm_dim
        q2_vm = s[base2]
        q2_conts = s[base2 + 1: base2 + 1 + K]
        total2 = q2_vm + sum(q2_conts)

        S_tot = total1 + total2

        total1_norm = total1 / cap_per_vm
        total2_norm = total2 / cap_per_vm
        S_norm = S_tot / cap_total

        feat = state_bits + [lambda_norm, total1_norm, total2_norm, S_norm]
        X_list.append(feat)

    X_np = np.array(X_list, dtype=np.float32)
    X = torch.from_numpy(X_np)

    with torch.no_grad():
        q = model(X)  # (num_states, 2)

    q_np = q.numpy()
    actions = np.argmin(q_np, axis=1) + 1  # {0,1} -> {1,2}

    policy_dict = {s: int(a) for s, a in zip(states, actions)}
    return policy_dict


def nn_induced_weights_for_lambda(model_nn, states, state_to_index, lambda_rate, lambda_max,
                                 N, K, C_vm, C_cont, mu_vm, mu_cont):
    nn_policy_dict = build_nn_policy_for_lambda_multi(model_nn, states, lambda_rate, lambda_max)
    w_states_nn = estimate_state_weights_for_lambda(
        lambda_rate, nn_policy_dict, states, state_to_index,
        N, K, C_vm, C_cont, mu_vm, mu_cont,
        simulation_time=simulation_time, burn_in=burn_in
    )
    return w_states_nn



# ==========================
# JSQ simulation
# ==========================

def policy(state):
    """
    JSQ policy for simulation: choose VM with smallest backlog.
    Uses global N and K. Assumes state is ordered as:
      (q1_vm, q1_c1, ..., q1_cK, q2_vm, q2_c1, ..., q2_cK)
    """
    per_vm_dim = 1 + K
    assert N == 2, "policy() currently implemented for N=2 only."

    # VM1
    base1 = 0
    q1_vm = state[base1]
    q1_conts = state[base1 + 1: base1 + 1 + K]
    total1 = q1_vm + sum(q1_conts)

    # VM2
    base2 = per_vm_dim
    q2_vm = state[base2]
    q2_conts = state[base2 + 1: base2 + 1 + K]
    total2 = q2_vm + sum(q2_conts)

    # JSQ with tie-breaking in favour of VM1
    return 1 if total1 <= total2 else 2


def simulate_performance(lambda_rate, N, K, C_vm, C_cont, mu_vm, mu_cont,
                         simulation_time=simulation_time, burn_in=burn_in):
    """
    Continuous-time event-based simulation using JSQ policy for N=2 and arbitrary K.

    State structure:
      For each VM i (i=0,1):
        state[base]     = q_vm (VM waiting buffer, 0 or 1)
        state[base+1: ] = K container occupancies (0/1 each)
      with base = i * (1 + K).

    Events:
      - Arrival at rate lambda_rate:
          choose VM by JSQ, try VM queue, else containers on that VM, else block.
      - Promotion (VM -> container) at rate mu_vm per eligible VM:
          eligible if VM queue occupied and at least one free container on that VM.
      - Service completion at rate mu_cont per busy container.
    """
    assert N == 2, "simulate_performance currently implemented for N=2 only."

    per_vm_dim = 1 + K
    # State initialization: all empty
    state = [0] * (N * per_vm_dim)
    t = 0.0

    blocked_jobs = 0
    arrivals = 0
    queue_lengths = []

    while t < simulation_time:
        # ---- Compute event rates ----
        lambda_arr = lambda_rate

        # Promotion rates (VM -> containers)
        promo_vms = []  # list of VM indices (0 or 1) that can promote
        for i in range(N):
            base = i * per_vm_dim
            q_vm = state[base]
            conts = state[base + 1: base + 1 + K]
            if q_vm > 0 and any(c < C_cont for c in conts):
                promo_vms.append(i)
        promo_rate = mu_vm * len(promo_vms)

        # Service rates (containers)
        busy_containers = []  # list of (vm_index, k_index)
        for i in range(N):
            base = i * per_vm_dim
            for k in range(K):
                if state[base + 1 + k] > 0:
                    busy_containers.append((i, k))
        service_rate = mu_cont * len(busy_containers)

        # Total rate
        R = lambda_arr + promo_rate + service_rate
        if R == 0.0:
            break

        # ---- Sample next event ----
        dt = np.random.exponential(1.0 / R)
        t += dt

        u = np.random.rand()
        if u < lambda_arr / R:
            event = "arrival"
        elif u < (lambda_arr + promo_rate) / R:
            event = "promotion"
        else:
            event = "service"

        # ---- Execute event ----
        if event == "arrival":
            # JSQ policy: action 1 or 2
            a = policy(state)
            vm_idx = a - 1
            base = vm_idx * per_vm_dim

            arrivals += 1

            # Try VM queue first
            if state[base] < C_vm:
                state[base] += 1
            else:
                # Try containers on that VM
                placed = False
                for k in range(K):
                    if state[base + 1 + k] < C_cont:
                        state[base + 1 + k] += 1
                        placed = True
                        break
                if not placed:
                    blocked_jobs += 1

        elif event == "promotion":
            if promo_vms:
                vm_idx = random.choice(promo_vms)
                base = vm_idx * per_vm_dim

                if state[base] > 0:
                    # Move from VM queue to first free container
                    for k in range(K):
                        if state[base + 1 + k] < C_cont:
                            state[base] -= 1
                            state[base + 1 + k] += 1
                            break

        else:  # event == "service"
            if busy_containers:
                vm_idx, k = random.choice(busy_containers)
                base = vm_idx * per_vm_dim
                state[base + 1 + k] = 0

        # ---- Statistics after burn-in ----
        if t > burn_in:
            queue_lengths.append(sum(state))

    blocking_prob = blocked_jobs / arrivals if arrivals > 0 else 0.0
    avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0.0
    return blocking_prob, avg_queue_length



# ==========================
# MDP Simulation
# ==========================

def simulate_performance_mdp(lambda_rate, policy_dict, N, K, C_vm, C_cont, mu_vm, mu_cont,
                             simulation_time=simulation_time, burn_in=burn_in):
    """
    Continuous-time simulation using the MDP optimal policy instead of JSQ.
    policy_dict maps full state tuples
      (q1_vm, q1_c1, ..., q1_cK, q2_vm, q2_c1, ..., q2_cK)
    to actions {1,2}. Currently implemented for N=2 and arbitrary K.
    """
    assert N == 2, "simulate_performance_mdp currently implemented for N=2 only."

    per_vm_dim = 1 + K
    state = [0] * (N * per_vm_dim)
    t = 0.0

    blocked_jobs = 0
    arrivals = 0
    queue_lengths = []

    while t < simulation_time:
        lambda_arr = lambda_rate

        # Promotion
        promo_vms = []
        for i in range(N):
            base = i * per_vm_dim
            q_vm = state[base]
            conts = state[base + 1: base + 1 + K]
            if q_vm > 0 and any(c < C_cont for c in conts):
                promo_vms.append(i)
        promo_rate = mu_vm * len(promo_vms)

        # Service
        busy_containers = []
        for i in range(N):
            base = i * per_vm_dim
            for k in range(K):
                if state[base + 1 + k] > 0:
                    busy_containers.append((i, k))
        service_rate = mu_cont * len(busy_containers)

        R = lambda_arr + promo_rate + service_rate
        if R == 0.0:
            break

        dt = np.random.exponential(1.0 / R)
        t += dt
        u = np.random.rand()

        if u < lambda_arr / R:
            event = "arrival"
        elif u < (lambda_arr + promo_rate) / R:
            event = "promotion"
        else:
            event = "service"

        if event == "arrival":
            arrivals += 1
            st = tuple(state)
            a = policy_dict.get(st, 1)  # default to 1 if missing
            vm_idx = a - 1
            base = vm_idx * per_vm_dim

            if state[base] < C_vm:
                state[base] += 1
            else:
                placed = False
                for k in range(K):
                    if state[base + 1 + k] < C_cont:
                        state[base + 1 + k] += 1
                        placed = True
                        break
                if not placed:
                    blocked_jobs += 1

        elif event == "promotion":
            if promo_vms:
                vm_idx = random.choice(promo_vms)
                base = vm_idx * per_vm_dim
                if state[base] > 0:
                    for k in range(K):
                        if state[base + 1 + k] < C_cont:
                            state[base] -= 1
                            state[base + 1 + k] += 1
                            break

        else:  # service
            if busy_containers:
                vm_idx, k = random.choice(busy_containers)
                base = vm_idx * per_vm_dim
                state[base + 1 + k] = 0

        if t > burn_in:
            queue_lengths.append(sum(state))

    blocking_prob = blocked_jobs / arrivals if arrivals > 0 else 0.0
    avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0.0
    return blocking_prob, avg_queue_length


# ==========================
# Stationary weights under optimal MDP policy
# ==========================

def estimate_state_weights_for_lambda(lambda_rate,
                                      policy_dict,
                                      states,
                                      state_to_index,
                                      N, K, C_vm, C_cont, mu_vm, mu_cont,
                                      simulation_time=simulation_time,
                                      burn_in=burn_in):
    """
    Approximate the stationary distribution over states under the given policy_dict
    for a fixed lambda_rate, by simulation.

    Returns:
        weights: np.array of shape (num_states,), non-negative, normalized so that
                 weights.mean() = 1.0 (for numerical stability).
    """
    assert N == 2, "estimate_state_weights_for_lambda currently implemented for N=2 only."

    num_states = len(states)
    visit_counts = np.zeros(num_states, dtype=float)

    per_vm_dim = 1 + K
    state = [0] * (N * per_vm_dim)
    t = 0.0

    while t < simulation_time:
        lambda_arr = lambda_rate

        # Promotion
        promo_vms = []
        for i in range(N):
            base = i * per_vm_dim
            q_vm = state[base]
            conts = state[base + 1: base + 1 + K]
            if q_vm > 0 and any(c < C_cont for c in conts):
                promo_vms.append(i)
        promo_rate = mu_vm * len(promo_vms)

        # Service
        busy_containers = []
        for i in range(N):
            base = i * per_vm_dim
            for k in range(K):
                if state[base + 1 + k] > 0:
                    busy_containers.append((i, k))
        service_rate = mu_cont * len(busy_containers)

        R = lambda_arr + promo_rate + service_rate
        if R == 0.0:
            break

        dt = np.random.exponential(1.0 / R)
        t += dt
        u = np.random.rand()

        if u < lambda_arr / R:
            event = "arrival"
        elif u < (lambda_arr + promo_rate) / R:
            event = "promotion"
        else:
            event = "service"

        if event == "arrival":
            st = tuple(state)
            a = policy_dict.get(st, 1)
            vm_idx = a - 1
            base = vm_idx * per_vm_dim

            if state[base] < C_vm:
                state[base] += 1
            else:
                placed = False
                for k in range(K):
                    if state[base + 1 + k] < C_cont:
                        state[base + 1 + k] += 1
                        placed = True
                        break
                # blocked arrivals don't change state, so nothing else to do

        elif event == "promotion":
            if promo_vms:
                vm_idx = random.choice(promo_vms)
                base = vm_idx * per_vm_dim
                if state[base] > 0:
                    for k in range(K):
                        if state[base + 1 + k] < C_cont:
                            state[base] -= 1
                            state[base + 1 + k] += 1
                            break

        else:  # service
            if busy_containers:
                vm_idx, k = random.choice(busy_containers)
                base = vm_idx * per_vm_dim
                state[base + 1 + k] = 0

        # After burn-in, accumulate visits
        if t > burn_in:
            idx = state_to_index[tuple(state)]
            # You can use dt here instead of 1.0; both are fine for weights.
            visit_counts[idx] += dt

    if visit_counts.sum() == 0:
        # fallback: uniform weights
        weights = np.ones(num_states, dtype=float)
    else:
        weights = visit_counts / visit_counts.sum()
        # rescale so that mean weight ~ 1 (keeps losses numerically stable)
        weights = weights / weights.mean()

    return weights




# ==========================
# Main driver: loop over lambda
# ==========================

def run_for_lambda_sweep():
    """
    For lambda in [lambda_min..lambda_max]:
      - run policy iteration
      - compute advantage dataframe and JSQ agreement per state
      - run JSQ simulation for comparison

    At the end:
      - concatenate all per-lambda data into one CSV
      - compute A(lambda) = mean jsq_agree per lambda
      - compute a single global tau (lambda threshold) for a given gamma.
    """

    lambda_values = list(range(lambda_min, lambda_max+1))
    states, state_to_index = generate_states(N, K, C_vm, C_cont)

    # ----- DAgger-style training loop -----
    dagger_iters = 3          # 2–5 is typical
    mix_teacher = 0.3         # keep some teacher distribution to avoid collapse

    print("Training initial multi-lambda NN (teacher stationary weights)...")
    model_nn, _ = train_multi_lambda_nn(lambda_values,
                                        hidden_dim=64,
                                        num_epochs=500,
                                        lr=5e-3,
                                        w_reg=0.5,
                                        w_cls=1.0,
                                        verbose=True)
    print("Initial model trained.")

    for it in range(dagger_iters):
        print(f"\n==== DAgger iteration {it+1}/{dagger_iters} ====")

        # 1) build weights under NN-induced trajectories (per lambda)
        nn_weights_by_lambda = {}
        teacher_weights_by_lambda = {}

        for lambda_rate in lambda_values:
            # teacher policy + teacher weights (for mixing)
            policy_opt, _, _ = policy_iteration(
                states, state_to_index, [1, 2],
                lambda_rate, N, K, C_vm, C_cont,
                beta, mu_vm, mu_cont
            )
            policy_teacher = {states[i]: policy_opt[i] for i in range(len(states))}
            w_teacher = estimate_state_weights_for_lambda(
                lambda_rate, policy_teacher, states, state_to_index,
                N, K, C_vm, C_cont, mu_vm, mu_cont,
                simulation_time=simulation_time, burn_in=burn_in
            )
            teacher_weights_by_lambda[lambda_rate] = w_teacher

            # NN-induced weights
            w_nn = nn_induced_weights_for_lambda(
                model_nn, states, state_to_index,
                lambda_rate, lambda_max, N, K, C_vm, C_cont, mu_vm, mu_cont
            )
            nn_weights_by_lambda[lambda_rate] = w_nn

        # 2) mix weights: (1-mix_teacher)*NN + mix_teacher*teacher
        mixed_weights_by_lambda = {}
        for lambda_rate in lambda_values:
            w_mix = (1.0 - mix_teacher) * nn_weights_by_lambda[lambda_rate] + mix_teacher * teacher_weights_by_lambda[lambda_rate]
            # normalize to mean 1
            w_mix = w_mix / (w_mix.mean() if w_mix.mean() > 0 else 1.0)
            mixed_weights_by_lambda[lambda_rate] = w_mix

        # 3) rebuild full dataset with the mixed weights, but still teacher labels (DP Q, optimal action)
        X_np, Y_np, lambda_np, states2, opt_actions_np, sample_weights_np = build_multi_lambda_dataset(
            lambda_values, state_weights_by_lambda=mixed_weights_by_lambda
        )

        # 4) fine-tune the existing model on this dataset (don’t restart from scratch)
        model_nn = train_multi_lambda_nn_from_dataset(
            X_np, Y_np, opt_actions_np, sample_weights_np,
            hidden_dim=64, num_epochs=300, lr=2e-3,
            w_reg=0.5, w_cls=1.0, verbose=True,
            init_model=model_nn
        )

    print("\nDone DAgger training.\n")



    states, state_to_index = generate_states(N, K, C_vm, C_cont)
    actions = [1, 2]  # dispatch to VM1 or VM2

    all_adv_dfs = []
    sim_results = []

    for lambda_rate in range(lambda_min, lambda_max + 1):
        print(f"\n=== Lambda = {lambda_rate} ===")
        policy_opt, V_opt, Q = policy_iteration(
            states, state_to_index, actions,
            lambda_rate, N, K, C_vm, C_cont,
            beta, mu_vm, mu_cont
        )

        df_adv = build_advantage_dataframe(states, state_to_index,
                                           policy_opt, Q, N, K)
        df_adv["lambda"] = lambda_rate

        # Optional: R(S) for analysis
        R = compute_jsq_agreement(df_adv)
        print("Mean JSQ agreement for lambda", lambda_rate, "=",
              df_adv["jsq_agree"].mean())

        all_adv_dfs.append(df_adv)

        # Build mapping state -> optimal action for MDP policy
        policy_dict = {states[i]: policy_opt[i] for i in range(len(states))}

        # JSQ simulation for this lambda (toy)
        block_prob_jsq, avg_q_jsq = simulate_performance(
            lambda_rate, N, K, C_vm, C_cont, mu_vm, mu_cont
        )

        # MDP-based simulation for this lambda
        block_prob_mdp, avg_q_mdp = simulate_performance_mdp(
            lambda_rate, policy_dict, N, K, C_vm, C_cont, mu_vm, mu_cont
        )

        # ==========================
        # NN-based policy: train on teacher Q-values for this lambda
        # ==========================

        # Build NN policy for this lambda using multi-lambda model
        nn_policy_dict = build_nn_policy_for_lambda_multi(
            model_nn, states, lambda_rate, lambda_max
        )

        # agreement print 
        nn_actions = np.array([nn_policy_dict[s] for s in states])
        opt_actions = np.array(policy_opt)
        nn_agreement = np.mean(nn_actions == opt_actions)
        print(f"NN agreement with optimal policy for lambda {lambda_rate}: {nn_agreement:.4f}")

        w_nn = nn_induced_weights_for_lambda(model_nn, states, state_to_index,
                                             lambda_rate, lambda_max,
                                             N, K, C_vm, C_cont, mu_vm, mu_cont)
        w_nn = w_nn / w_nn.sum()
        weighted_agreement = np.sum(w_nn * (nn_actions == opt_actions))
        print(f"NN weighted agreement (NN-stationary) for lambda {lambda_rate}: {weighted_agreement:.4f}")


        block_prob_nn, avg_q_nn = simulate_performance_mdp(
            lambda_rate, nn_policy_dict, N, K, C_vm, C_cont, mu_vm, mu_cont
        )



        sim_results.append({
            "lambda": lambda_rate,
            "blocking_prob_jsq": block_prob_jsq,
            "avg_queue_length_jsq": avg_q_jsq,
            "blocking_prob_mdp": block_prob_mdp,
            "avg_queue_length_mdp": avg_q_mdp,
            "blocking_prob_nn": block_prob_nn,
            "avg_queue_length_nn": avg_q_nn
        })



    # Concatenate all lambda results into a single DataFrame
    df_all = pd.concat(all_adv_dfs, ignore_index=True)

    # Save ALL experiments in ONE CSV (per state, per lambda)
    df_all.to_csv("advantage_jsq_data_all_lambda.csv", index=False)

    # JSQ simulation summary per lambda
    df_sim = pd.DataFrame(sim_results)
    df_sim.to_csv("jsq_simulation_results.csv", index=False)
    print("\nJSQ simulation summary:\n", df_sim)

    # Compute A(lambda) = mean jsq_agree per lambda
    jsq_agreement_by_lambda = df_all.groupby("lambda")["jsq_agree"].mean().reset_index()
    jsq_agreement_by_lambda.columns = ["lambda", "A_lambda"]
    jsq_agreement_by_lambda.to_csv("jsq_agreement_by_lambda.csv", index=False)
    print("\nJSQ agreement by lambda:\n", jsq_agreement_by_lambda)

    
    return df_all, jsq_agreement_by_lambda, df_sim

def main():
    np.random.seed(0)
    random.seed(0)

    df_all, jsq_agreement_by_lambda, df_sim = run_for_lambda_sweep()
    

if __name__ == "__main__":
    main()
