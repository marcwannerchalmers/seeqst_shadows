from jaxed.tools.observable import PauliObservable
from jaxed.tools.majorana import (
    FermionicGaussianShadow,
    MajoranaState,
    majorana_observables_list,
)
from jaxed.tools.state import HRState, GHZType, State, noisy_version
from jaxed.tools.distributions import binomial
from jaxed.tools.noise import depolarizing_noise
from jaxed.shadow import SEEQSTShadow, PauliShadow, CliffordShadow
from jaxed.experiments import Experiments, ShadowScalingExperiment
from functools import partial
from jax import random, Array
from jax import numpy as jnp
from GPshadow import GPShadow


def GHZ_Klocal(locality: int):
    """Return a GHZType subclass with exactly ``locality`` active qubits."""

    class KLocalGHZ(GHZType):
        @classmethod
        def init_random(cls, key: Array, N_state: int, n: int) -> State:
            if not 1 <= locality <= n:
                raise ValueError("GHZ locality must satisfy 1 <= locality <= n")

            key_blocks, key_xy = random.split(key)
            priorities = random.uniform(
                key_blocks,
                (N_state, n),
                dtype=jnp.float32,
            )
            active = jnp.argsort(priorities, axis=1)[:, :locality]
            blocks = jnp.zeros((N_state, n), dtype=jnp.int32)
            blocks = blocks.at[jnp.arange(N_state)[:, None], active].set(1)
            xy = random.randint(
                key_xy,
                (N_state,),
                0,
                2,
                dtype=jnp.int32,
            )
            return cls.init(blocks, xy)

    KLocalGHZ.__name__ = f"GHZ_{locality}local"
    return KLocalGHZ


def test_multiple_hom_paulis(n: int):
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 15)]
    N_obs = 50
    reps = 5
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2,3], n=n, N=N_obs) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQST", "results/Pauli", "results/Clifford"]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    kernels = ["Hamming", "SeparatedHamming", "PauliProduct"]
    paths_GP = ["results/Hamming", "results/SeparatedHamming", "results/PauliProduct"]
    experiments_list = []
    for i, ker in enumerate(kernels):
        shadow_args = {"cfg":{"kernel_type": ker, "kernel_args": {},
                            "share_parameters": True}}
        exp = ShadowScalingExperiment(GPShadow, HRState, shadow_args=shadow_args, state_name=state_name,
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                         verbose=True, key=key2, path_save=paths_GP[i],k_local=n)
        experiments_list.append(exp)
        
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                         verbose=True, key=key2, path_save=data_paths[i], 
                                         state_batch_size=1,k_local=n
                                         )
        experiments_list.append(exp)
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_n{}_Nobs{}_reps{}_N{}.pdf".format(n,N_obs,reps,max(Ns)))

def test_multiple_hom_paulis_klocal(n,k):
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 15)]
    N_obs = 50
    reps = 5
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2,3], n=n, N=N_obs,k_local=k) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQST", "results/Pauli", "results/Clifford"]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    kernels = ["Hamming", "SeparatedHamming", "PauliProduct"]
    paths_GP = ["results/Hamming", "results/SeparatedHamming", "results/PauliProduct"]
    experiments_list = []
    for i, ker in enumerate(kernels):
        shadow_args = {"cfg":{"kernel_type": ker, "kernel_args": {},
                            "share_parameters": True}}
        exp = ShadowScalingExperiment(GPShadow, HRState, shadow_args=shadow_args, state_name=state_name,
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                         verbose=True, key=key2, path_save=paths_GP[i],k_local=k)
        experiments_list.append(exp)
        
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                         verbose=True, key=key2, path_save=data_paths[i], 
                                         state_batch_size=1,k_local=k
                                         )
        experiments_list.append(exp)
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_n{}_k{}_Nobs{}_reps{}_N{}.pdf".format(n,k,N_obs,reps,max(Ns)))

def XY_combos(n):
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 17)]
    N_obs = 50
    reps = 1
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2], n=n, N=N_obs) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQSTXY", "results/PauliXY", "results/CliffordXY"]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiments_list=[]
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                            N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                            verbose=True, key=key2, path_save=data_paths[i], 
                                            state_batch_size=1
                                            )
        experiments_list.append(exp)
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_n{}_Nobs{}_reps{}_N{}.pdf".format(n,N_obs,reps,max(Ns)))

def example_experiment(n, k):
    if not 0 < k < n:
        raise ValueError("The tuned binomial experiment requires 0 < k < n")

    q = k/n
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 13)]
    N_obs = 50
    reps = 10
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2], n=n, N=N_obs,
                                             k_local=k, padding_indices=[0,3]) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQSTEx1", "results/PauliEx", "results/CliffordEx",
                  f"results/SEEQSTEx2_q{k}of{n}"]
    experiment_names = []
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiment_names = ["SEEQST unif", "Pauli", "Clifford", "SEEQST binom"]
    experiments_list=[]
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                            N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                            verbose=True, key=key2, path_save=data_paths[i],
                                            state_batch_size=1,
                                            experiment_name=experiment_names[i],
                                            )
        experiments_list.append(exp)
    # For observables with exactly k X/Y factors, q=k/n maximizes the
    # probability q**k * (1-q)**(n-k) of drawing the useful block mask.
    experiments_list.append(ShadowScalingExperiment(SEEQSTShadow, HRState, shadow_args={"distribution": partial(binomial, q=q)},
                                                    state_name=state_name,
                                                    N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                                    verbose=True, key=key2, path_save=data_paths[3],
                                                    state_batch_size=1,
                                                    experiment_name=experiment_names[3]
                                                    ))
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_n{}_Nobs{}_reps{}_N{}.pdf".format(n,N_obs,reps,max(Ns)),
                             xlog=True, ylog=True)

def example_experiment_batched(n, k, bs_state=1, bs_obs=1, bs_N=1):
    if not 0 < k < n:
        raise ValueError("The tuned binomial experiment requires 0 < k < n")

    q = k/n
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 13)]
    N_obs = 50
    reps = 10
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2], n=n, N=N_obs,
                                             k_local=k, padding_indices=[0,3]) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQSTEx1", "results/PauliEx", "results/CliffordEx",
                  f"results/SEEQSTEx2_q{k}of{n}"]
    experiment_names = []
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiment_names = ["SEEQST unif", "Pauli", "Clifford", "SEEQST binom"]
    experiments_list=[]
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = lambda n: {"simulator": "pennylane"} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                            N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                            verbose=True, key=key2, path_save=data_paths[i],
                                            state_batch_size=bs_state, batch_size_obs=bs_obs, batch_size_N=bs_N,
                                            experiment_name=experiment_names[i]
                                            )
        experiments_list.append(exp)
    # For observables with exactly k X/Y factors, q=k/n maximizes the
    # probability q**k * (1-q)**(n-k) of drawing the useful block mask.
    experiments_list.append(ShadowScalingExperiment(SEEQSTShadow, HRState, 
                                                    shadow_args=lambda n: {"distribution": partial(binomial, q=q),
                                                                 "simulator": "pennylane"},
                                                    state_name=state_name,
                                                    N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                                    verbose=True, key=key2, path_save=data_paths[3],
                                                    state_batch_size=bs_state, batch_size_obs=bs_obs, batch_size_N=bs_N,
                                                    experiment_name=experiment_names[3]
                                                    ))
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var_zero_segments(
        path_save="results/scaling_batched_n{}_Nobs{}_reps{}_N{}.pdf".format(
            n, N_obs, reps, max(Ns)
        ),
        xlog=True,
        ylog=True,
        metric="rmse",
    )


def experiment_majorana_batched(n, k, bs_state=1, bs_obs=1, bs_N=1):
    if not 1 <= k <= n:
        raise ValueError("The fermionic RDM order must satisfy 1 <= k <= n")

    ns = list(range(n, n + 1))
    Ns = [2**power for power in range(7, 13)]
    reps = 10
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = majorana_observables_list(ns, k)
    N_obs = obs_lists[0].params.shape[0]

    # Jordan-Wigner Majorana strings have varying X/Y weights. Use their
    # mean weight for the single tuned Bernoulli-mask comparison.
    xy_support = (obs_lists[0].params == 1) | (obs_lists[0].params == 2)
    q = float(jnp.mean(jnp.sum(xy_support, axis=1)) / n)
    q = min(max(q, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))

    state_name = "Majorana_state"
    data_paths = [
        f"results/SEEQSTMajorana_rdm{k}",
        f"results/PauliMajorana_rdm{k}",
        f"results/CliffordMajorana_rdm{k}",
        f"results/FGUMajorana_rdm{k}",
        f"results/SEEQSTMajoranaBinom_rdm{k}_q{q:.3f}",
    ]
    shadow_classes = [
        SEEQSTShadow,
        PauliShadow,
        CliffordShadow,
        FermionicGaussianShadow,
    ]
    experiment_names = [
        "SEEQST unif",
        "Pauli",
        "Clifford",
        "FGU",
        "SEEQST binom",
    ]
    experiments_list = []

    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        experiments_list.append(
            ShadowScalingExperiment(
                shadow_cls,
                MajoranaState,
                shadow_args={
                    "simulator": (
                        "statevector"
                        if shadow_cls is FermionicGaussianShadow
                        else "pennylane"
                    )
                },
                state_name=state_name,
                N_list=Ns,
                n_list=ns,
                obs_lists=obs_lists,
                N_state_reps=reps,
                verbose=True,
                key=key2,
                path_save=data_paths[i],
                state_batch_size=bs_state,
                batch_size_obs=bs_obs,
                batch_size_N=bs_N,
                k_local=k,
                experiment_name=experiment_names[i],
            )
        )

    key, key2 = random.split(key)
    experiments_list.append(
        ShadowScalingExperiment(
            SEEQSTShadow,
            MajoranaState,
            shadow_args={
                "distribution": partial(binomial, q=q),
                "simulator": "pennylane",
            },
            state_name=state_name,
            N_list=Ns,
            n_list=ns,
            obs_lists=obs_lists,
            N_state_reps=reps,
            verbose=True,
            key=key2,
            path_save=data_paths[4],
            state_batch_size=bs_state,
            batch_size_obs=bs_obs,
            batch_size_N=bs_N,
            k_local=k,
            experiment_name=experiment_names[4],
        )
    )

    experiments = Experiments(experiments_list)
    experiments.plot_avg_var_zero_segments(
        path_save=(
            f"results/majorana_scaling_batched_n{n}_k{k}"
            f"_Nobs{N_obs}_reps{reps}_N{max(Ns)}.pdf"
        ),
        xlog=True,
        ylog=True,
        metric="rmse",
    )

def experiment_random_GHZ(nmin, nmax, n_step, N, k, bs_state=1, bs_obs=1, bs_N=1):
    ns = list(range(nmin, nmax, n_step))
    if not ns:
        raise ValueError("The qubit range must contain at least one value")
    if not 1 <= k < min(ns):
        raise ValueError("k must satisfy 1 <= k < min(ns)")

    N_obs = 100
    reps = 1
    key = random.PRNGKey(12345)
    state_name = "GHZ_state"
    experiments_list = []
    for xy_locality in range(1, k + 1):
        key, key2 = random.split(key)
        obs_keys = random.split(key2, len(ns))
        obs_lists = [
            PauliObservable.init_random(
                obs_key,
                sample_indices=[1, 2],
                n=n,
                N=N_obs,
                k_local=xy_locality,
                padding_indices=[0, 3],
            )
            for obs_key, n in zip(obs_keys, ns)
        ]
        key, key2 = random.split(key)
        experiments_list.append(
            ShadowScalingExperiment(
                SEEQSTShadow,
                GHZ_Klocal(xy_locality),
                shadow_args=lambda n, locality=xy_locality: {
                    "distribution": partial(binomial, q=locality / n),
                    "simulator": "clifford",
                },
                state_name=state_name,
                N_list=[N],
                n_list=ns,
                obs_lists=obs_lists,
                N_state_reps=reps,
                verbose=True,
                key=key2,
                path_save=f"results_new/SEEQSTGHZ_xy{xy_locality}",
                state_batch_size=bs_state,
                batch_size_obs=bs_obs,
                batch_size_N=bs_N,
                k_local=xy_locality,
                experiment_name=(
                    f"SEEQST binom q={xy_locality}/n; "
                    f"Z/I locality n-{xy_locality}"
                ),
            )
        )
    experiments = Experiments(experiments_list)
    experiments.plot_avg_var(
        path_save="results_new/GHZscaling_batched_nmax{}_Nobs{}_reps{}_N{}.pdf".format(
            max(ns), N_obs, reps, N
        ),
        x="n",
        xlog=False,
        ylog=True,
        metric="rmse",
    )

def noisy_experiment_batched(n, k, bs_state=1, bs_obs=1, bs_N=1):
    if not 0 < k < n:
        raise ValueError("The tuned binomial experiment requires 0 < k < n")

    q = k/n
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 13)]
    N_obs = 50
    reps = 10
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2], n=n, N=N_obs,
                                             k_local=k, padding_indices=[0,3]) for n in ns]
    state_name = "HR_state"
    data_paths = ["results_new/SEEQSTEx1", "results_new/PauliEx", "results_new/CliffordEx",
                  f"results_new/SEEQSTEx2_q{k}of{n}"]
    experiment_names = []
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiment_names = ["SEEQST unif", "Pauli", "Clifford", "SEEQST binom"]
    experiments_list=[]
    noise_fn = partial(depolarizing_noise, p=0.1)
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = lambda n: {"simulator": "pennylane",
                                 "noise_fun": noise_fn} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState,
                                      shadow_args=shadow_args, state_name=state_name,
                                            N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                            verbose=True, key=key2, path_save=data_paths[i],
                                            state_batch_size=bs_state, batch_size_obs=bs_obs, batch_size_N=bs_N,
                                            experiment_name=experiment_names[i]
                                            )
        experiments_list.append(exp)
    # For observables with exactly k X/Y factors, q=k/n maximizes the
    # probability q**k * (1-q)**(n-k) of drawing the useful block mask.
    experiments_list.append(ShadowScalingExperiment(SEEQSTShadow, HRState,
                                                    shadow_args=lambda n: {"distribution": partial(binomial, q=q),
                                                                 "simulator": "pennylane",
                                                                 "noise_fun": noise_fn},
                                                    state_name=state_name,
                                                    N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                                    verbose=True, key=key2, path_save=data_paths[3],
                                                    state_batch_size=bs_state, batch_size_obs=bs_obs, batch_size_N=bs_N,
                                                    experiment_name=experiment_names[3]
                                                    ))
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var_zero_segments(
        path_save="results_new/scaling_batched_n{}_Nobs{}_reps{}_N{}.pdf".format(
            n, N_obs, reps, max(Ns)
        ),
        xlog=True,
        ylog=True,
        metric="rmse",
    )

if __name__ == "__main__":
    """test_multiple_hom_paulis(4)
    test_multiple_hom_paulis_klocal(4,2)
    test_multiple_hom_paulis(5)
    test_multiple_hom_paulis_klocal(5,2)
    test_multiple_hom_paulis(10)
    test_multiple_hom_paulis_klocal(10,2)"""
    # XY_combos(6)
    # example_experiment_batched(20, 19, 1, None, 64)
    # example_experiment_batched(10, 8, 1, None, None)
    # example_experiment(15, 14)
    # experiment_random_GHZ(4, 10, 2, 2**12, 3, 1, 5, 1024)
    noisy_experiment_batched(10, 8, 1, None, None)
