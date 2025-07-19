"""
Distributed processing utilities.

@author: Nicola Vigano
"""

from collections.abc import Callable, Sequence
from multiprocessing import cpu_count

import numpy as np
from dask import config as dd_config
from dask_jobqueue.slurm import SLURMCluster
from distributed import Client, Future, LocalCluster, SpecCluster, as_completed
from numpy.typing import NDArray
from tqdm.auto import tqdm, trange


def _get_inherit_config() -> list[str]:
    return [f"export DASK_INTERNAL_INHERIT_CONFIG={dd_config.serialize(dd_config.global_config)}"]


def _get_pre_spawn_env(num_threads: int) -> dict[str, str]:
    env_threading = {var: f"{num_threads}" for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}
    return {**env_threading, "MALLOC_TRIM_THRESHOLD_": f"{2**17}"}


def _get_dask_config(num_threads: int) -> dict:
    return {"distributed.nanny.pre-spawn-environ": _get_pre_spawn_env(num_threads)}


def get_cluster(
    cluster_type: str = "local",
    num_workers: int = 1,
    num_threads: int | None = None,
    memory: str | None = None,
    walltime: str | None = None,
    job_extra_directives: Sequence[str] | None = None,
) -> SpecCluster:
    """Create a cluster object, to be used for distributed computation.

    Parameters
    ----------
    cluster_type : str, optional
        Type of cluster. Possible choices: ["local" | "slurm" | "slurm-cpu"
        | "slurm-cpu-long" | "slurm-gpu" | "slurm-gpu-long"], by default "local"
    num_workers : int, optional
        Number of jobs/workers, by default 1
    num_threads : int | None, optional
        Number of threads to grant to the underlying implementation (NumPy, BLAS,
        MKL, etc), by default None, which will be converted in a specific number,
        depending on the cluster type
    memory : str | None, optional
        How much memory to assign to the jobs, by default None
    walltime : str | None, optional
        How much time to assign for the jobs, by default None
    job_extra_directives : Sequence[str] | None, optional
        Extra directives like e.g. ["--constraint", "hpc7"], by default None

    Returns
    -------
    SpecCluster
        A cluster object, that can be used in context managers.
    """
    match (cluster_type.split("-")):
        case ["local"]:
            if num_threads is None:
                num_cores = cpu_count()
                num_threads = num_cores // num_workers

            with dd_config.set(_get_dask_config(num_threads)):
                cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)

        case ["slurm", *args]:
            if num_threads is None:
                num_threads = 8

            if job_extra_directives is None:
                job_extra_directives = []
            else:
                job_extra_directives = list(job_extra_directives)

            match (args):
                case [] | ["cpu"] | ["cpu", "long"]:
                    queue = "-".join(["nice", *args[1:]])
                    if memory is None:
                        memory = "4GB"
                    if walltime is None:
                        walltime = "2:00:00" if len(args) < 2 else "24:00:00"
                case ["gpu"] | ["gpu", "long"]:
                    queue = "-".join(args)
                    if memory is None:
                        memory = "16GB"
                    job_extra_directives.append("--gres=gpu:1")
                    if walltime is None:
                        walltime = "1:00:00" if len(args) < 2 else "12:00:00"
                case _:
                    raise ValueError(f"Unknown cluster queue: {cluster_type}")

            with dd_config.set(_get_dask_config(num_threads)):
                inherit_config = _get_inherit_config()

            cluster = SLURMCluster(
                queue=queue,
                cores=1,
                processes=1,
                job_cpu=num_threads,
                memory=memory,
                log_directory="tmp",
                job_script_prologue=inherit_config,
                job_extra_directives=job_extra_directives,
            )
            cluster.scale(jobs=num_workers)

        case _:
            raise ValueError(f"Unknown cluster queue: {cluster_type}")

    return cluster


def get_client(cluster: SpecCluster, verbose: bool = True) -> Client:
    """
    Create a client connected to the specified cluster.

    Parameters
    ----------
    cluster : SpecCluster
        The cluster specification to connect to.
    verbose : bool, optional
        If True, print the client's dashboard link. Default is True.

    Returns
    -------
    Client
        The client connected to the specified cluster.
    """
    client = Client(cluster)
    if verbose:
        print(f"Client's dashboard: {client.dashboard_link}")
    return client


def process_buckets_series(
    masks: NDArray,
    buckets: NDArray,
    reconstruction: Callable[[NDArray, NDArray], tuple[float, NDArray]],
    client: Client | None = None,
    verbose: bool = True,
) -> tuple[NDArray, NDArray]:
    """Batch processing routine for buckets series.

    Parameters
    ----------
    masks : NDArray
        Set or sets of masks
    buckets : NDArray
        Sets of buckets
    reconstruction : Callable[[NDArray, NDArray], tuple[float, NDArray]]
        Reconstruction function
    client : Union[Client, None], optional
        Dask distributed client, by default None, which does not use a cluster
    verbose : bool, optional
        Whether to display verbose information, by default True

    Returns
    -------
    tuple[NDArray, NDArray]
        The list of regularization weights and reconstructions.
    """
    buckets = np.array(buckets, ndmin=2)
    num_bucket_sets = buckets.shape[-2]

    reg_vals = np.zeros(num_bucket_sets)
    recs = [np.array([])] * num_bucket_sets

    broadcast_masks = masks.ndim == 3
    if not broadcast_masks and masks.shape[-4] != num_bucket_sets:
        raise ValueError(f"Number of masks sets {masks.shape[-4]} different from the number of bucket sets {num_bucket_sets}")
    if verbose:
        print(f"Buckets sets: {num_bucket_sets}, broadcasting of masks: {broadcast_masks}")

    if client is None:
        if verbose:
            print("Serial processing: Using local machine")
        for ii in trange(num_bucket_sets, disable=not verbose, desc="Bucket sets"):
            reg_vals[ii], recs[ii] = reconstruction(masks if broadcast_masks else masks[ii], buckets[ii])
    else:

        def _rec(unaligned_w: NDArray, unaligned_y: NDArray) -> tuple[float, NDArray]:
            return reconstruction(unaligned_w.copy(), unaligned_y.copy())

        if verbose:
            print(f"Parallel processing: Using a cluster of {len(client.cluster.workers)} workers")
            print(f"- Client dashboard: {client.dashboard_link}")
        if broadcast_masks:
            masks_dd = client.scatter(masks, broadcast=True)
            future_to_ind = {client.submit(_rec, masks_dd, bucks): ii for ii, bucks in enumerate(buckets)}
        else:
            masks_dd = [client.scatter(m) for m in masks]
            future_to_ind = {client.submit(_rec, masks_dd[ii], bucks): ii for ii, bucks in enumerate(buckets)}

        recs = [np.array([])] * num_bucket_sets
        reg_vals = np.zeros(num_bucket_sets)
        for future in tqdm(as_completed(future_to_ind), desc="Bucket sets", disable=not verbose, total=num_bucket_sets):
            if not isinstance(future, Future):
                raise ValueError("Error with distributed!", future)
            ii = future_to_ind[future]
            try:
                reg_vals[ii], recs[ii] = future.result()
            except ValueError as exc:
                print(f"Bucket set #{ii} generated an exception: {exc}")
                raise

    return reg_vals, np.stack(recs, axis=0)
