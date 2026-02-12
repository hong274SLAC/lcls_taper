import numpy as np


def k_taper(k0=3.5, a=0.5, n=200, split_ix=80, powr=2):
    """
    Build an undulator K taper profile of length n.

    Piecewise:
      i < split_ix:     K[i] = k0
      i >= split_ix:    K[i] = k0 * (1 - a * x^powr), x in [0,1]

    Returns
    -------
    K : np.ndarray, shape (n,)
    """
    n = int(n)
    split_ix = int(split_ix)

    if n <= 1:
        raise ValueError(f"n must be > 1, got {n}")
    if not (0 <= split_ix <= n):
        raise ValueError(f"split_ix must be within [0, n], got split_ix={split_ix}, n={n}")

    # If split_ix == n, taper part is empty -> all ones
    head = np.ones(split_ix, dtype=float)
    tail_len = n - split_ix
    if tail_len > 0:
        x = np.linspace(0.0, 1.0, tail_len, dtype=float)
        tail = 1.0 - float(a) * (x ** float(powr))
        profile = np.hstack([head, tail])
    else:
        profile = head

    return profile * float(k0)


def taper_output(unduK, DEFAULT_INPUT):
    """
    Run zfel.sase1d with an undulator K profile.

    Parameters
    ----------
    unduK : array-like, shape (n,)
        Taper profile along the undulator.
    DEFAULT_INPUT : dict
        Base sase1d input dictionary.

    Returns
    -------
    z : np.ndarray
        Position array along the undulator.
    power_z : np.ndarray
        Output power along the undulator.
    """
    # Import zfel lazily so Badger can import the plugin even if zfel is missing.
    try:
        from zfel import sase1d
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "zfel is required to run this environment but is not installed in the current Python env.\n"
            "Install it with:  pip install zfel\n"
            "Then restart the Badger GUI."
        ) from e

    unduK = np.asarray(unduK, dtype=float).reshape(-1)
    if unduK.size < 2:
        raise ValueError(f"unduK must have length >= 2, got {unduK.size}")

    sase_input = DEFAULT_INPUT.copy()
    sase_input["unduK"] = unduK
    sase_input["z_steps"] = int(unduK.shape[0])

    output = sase1d.sase(sase_input)

    # These keys are what your code expects from zfel
    z = np.asarray(output["z"])
    power_z = np.asarray(output["power_z"])

    return z, power_z

def k_profile_to_segments(K_profile, n_segments, method="mean"): 
    """ Convert a fine K(z) profile (length n) into one K value per undulator segment. 
    Parameters 
    ---------- 
    K_profile : array-like, shape (n,) 
        Fine K profile (e.g. output of k_taper). 
    n_segments : int Number of undulator segments (e.g. HXR segments). 
    method : str 'mean' (recommended), 'median', or 'end' 
    
    Returns 
    ------- 
    Kseg : np.ndarray, shape (n_segments,) 
        One K value per segment. 
    edges : np.ndarray, shape (n_segments+1,) 
        Index boundaries used for each segment in the original profile. 
    """ 
    K = np.asarray(K_profile, dtype=float).reshape(-1) 
    n = K.size 
    n_segments = int(n_segments) 
    
    if n_segments <= 0: 
        raise ValueError("n_segments must be > 0") 
    if n_segments > n: 
        raise ValueError(f"n_segments ({n_segments}) cannot exceed len(K_profile) ({n})") 
    
    # segment boundaries in index space 
    edges = np.floor(np.linspace(0, n, n_segments + 1)).astype(int) 
    edges[-1] = n 
    
    # ensure last edge lands exactly on n 
    Kseg = np.empty(n_segments, dtype=float) 
    for j in range(n_segments): 
        sl = slice(edges[j], edges[j + 1]) 
        chunk = K[sl] 
        if chunk.size == 0: 
            raise RuntimeError(f"Empty segment slice for segment {j}: edges={edges}") 
        if method == "mean": Kseg[j] = chunk.mean() 
        
        elif method == "median": Kseg[j] = np.median(chunk) 
        
        elif method == "end": Kseg[j] = chunk[-1] 
        
        else: 
            raise ValueError("method must be 'mean', 'median', or 'end'") 
        
    return Kseg, edges

def k_profile_to_segments_2endpoints(K_profile, n_segments, mode="endpoint"):
    """
    Convert a fine K(z) profile into per-segment values with TWO endpoints:
      - Kact   : K at beginning of each undulator segment (Ki)
      - DSKact : K at end of each undulator segment (Kt)

    Parameters
    ----------
    K_profile : array-like, shape (n,)
        Fine K profile along z (e.g. output of k_taper).
    n_segments : int
        Number of undulator segments.
    mode : str
        How to define "beginning/end" from the fine grid:
        - "endpoint" (default): Ki = first sample in chunk, Kt = last sample in chunk
        - "mean_endcaps": Ki = mean of first few points, Kt = mean of last few points (less noisy)

    Returns
    -------
    Kact : np.ndarray, shape (n_segments,)
        Per-segment beginning K values (Ki).
    DSKact : np.ndarray, shape (n_segments,)
        Per-segment end K values (Kt).
    edges : np.ndarray, shape (n_segments+1,)
        Index boundaries used for each segment in the original profile.
    """
    K = np.asarray(K_profile, dtype=float).reshape(-1)
    n = K.size
    n_segments = int(n_segments)

    if n_segments <= 0:
        raise ValueError("n_segments must be > 0")
    if n_segments > n:
        raise ValueError(f"n_segments ({n_segments}) cannot exceed len(K_profile) ({n})")

    # Segment boundaries in index space
    edges = np.floor(np.linspace(0, n, n_segments + 1)).astype(int)
    edges[-1] = n

    Kact = np.empty(n_segments, dtype=float)
    DSKact = np.empty(n_segments, dtype=float)

    for j in range(n_segments):
        sl = slice(edges[j], edges[j + 1])
        chunk = K[sl]
        if chunk.size == 0:
            raise RuntimeError(f"Empty segment slice for segment {j}: edges={edges}")

        if mode == "endpoint":
            # Ki at beginning, Kt at end
            Kact[j] = chunk[0]
            DSKact[j] = chunk[-1]

        elif mode == "mean_endcaps":
            # Average a small cap at each end for stability
            m = max(1, min(3, chunk.size // 2))  # use up to 3 points, but not more than half the chunk
            Kact[j] = chunk[:m].mean()
            DSKact[j] = chunk[-m:].mean()

        else:
            raise ValueError("mode must be 'endpoint' or 'mean_endcaps'")

    return Kact, DSKact, edges