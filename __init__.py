import os
import numpy as np
import time
from badger import environment
from .utils import k_taper, taper_output


class Environment(environment.Environment):
    name = "lcls_taper"

    variables = {
        "a": [0.02, 0.06],
        "split_ix": [16,48],
        "powr": [1.9, 2.1],
    }
    observables = ["power","k_profile","Kact","DSKact"]
    #observables = ["power","k_profile","Kact","DSKact","Stop","Stop_Reason","Runtime_s","Eval_count"]

    def __init__(self, interface=None, params=None):
        super().__init__(interface=interface, params=params)

        # Params as plain dict (avoid self.params)
        p = dict(params) if isinstance(params, dict) else {}
        p.setdefault("particle_pos", "SASE_particle_position.csv")
        p.setdefault("k0", 2.55)
        p.setdefault("n", 64)

        # --- add these here ---
        p.setdefault("mode", "sim")  # "sim" or "machine"
        p.setdefault("power_pv", "GDET:FEE1:241:ENRCHSTCUHBR")
        p.setdefault("power_points", 120)
        p.setdefault("power_stat", "percent_80")  # "percent_80" | "mean" | "last"
        p.setdefault("power_scale", 1.0)
        # ----------------------

        object.__setattr__(self, "_params", p)

        n = int(p["n"])
        self.variables["split_ix"] = [int(0.25 * n), int(0.75 * n)]

        object.__setattr__(self, "_vars", {"a": 0.02, "split_ix": 16, "powr": 2.0})
        object.__setattr__(self, "_obs", {"power": None, "k_profile": None,"Kact": None, "DSKact": None})
        object.__setattr__(self, "_modified", True)
        ###
        #object.__setattr__(self, "_obs", {
        #                    "power": None,
        #                    "k_profile": None,
        #                    "Kact": None,
        #                    "DSKact": None,
        #                    "stop": 0,
        #                    "stop_reason": "",
        #                    "runtime_s": 0.0,
        #                    "eval_count": 0,
        #})
        #object.__setattr__(self, "_modified", True)
        #object.__setattr__(self, "_t0", time.time())
        #object.__setattr__(self, "_p_hist", [])
        ###
        env_root = os.path.dirname(os.path.realpath(__file__))
        particle_pos_file = os.path.join(env_root, "data", p["particle_pos"])
        object.__setattr__(self, "_particle_pos_file", particle_pos_file)

        default_input = dict(
            npart=512,
            s_steps=64,
            z_steps=n,
            energy=4313.34e6,
            eSpread=0,
            emitN=1.2e-6,
            currentMax=3900,
            beta=26,
            unduPeriod=0.03,
            unduK=np.full(n, float(p["k0"])),
            unduL=70,
            radWavelength=None,
            random_seed=31,
            particle_position=None,  # lazy-load
            hist_rule="square-root", # 'square-root' or 'sturges' or 'rice-rule' or 'self-design'
            iopt="sase",
        )
        object.__setattr__(self, "DEFAULT_INPUT", default_input)
        object.__setattr__(self, "z", None)
        object.__setattr__(self, "power", None)

    def get_variables(self, variable_names):
        v = getattr(self, "_vars")
        return {k: v[k] for k in variable_names}

    def set_variables(self, variable_inputs):
        v = getattr(self, "_vars")
        changed = False
        for k, val in variable_inputs.items():
            if k == "split_ix":
                val = int(val)
            else:
                val = float(val)
            if v.get(k) != val:
                v[k] = val
                changed = True
        if changed:
            object.__setattr__(self, "_modified", True)

    def get_observables(self, observable_names):
        if getattr(self, "_modified"):
            default_input = getattr(self, "DEFAULT_INPUT")

            # lazy-load particle positions
            if default_input.get("particle_position", None) is None:
                pfile = getattr(self, "_particle_pos_file")
                if os.path.exists(pfile):
                    default_input["particle_position"] = np.genfromtxt(pfile, delimiter=",")
                else:
                    default_input["particle_position"] = None

            p = getattr(self, "_params")
            v = getattr(self, "_vars")

            K = k_taper(
                k0=float(p["k0"]),
                n=int(p["n"]),
                a=float(v["a"]),
                split_ix=int(v["split_ix"]),
                powr=float(v["powr"]),
            )

            K = np.asarray(K, dtype=float)
            n = K.size
            nseg = 32

            x_old = np.linspace(0.0, 1.0, n, dtype = float)
            x_edges = np.linspace(0.0,1.0, nseg+1, dtype = float)

            Kact = np. interp(x_edges[:-1], x_old, K)    # upstream of undulator
            DSKact = np. interp(x_edges[1:], x_old, K)    # downstream of undulator

            _count = getattr(self, "_count", 0) + 1
            object.__setattr__(self,"_count", _count)
            
            print(f"\n[count {_count:04d}] Kact/DSKact (nseg={nseg}, from n={n})")
            print("Kact  :", np.array2string(Kact,   precision=3, separator=", ", max_line_width=200))
            print("DSKact:", np.array2string(DSKact, precision=3, separator=", ", max_line_width=200))

            #### working version 02182026
            #z, power = taper_output(K, default_input)
            #object.__setattr__(self, "z", z)
            #object.__setattr__(self, "power", power)
            #
            #obs = getattr(self, "_obs")
            #obs["power"] = float(power[-1]) * 1e-9  # GW
            #obs["k_profile"] = np.asarray(K, dtype = float)
            #obs["Kact"] = np.asanyarray(Kact, dtype = float)
            #obs["DSKact"] = np.asanyarray(DSKact, dtype = float)
            #object.__setattr__(self, "_modified", False)
            #### working version 02182026
            
            obs = getattr(self, "_obs")

            mode = p.get("mode", "sim")

            if mode == "machine":
                # Optional: allow detector buffer to accumulate
                # (uncomment import time at top if you use this)
                sleep_s = float(p.get("power_sleep_s", 0.0))

                pv = p.get("power_pv", "GDET:FEE1:241:ENRCHSTCUHBR")
                points = int(p.get("power_points", 120))
                stat = p.get("power_stat", "percent_80")   # "percent_80" | "mean" | "last"
                scale = float(p.get("power_scale", 1.0))

                raw = self.interface.get_value(pv)

                # raw may be waveform (array-like) or scalar
                arr = np.asarray(raw, dtype=float).reshape(-1)
                if arr.size > 0:
                    if arr.size >= points:
                         arr = arr[-points:]

                    if stat == "mean":
                        power_val = float(np.mean(arr))
                    elif stat == "last":
                        power_val = float(arr[-1])
                    else:
                        power_val = float(np.percentile(arr, 80))
                else:
                    power_val = float(raw)

                # No sim arrays in machine mode
                object.__setattr__(self, "z", None)
                object.__setattr__(self, "power", None)

                obs["power"] = power_val * scale

            else:
                # Simulation mode (your current behavior)
                z, power = taper_output(K, default_input)
                object.__setattr__(self, "z", z)
                object.__setattr__(self, "power", power)
                obs["power"] = float(power[-1]) * 1e-9  # GW

            # Always publish these (GUI + logging)
            obs["k_profile"] = np.asarray(K, dtype=float)
            obs["Kact"] = np.asanyarray(Kact, dtype=float)
            obs["DSKact"] = np.asanyarray(DSKact, dtype=float)

            object.__setattr__(self, "_modified", False)
            
            
            
            
            
            
            
            ###
            #p_gw = float(obs["power"])
            #t0 = getattr(self, "_t0", None)
            #if t0 is None or callable(t0):
            #    t0 = time.time()
            #    object.__setattr__(self, "_t0", t0)

            #Runtime_s = time.time() - t0

            #Eval_count = int(getattr(self, "_count", 0))

            #p_hist = getattr(self, "_p_hist", None)
            #if p_hist is None or callable(p_hist):
            #    p_hist = []
            #p_hist.append(p_gw)
            #p_hist = p_hist[-5:]
            #object.__setattr__(self, "_p_hist", p_hist)

            #stop = False
            #reasons = []

            # (1) runtime > 5 min
            #if Runtime_s >= 1 * 60:
            #    stop = True
            #    reasons.append(f"runtime {Runtime_s:.1f}s >= 300s")
            # (2) max iteration >= 300
            #if Eval_count >= 30:
            #    stop = True
            #    reasons.append(f"eval {Eval_count} >= 300")
            # (3) last-5 variation within 5% of last-5 max
            #if len(p_hist) == 5:
            #    pmax = max(p_hist)
            #    pmin = min(p_hist)
            #    if pmax > 0:
            #        variation = (pmax - pmin) / pmax
            #        if variation <= 0.5:
            #            stop = True
            #            reasons.append(f"last5 variation {(variation*100):.2f}% <= 5% (pmax={pmax:.4g}GW)")
            #    else:
            #        if pmin == 0:
            #            stop = True
            #            reasons.append("last5 all zero power")

            #obs["stop"] = int(stop)
            #obs["stop_reason"] = "; ".join(reasons) if reasons else ""
            #obs["runtime_s"] = float(Runtime_s)
            #obs["eval_count"] = int(Eval_count)
            ###
            

        obs = getattr(self, "_obs")
        return {k: obs[k] for k in observable_names}
