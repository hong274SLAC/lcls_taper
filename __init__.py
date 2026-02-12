import os
import numpy as np

from badger import environment
from .utils import k_taper, taper_output


class Environment(environment.Environment):
    name = "lcls_taper"

    variables = {
        "a": [0.0, 0.5],
        "split_ix": [50, 150],
        "powr": [1.9, 2.1],
    }
    observables = ["power"]

    def __init__(self, interface=None, params=None):
        super().__init__(interface=interface, params=params)

        # Params as plain dict (avoid self.params)
        p = dict(params) if isinstance(params, dict) else {}
        p.setdefault("particle_pos", "SASE_particle_position.csv")
        p.setdefault("k0", 3.5)
        p.setdefault("n", 200)
        object.__setattr__(self, "_params", p)

        n = int(p["n"])
        self.variables["split_ix"] = [int(0.25 * n), int(0.75 * n)]

        object.__setattr__(self, "_vars", {"a": 0.06, "split_ix": 80, "powr": 2.0})
        object.__setattr__(self, "_obs", {"power": None})
        object.__setattr__(self, "_modified", True)

        env_root = os.path.dirname(os.path.realpath(__file__))
        particle_pos_file = os.path.join(env_root, "data", p["particle_pos"])
        object.__setattr__(self, "_particle_pos_file", particle_pos_file)

        default_input = dict(
            npart=512,
            s_steps=200,
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

            z, power = taper_output(K, default_input)
            object.__setattr__(self, "z", z)
            object.__setattr__(self, "power", power)

            obs = getattr(self, "_obs")
            obs["power"] = float(power[-1]) * 1e-9  # GW
            object.__setattr__(self, "_modified", False)

        obs = getattr(self, "_obs")
        return {k: obs[k] for k in observable_names}
