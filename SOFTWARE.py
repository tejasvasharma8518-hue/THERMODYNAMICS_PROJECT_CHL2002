import math
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# SI units are used internally: Pa, m^3, J and mol.
R = 8.314462618
T_REF = 298.15
P_REF = 1.0e5

# Critical properties and ideal-gas heat-capacity coefficients for each gas.
FLUID_DATA = {
    "Carbon Dioxide (CO2)": {
        "Tc": 304.13, "Pc": 7.3773e6, "omega": 0.224,
        "cp": (19.8, 0.07344, -5.602e-5, 1.715e-8)
    },
    "Methane (CH4)": {
        "Tc": 190.56, "Pc": 4.5992e6, "omega": 0.011,
        "cp": (19.25, 0.0521, 1.197e-5, -1.132e-8)
    },
    "Ethane (C2H6)": {
        "Tc": 305.32, "Pc": 4.872e6, "omega": 0.099,
        "cp": (5.409, 0.1781, -6.938e-5, 8.713e-9)
    },
    "Nitrogen (N2)": {
        "Tc": 126.20, "Pc": 3.396e6, "omega": 0.037,
        "cp": (28.98, -1.57e-3, 8.08e-6, -2.87e-9)
    },
    "Water (H2O)": {
        "Tc": 647.10, "Pc": 22.064e6, "omega": 0.344,
        "cp": (32.24, 1.92e-3, 1.055e-5, -3.59e-9)
    },
}

class CubicModel:
    """Calculations for one pure gas using a cubic equation of state."""

    MODEL_CONSTANTS = {
        "VDW": (0.0, 0.0, 27.0 / 64.0, 1.0 / 8.0),
        "RK":  (1.0, 0.0, 0.42748, 0.08664),
        "SRK": (1.0, 0.0, 0.42748, 0.08664),
        "PR":  (2.0, -1.0, 0.45724, 0.07780),
    }

    def __init__(self, substance, model="PR"):
        if model not in self.MODEL_CONSTANTS:
            raise ValueError("Unknown EOS model.")
        self.substance = substance
        self.data = FLUID_DATA[substance]
        self.model = model
        self.Tc = self.data["Tc"]
        self.Pc = self.data["Pc"]
        self.omega = self.data["omega"]
        self.u, self.w, self.omega_a, self.omega_b = self.MODEL_CONSTANTS[model]
        self.b = self.omega_b * R * self.Tc / self.Pc
        self.a_c = self.omega_a * (R * self.Tc) ** 2 / self.Pc

    def _m_parameter(self):
        if self.model == "SRK":
            return 0.480 + 1.574 * self.omega - 0.176 * self.omega ** 2
        if self.model == "PR":
            if self.omega <= 0.49:
                return (0.37464 + 1.54226 * self.omega
                        - 0.26992 * self.omega ** 2)
            return (0.379642 + 1.48503 * self.omega
                    - 0.164423 * self.omega ** 2
                    + 0.016666 * self.omega ** 3)
        return 0.0

    # The attraction parameter changes with temperature for RK, SRK and PR.
    def attraction_terms(self, T):
        """Calculate the temperature-dependent attraction parameter."""
        if T <= 0:
            raise ValueError("Temperature must be greater than zero.")
        Tr = T / self.Tc

        if self.model == "VDW":
            alpha, kappa = 1.0, 0.0
        elif self.model == "RK":
            alpha = Tr ** -0.5
            kappa = -0.5
        else:
            m = self._m_parameter()
            root_Tr = math.sqrt(Tr)
            g = 1.0 + m * (1.0 - root_Tr)
            if abs(g) < 1e-14:
                raise ValueError("Alpha function is singular at this temperature.")
            alpha = g * g
            kappa = -m * root_Tr / g

        a = self.a_c * alpha
        da_dT = a * kappa / T
        return a, da_dT, kappa

    # Python does not have a separate real cube-root function in every version.
    @staticmethod
    def _cbrt(x):
        return math.copysign(abs(x) ** (1.0 / 3.0), x)

    @classmethod
    def real_cubic_roots(cls, a2, a1, a0):
        """Find the real roots of the cubic in Z."""
        p = a1 - a2 * a2 / 3.0
        q = (2.0 * a2 ** 3) / 27.0 - (a2 * a1) / 3.0 + a0
        discriminant = (q / 2.0) ** 2 + (p / 3.0) ** 3
        tol = 1e-14

        if discriminant > tol:
            root_disc = math.sqrt(discriminant)
            u = cls._cbrt(-q / 2.0 + root_disc)
            v = cls._cbrt(-q / 2.0 - root_disc)
            roots = [u + v - a2 / 3.0]
        elif discriminant >= -tol:
            u = cls._cbrt(-q / 2.0)
            roots = [2.0 * u - a2 / 3.0, -u - a2 / 3.0]
        else:
            radius = 2.0 * math.sqrt(-p / 3.0)
            arg = (3.0 * q / (2.0 * p)) * math.sqrt(-3.0 / p)
            theta = math.acos(max(-1.0, min(1.0, arg))) / 3.0
            roots = [
                radius * math.cos(theta - 2.0 * math.pi * k / 3.0)
                - a2 / 3.0 for k in range(3)
            ]

        unique = []
        for value in sorted(roots):
            if not unique or abs(value - unique[-1]) > 1e-9:
                unique.append(value)
        return unique

    # Convert the EOS into a cubic equation in the compressibility factor Z.
    def find_z_roots(self, T, P):
        if T <= 0 or P <= 0:
            raise ValueError("Temperature and pressure must be positive.")

        a, _, _ = self.attraction_terms(T)
        A = a * P / (R * T) ** 2
        B = self.b * P / (R * T)

        c2 = -(1.0 + B - self.u * B)
        c1 = A + self.w * B ** 2 - self.u * B - self.u * B ** 2
        c0 = -(A * B + self.w * B ** 2 + self.w * B ** 3)

        roots = self.real_cubic_roots(c2, c1, c0)
        valid = [z for z in roots if z > B and z > 0]
        return valid, A, B

    def _log_integral(self, Z, B):
        """Integral term used in the fugacity and departure-property equations."""
        if self.model == "VDW":
            return B / Z

        sigma = (self.u + math.sqrt(self.u ** 2 - 4.0 * self.w)) / 2.0
        epsilon = (self.u - math.sqrt(self.u ** 2 - 4.0 * self.w)) / 2.0
        numerator = Z + sigma * B
        denominator = Z + epsilon * B
        if numerator <= 0 or denominator <= 0:
            raise ValueError("Invalid logarithm argument in EOS property calculation.")
        return math.log(numerator / denominator) / (sigma - epsilon)

    # Use one admissible Z root to calculate fugacity, H and S.
    def state_properties(self, T, P, Z):
        """Calculate the properties at a selected compressibility-factor root."""
        a, da_dT, kappa = self.attraction_terms(T)
        A = a * P / (R * T) ** 2
        B = self.b * P / (R * T)

        if Z <= B:
            raise ValueError("Selected compressibility root is not physically admissible.")

        I = self._log_integral(Z, B)
        ln_phi = Z - 1.0 - math.log(Z - B) - (A / B) * I

        H_dep = R * T * (Z - 1.0 + (kappa - 1.0) * (A / B) * I)
        S_dep = R * (math.log(Z - B) + kappa * (A / B) * I)

        phi = math.exp(max(-700.0, min(700.0, ln_phi)))
        fugacity = phi * P

        H_ig, S_ig = self.ideal_gas_terms(T, P)
        return {
            "Z": Z,
            "Vm": Z * R * T / P,
            "ln_phi": ln_phi,
            "phi": phi,
            "f": fugacity,
            "H_dep": H_dep,
            "S_dep": S_dep,
            "H_ig": H_ig,
            "S_ig": S_ig,
            "H": H_ig + H_dep,
            "S": S_ig + S_dep,
        }

    # The ideal-gas contribution is measured from the stated reference state.
    def ideal_gas_terms(self, T, P):
        """Calculate ideal-gas H and S from the Cp polynomial."""
        a, b, c, d = self.data["cp"]

        H_ig = (
            a * (T - T_REF)
            + b * (T ** 2 - T_REF ** 2) / 2.0
            + c * (T ** 3 - T_REF ** 3) / 3.0
            + d * (T ** 4 - T_REF ** 4) / 4.0
        )

        S_ig = (
            a * math.log(T / T_REF)
            + b * (T - T_REF)
            + c * (T ** 2 - T_REF ** 2) / 2.0
            + d * (T ** 3 - T_REF ** 3) / 3.0
            - R * math.log(P / P_REF)
        )
        return H_ig, S_ig

    def pressure_from_volume(self, T, Vm):
        """Get pressure from temperature and molar volume."""
        if Vm <= self.b:
            return None
        a, _, _ = self.attraction_terms(T)
        denominator = Vm ** 2 + self.u * self.b * Vm + self.w * self.b ** 2
        if denominator <= 0:
            return None
        pressure = R * T / (Vm - self.b) - a / denominator
        return pressure if pressure > 0 else None


class PropertyApp(tk.Tk):
    """Window used to enter conditions and display the calculated results."""

    def __init__(self):
        super().__init__()
        self.title("Real Gas Analyser | Thermodynamic Dashboard")
        self.geometry("1380x900")
        self.minsize(1080, 760)
        self.configure(bg="#f3f5f8")
        self.build_interface()
        self.run_calculation()

    def build_interface(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#f3f5f8")
        style.configure("TLabel", background="#f3f5f8", foreground="#26364a",
                        font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10, "bold"),
                        padding=(14, 9), background="#137c75", foreground="white")
        style.map("TButton", background=[("active", "#0f665f")])
        style.configure("TNotebook", background="#f3f5f8", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(20, 10),
                        font=("Helvetica", 10, "bold"))
        style.configure("Treeview", rowheight=27, font=("Helvetica", 9),
                        background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Helvetica", 9, "bold"))

        # Header and short description
        header = tk.Frame(self, bg="#192d42", height=88)
        header.pack(fill=tk.X)
        tk.Label(header, text="REAL GAS ANALYSER",
                 bg="#192d42", fg="#ffffff",
                 font=("Helvetica", 20, "bold")).pack(anchor="w", padx=26, pady=(15, 1))
        tk.Label(header, text="Thermodynamic properties from cubic equations of state",
                 bg="#192d42", fg="#b9c9d8",
                 font=("Helvetica", 10)).pack(anchor="w", padx=27, pady=(0, 13))

        body = tk.Frame(self, bg="#f3f5f8")
        body.pack(fill=tk.BOTH, expand=True, padx=22, pady=17)

        # System definition card
        system = tk.Frame(body, bg="#ffffff", highlightbackground="#dce3ea",
                          highlightthickness=1)
        system.pack(fill=tk.X, pady=(0, 14))
        tk.Label(system, text="01   DEFINE THE SYSTEM", bg="#ffffff", fg="#64748b",
                 font=("Helvetica", 9, "bold")).grid(row=0, column=0, columnspan=5,
                                                     sticky="w", padx=18, pady=(13, 8))

        self.substance_var = tk.StringVar(value="Carbon Dioxide (CO2)")
        self.model_var = tk.StringVar(value="PR")
        self.t_var = tk.StringVar(value="320.0")
        self.p_var = tk.StringVar(value="50.0")

        fields = [
            ("PURE COMPONENT", "component"),
            ("CUBIC EOS", "model"),
            ("TEMPERATURE (K)", "temperature"),
            ("PRESSURE (bar)", "pressure"),
        ]
        for col, (label, _) in enumerate(fields):
            tk.Label(system, text=label, bg="#ffffff", fg="#526174",
                     font=("Helvetica", 8, "bold")).grid(row=1, column=col,
                                                         sticky="w", padx=18)
        self.substance_cb = ttk.Combobox(system, textvariable=self.substance_var,
            values=list(FLUID_DATA.keys()), state="readonly", width=23)
        self.substance_cb.grid(row=2, column=0, sticky="ew", padx=18, pady=(5, 15))
        self.substance_cb.bind("<<ComboboxSelected>>", lambda event: self.run_calculation())

        self.model_cb = ttk.Combobox(system, textvariable=self.model_var,
            values=["PR", "SRK", "RK", "VDW"], state="readonly", width=11)
        self.model_cb.grid(row=2, column=1, sticky="ew", padx=18, pady=(5, 15))
        self.model_cb.bind("<<ComboboxSelected>>", lambda event: self.run_calculation())

        ttk.Entry(system, textvariable=self.t_var, width=15).grid(
            row=2, column=2, sticky="ew", padx=18, pady=(5, 15))
        ttk.Entry(system, textvariable=self.p_var, width=15).grid(
            row=2, column=3, sticky="ew", padx=18, pady=(5, 15))

        ttk.Button(system, text="CALCULATE PROPERTIES",
                   command=self.run_calculation).grid(
            row=2, column=4, sticky="ew", padx=(10, 18), pady=(5, 15))
        for col in range(5):
            system.columnconfigure(col, weight=1 if col in (0, 2, 3, 4) else 0)

        # Summary cards
        tk.Label(body, text="02   PROPERTY SUMMARY", bg="#f3f5f8",
                 fg="#64748b", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(0, 8))
        cards = tk.Frame(body, bg="#f3f5f8")
        cards.pack(fill=tk.X, pady=(0, 14))
        self.summary_values = {}
        card_specs = [
            ("COMPRESSIBILITY FACTOR", "Z", "#e8f2f7"),
            ("MOLAR VOLUME", "Vm", "#edf4ed"),
            ("FUGACITY", "f", "#f7f0e7"),
            ("FUGACITY COEFFICIENT", "phi", "#f1edf7"),
        ]
        for col, (label, key, shade) in enumerate(card_specs):
            card = tk.Frame(cards, bg="#ffffff", highlightbackground="#dce3ea",
                            highlightthickness=1)
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
            tk.Label(card, text=label, bg="#ffffff", fg="#64748b",
                     font=("Helvetica", 8, "bold")).pack(anchor="w", padx=14, pady=(12, 5))
            value = tk.Label(card, text="—", bg="#ffffff", fg="#1e3449",
                             font=("Helvetica", 19, "bold"))
            value.pack(anchor="w", padx=14)
            unit = {"Z": "", "Vm": "cm³/mol", "f": "bar", "phi": ""}[key]
            tk.Label(card, text=unit, bg="#ffffff", fg="#8491a0",
                     font=("Helvetica", 8)).pack(anchor="w", padx=14, pady=(1, 12))
            self.summary_values[key] = value
            cards.columnconfigure(col, weight=1)

        # Analysis tabs: plots are no longer displayed as a permanent pair.
        tk.Label(body, text="03   DETAILED ANALYSIS", bg="#f3f5f8",
                 fg="#64748b", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(0, 7))
        analysis = tk.Frame(body, bg="#ffffff", highlightbackground="#dce3ea",
                            highlightthickness=1)
        analysis.pack(fill=tk.BOTH, expand=True)
        self.analysis_tabs = ttk.Notebook(analysis)
        self.analysis_tabs.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        pv_tab = tk.Frame(self.analysis_tabs, bg="#ffffff")
        fug_tab = tk.Frame(self.analysis_tabs, bg="#ffffff")
        hs_tab = tk.Frame(self.analysis_tabs, bg="#ffffff")
        self.analysis_tabs.add(pv_tab, text="P–V BEHAVIOUR")
        self.analysis_tabs.add(fug_tab, text="FUGACITY")
        self.analysis_tabs.add(hs_tab, text="ENTHALPY & ENTROPY")

        self.fig_pv = Figure(figsize=(8.4, 4.0), dpi=100, facecolor="white")
        self.ax_pv = self.fig_pv.add_subplot(111)
        self.canvas_pv = FigureCanvasTkAgg(self.fig_pv, master=pv_tab)
        self.canvas_pv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_fug = Figure(figsize=(8.4, 4.0), dpi=100, facecolor="white")
        self.ax_fug = self.fig_fug.add_subplot(111)
        self.canvas_fug = FigureCanvasTkAgg(self.fig_fug, master=fug_tab)
        self.canvas_fug.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.detail_text = tk.Text(hs_tab, wrap=tk.WORD, font=("Menlo", 10),
            bg="#fbfcfe", fg="#26364a", relief="flat", padx=22, pady=18)
        self.detail_text.pack(fill=tk.BOTH, expand=True)

        footer = tk.Label(body,
            text="Reference state for total H and S: 298.15 K, 1 bar     •     Pure-component calculations",
            bg="#f3f5f8", fg="#7b8794", font=("Helvetica", 8))
        footer.pack(anchor="w", pady=(8, 0))

    def run_calculation(self):
        try:
            T = float(self.t_var.get())
            P_bar = float(self.p_var.get())
            if T <= 0 or P_bar <= 0:
                raise ValueError("Temperature and pressure must be positive.")

            P = P_bar * 1.0e5
            substance = self.substance_var.get()
            model = self.model_var.get()
            eos = CubicModel(substance, model)
            roots, _, _ = eos.find_z_roots(T, P)

            if not roots:
                raise ValueError("No physically admissible compressibility-factor roots found.")

            self.display_results(eos, T, P, P_bar, roots)
            self.draw_plots(eos, T, P, roots)

        except ValueError as error:
            messagebox.showerror("Calculation/input error", str(error))
        except (OverflowError, ZeroDivisionError) as error:
            messagebox.showerror("Numerical error", str(error))

    def format_phase_block(self, eos, T, P, Z, phase_name):
        prop = eos.state_properties(T, P, Z)
        lines = [
            phase_name,
            f"Compressibility factor Z       : {prop['Z']:.6f}",
            f"Molar volume                   : {prop['Vm'] * 1e6:.4f} cm^3/mol",
            f"Fugacity coefficient           : {prop['phi']:.6f}",
            f"Pure-component fugacity        : {prop['f'] / 1e5:.6f} bar",
            "",
            f"Residual enthalpy H^R          : {prop['H_dep'] / 1000:.6f} kJ/mol",
            f"Residual entropy S^R           : {prop['S_dep']:.6f} J/(mol K)",
            f"Ideal-gas enthalpy H^ig        : {prop['H_ig'] / 1000:.6f} kJ/mol",
            f"Ideal-gas entropy S^ig         : {prop['S_ig']:.6f} J/(mol K)",
            f"Total enthalpy H               : {prop['H'] / 1000:.6f} kJ/mol",
            f"Total entropy S                : {prop['S']:.6f} J/(mol K)",
        ]
        return lines, prop

    def display_results(self, eos, T, P, P_bar, roots):
        # The summary cards show the highest-Z root, while the detail tab lists
        # every admissible root when the cubic has more than one solution.
        selected = eos.state_properties(T, P, roots[-1])
        self.summary_values["Z"].configure(text=f"{selected['Z']:.4f}")
        self.summary_values["Vm"].configure(text=f"{selected['Vm'] * 1e6:.2f}")
        self.summary_values["f"].configure(text=f"{selected['f'] / 1e5:.3f}")
        self.summary_values["phi"].configure(text=f"{selected['phi']:.4f}")

        lines = [
            f"Component       : {eos.substance}",
            f"Equation of state: {eos.model}",
            f"Temperature     : {T:.3f} K",
            f"Pressure        : {P_bar:.4f} bar",
            f"Critical data   : Tc = {eos.Tc:.3f} K, Pc = {eos.Pc / 1e5:.4f} bar",
            f"Acentric factor : {eos.omega:.5f}",
            "",
            "REFERENCE STATE",
            f"Tref = {T_REF:.2f} K, Pref = 1.000 bar",
            "Ideal-gas heat capacity: Cp = a + bT + cT^2 + dT^3",
            "",
        ]

        if len(roots) == 1:
            phase_lines, _ = self.format_phase_block(
                eos, T, P, roots[0], "SINGLE ADMISSIBLE ROOT"
            )
            lines.extend(phase_lines)
        else:
            liquid_lines, liquid = self.format_phase_block(
                eos, T, P, roots[0], "LOWEST ROOT (LIQUID-LIKE)"
            )
            vapor_lines, vapor = self.format_phase_block(
                eos, T, P, roots[-1], "HIGHEST ROOT (VAPOR-LIKE)"
            )
            lines.extend(liquid_lines)
            lines.extend(["", "-" * 68, ""])
            lines.extend(vapor_lines)
            lines.extend([
                "",
                f"Fugacity ratio fL/fV : {liquid['f'] / vapor['f']:.6f}",
                "Multiple roots alone do not establish vapor-liquid equilibrium."
            ])

        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, "\n".join(lines))

    # The first graph is the P-V isotherm; the second compares fugacity with P.
    def draw_plots(self, eos, T, P, roots):
        self.ax_pv.clear()
        v_min = eos.b * 1.01
        v_max = max(R * T / 1.0e5 * 2.0, v_min * 10.0)
        volumes, pressures = [], []

        for i in range(350):
            fraction = i / 349.0
            vm = math.exp(math.log(v_min) + fraction * (math.log(v_max) - math.log(v_min)))
            p_model = eos.pressure_from_volume(T, vm)
            if p_model is not None and p_model <= 2.5 * eos.Pc:
                volumes.append(vm * 1e6)
                pressures.append(p_model / 1e5)

        self.ax_pv.plot(volumes, pressures, color="#167d8d", lw=2.4,
                        label=f"{eos.model} isotherm")
        self.ax_pv.set_xscale("log")
        self.ax_pv.set_xlabel("Molar volume [cm³/mol]")
        self.ax_pv.set_ylabel("Pressure [bar]")
        self.ax_pv.set_title(f"P-V isotherm at T = {T:.2f} K")
        self.ax_pv.grid(True, which="both", linestyle=":", alpha=0.55)
        for index, z in enumerate(roots):
            vm = z * R * T / P
            self.ax_pv.scatter(vm * 1e6, P / 1e5, s=65, zorder=5,
                               label=f"Root {index + 1}: Z={z:.4f}")
        self.ax_pv.legend(fontsize=9)
        self.fig_pv.tight_layout(pad=2.2)
        self.canvas_pv.draw()

        self.ax_fug.clear()
        p_max_bar = max(1.8 * eos.Pc / 1e5, P / 1e5 * 1.1)
        p_values, f_values = [], []
        for i in range(1, 161):
            p_bar = p_max_bar * i / 160
            p_pa = p_bar * 1e5
            try:
                z_roots, _, _ = eos.find_z_roots(T, p_pa)
                if z_roots:
                    prop = eos.state_properties(T, p_pa, z_roots[-1])
                    p_values.append(p_bar)
                    f_values.append(prop["f"] / 1e5)
            except (ValueError, OverflowError, ZeroDivisionError):
                continue

        self.ax_fug.plot(p_values, f_values, color="#167d8d", lw=2.4,
                         label="Real-gas fugacity")
        self.ax_fug.plot(p_values, p_values, color="#e18b38", ls="--",
                         label="Ideal gas: f = P")
        self.ax_fug.axvline(P / 1e5, color="#64748b", ls=":",
                            label="Input pressure")
        self.ax_fug.set_xlabel("Pressure [bar]")
        self.ax_fug.set_ylabel("Pure-component fugacity [bar]")
        self.ax_fug.set_title("Pure-component fugacity vs pressure")
        self.ax_fug.grid(True, linestyle=":", alpha=0.55)
        self.ax_fug.legend(fontsize=9)
        self.fig_fug.tight_layout(pad=2.2)
        self.canvas_fug.draw()


if __name__ == "__main__":
    app = PropertyApp()
    app.mainloop()
