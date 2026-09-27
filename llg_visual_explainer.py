"""
===============================================================================
Landau-Lifshitz-Gilbert (LLG) Equation 3D Visual Explainer
===============================================================================

An interactive, high-quality 3D Python GUI application for learning spintronics
and magnetization dynamics.

THE PHYSICS OF THE LLG EQUATION:
--------------------------------
The Landau-Lifshitz-Gilbert (LLG) equation describes the time-dependent
precession and relaxation of a normalized macrospin magnetization vector M
|M| = Ms = 1 under an effective magnetic field H_eff.

1. Implicit Gilbert Form:
   dM/dt = -γ (M × H_eff) + (α / Ms) (M × dM/dt)

   - Precession term: -γ (M × H_eff)
     Perpendicular to both M and H_eff. Causes M to precess in a circle around
     H_eff at constant polar angle, conserving magnetic energy.
   - Damping term: (α / Ms) (M × dM/dt)
     Phenomenological dissipation term introduced by T. L. Gilbert. Pulls M
     towards alignment with H_eff.

2. Explicit Landau-Lifshitz Form (used for numerical integration):
   dM/dt = - γ / (1 + α²) (M × H_eff) - γ α / ((1 + α²) Ms) [M × (M × H_eff)]

   - First term: Precession torque scaled by 1 / (1 + α²).
   - Second term: Damping torque acting perpendicular to M and (M × H_eff),
     directly pulling M towards H_eff along the shortest path.

THE FOUR CONCEPTUAL SCENARIOS DEMONSTRATED:
-------------------------------------------
1. "Before field applied" (H_eff = 0):
   M remains static at its initial orientation. Demonstrates baseline equilibrium.
2. "Precession only (α = 0)":
   Damping is set to zero. M executes a closed circular orbit around H_eff forever.
3. "Damping only (conceptual)":
   Precession torque is set to zero. M relaxes directly towards H_eff along a
   meridian arc (great circle). Note: This is a conceptual isolation of the damping
   torque component, not a physically independent real-world state.
4. "Full LLG (precession + damping)":
   The full equation in action. M spirals around H_eff while gradually relaxing
   into alignment along H_eff.

DEVELOPMENT & DESIGN PRIORITIES:
--------------------------------
- Designed for spintronics students & researchers.
- Visual clarity and physical accuracy are top priorities.
- Uses scipy.integrate.solve_ivp (RK45) for precomputing smooth trajectories.
- Native Tkinter GUI + Matplotlib 3D graphics canvas with custom shading,
  arc-length color-gradient trajectory lines, anti-aliasing, and clean typography.
===============================================================================
"""

import sys
import math
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import tkinter as tk
from tkinter import ttk, messagebox


# =============================================================================
# NAMED CONSTANTS & PHYSICAL PARAMETERS
# =============================================================================
GAMMA = 1.0           # Gyromagnetic ratio (normalized for visual clarity)
ALPHA_DEFAULT = 0.15  # Gilbert damping factor for Full LLG scenario
MS = 1.0              # Saturation magnetization magnitude |M| = 1
HEFF_MAG = 1.0        # Magnitude of effective field along +z axis

# Initial orientation of magnetization vector M (in spherical coordinates)
THETA_0 = np.radians(60.0)  # Polar angle from +z axis (60 degrees)
PHI_0 = np.radians(0.0)     # Azimuthal angle (0 degrees)

# Visual & Animation Timing Parameters
ANIMATION_DURATION_SECONDS = 9.0  # Total animation duration (~9 seconds)
ANIMATION_FPS = 40                # Target frames per second
ANIMATION_INTERVAL_MS = int(1000 / ANIMATION_FPS)  # Frame interval in milliseconds (25 ms)
NUM_ANIM_FRAMES = int(ANIMATION_DURATION_SECONDS * ANIMATION_FPS)  # Total frames (360)

TRAJECTORY_COLORMAP = "magma"     # Perceptually uniform colormap ('magma', 'cividis', 'cool')
COLORMAP_RANGE = (0.12, 0.98)     # Truncated range to ensure vibrant contrast on dark background
MESH_RESOLUTION = 70            # Resolution of the background reference sphere mesh
DEFAULT_ELEV = 22               # Default camera elevation angle
DEFAULT_AZIM = 45               # Default camera azimuth angle


# =============================================================================
# NUMERICAL ODE RIGHT-HAND-SIDE (RHS) FUNCTIONS
# =============================================================================

def llg_rhs_precession_only(t, M, gamma, H_eff_vec):
    """
    RHS for Precession Only (alpha = 0).
    dM/dt = -γ (M × H_eff)
    """
    dM_dt = -gamma * np.cross(M, H_eff_vec)
    return dM_dt


def llg_rhs_damping_only(t, M, gamma, alpha, Ms, H_eff_vec):
    """
    RHS for Damping Only (conceptual isolation of the damping term).
    dM/dt = - (γ α / Ms) [M × (M × H_eff)]
    """
    m_cross_h = np.cross(M, H_eff_vec)
    dM_dt = - (gamma * alpha / Ms) * np.cross(M, m_cross_h)
    return dM_dt


def llg_rhs_full(t, M, gamma, alpha, Ms, H_eff_vec):
    """
    RHS for Full Explicit LLG equation.
    dM/dt = - [γ / (1 + α²)] (M × H_eff) - [γ α / ((1 + α²) Ms)] [M × (M × H_eff)]
    """
    denom = 1.0 + alpha**2
    m_cross_h = np.cross(M, H_eff_vec)
    m_cross_m_cross_h = np.cross(M, m_cross_h)
    
    term_precession = - (gamma / denom) * m_cross_h
    term_damping = - (gamma * alpha / (denom * Ms)) * m_cross_m_cross_h
    
    return term_precession + term_damping


def compute_trajectory(scenario_key, alpha=ALPHA_DEFAULT, num_frames=NUM_ANIM_FRAMES):
    """
    Precompute trajectory array (N x 3) using RK45 solver for a given scenario.
    Returns: (t_eval, M_trajectory)
    """
    # Initial vector M0
    Mx0 = MS * np.sin(THETA_0) * np.cos(PHI_0)
    My0 = MS * np.sin(THETA_0) * np.sin(PHI_0)
    Mz0 = MS * np.cos(THETA_0)
    M0 = np.array([Mx0, My0, Mz0], dtype=float)

    H_eff_vec = np.array([0.0, 0.0, HEFF_MAG], dtype=float)

    if scenario_key == "before":
        # Static orientation over time
        t_eval = np.linspace(0, 10, num_frames)
        M_traj = np.tile(M0, (num_frames, 1))
        return t_eval, M_traj

    elif scenario_key == "precession":
        # Extended duration: run precession for 7 full orbits so motion is smooth and detailed
        t_max = 7.0 * (2.0 * np.pi / (GAMMA * HEFF_MAG))
        t_eval = np.linspace(0, t_max, num_frames)
        
        sol = solve_ivp(
            fun=lambda t, y: llg_rhs_precession_only(t, y, GAMMA, H_eff_vec),
            t_span=(0, t_max),
            y0=M0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )
        M_traj = sol.y.T

    elif scenario_key == "damping":
        # Extended duration: smooth relaxation along meridian arc towards +z
        t_max = 40.0
        t_eval = np.linspace(0, t_max, num_frames)
        
        sol = solve_ivp(
            fun=lambda t, y: llg_rhs_damping_only(t, y, GAMMA, alpha, MS, H_eff_vec),
            t_span=(0, t_max),
            y0=M0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )
        M_traj = sol.y.T

    elif scenario_key == "full":
        # Extended duration: full spiral relaxation until M settles almost completely onto H_eff (+z)
        t_max = 50.0
        t_eval = np.linspace(0, t_max, num_frames)
        
        sol = solve_ivp(
            fun=lambda t, y: llg_rhs_full(t, y, GAMMA, alpha, MS, H_eff_vec),
            t_span=(0, t_max),
            y0=M0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )
        M_traj = sol.y.T

    else:
        raise ValueError(f"Unknown scenario key: {scenario_key}")

    # Explicitly enforce unit norm |M| = Ms to eliminate numerical drift
    norms = np.linalg.norm(M_traj, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M_traj = (M_traj / norms) * MS

    return t_eval, M_traj


# =============================================================================
# SCENARIO DESCRIPTIONS & METADATA
# =============================================================================
SCENARIOS = {
    "before": {
        "title": r"Scenario 1: Before Field Applied ($\mathbf{H}_{\mathrm{eff}} = 0$)",
        "title_tk": "Scenario 1: Before Field Applied (H_eff = 0)",
        "button_text": "1. Before field applied",
        "formula": r"$\mathbf{H}_{\mathrm{eff}} = 0 \implies \frac{d\mathbf{M}}{dt} = 0$",
        "formula_tk": "dM/dt = 0  (H_eff = 0)",
        "explanation": (
            "No effective magnetic field is present (H_eff = 0). The magnetization "
            "vector M remains statically fixed at its initial orientation (θ = 60°). "
            "This serves as the baseline equilibrium state before magnetic excitation."
        ),
        "colormap": "magma",
        "colormap_range": (0.12, 0.98)
    },
    "precession": {
        "title": r"Scenario 2: Precession Only ($\alpha = 0$)",
        "title_tk": "Scenario 2: Precession Only (α = 0)",
        "button_text": "2. Precession only (α = 0)",
        "formula": r"$\frac{d\mathbf{M}}{dt} = -\gamma (\mathbf{M} \times \mathbf{H}_{\mathrm{eff}})$",
        "formula_tk": "dM/dt = -γ (M × H_eff)",
        "explanation": (
            "Damping is set to zero (α = 0). The magnetic torque -γ(M × H_eff) acts "
            "perpendicular to both M and H_eff. This forces M into a continuous, "
            "conservative circular precession around H_eff without ever losing energy or aligning."
        ),
        "colormap": "cool",
        "colormap_range": (0.15, 1.0)
    },
    "damping": {
        "title": r"Scenario 3: Damping Only (Conceptual)",
        "title_tk": "Scenario 3: Damping Only (Conceptual)",
        "button_text": "3. Damping only (conceptual)",
        "formula": r"$\frac{d\mathbf{M}}{dt} = -\frac{\gamma \alpha}{M_s} \mathbf{M} \times (\mathbf{M} \times \mathbf{H}_{\mathrm{eff}})$",
        "formula_tk": "dM/dt = -(γ α / Ms) [M × (M × H_eff)]",
        "explanation": (
            "Conceptual isolation of the Gilbert damping term without precession. The damping torque "
            "pulls M directly along a great-circle arc toward H_eff (+z axis). Note: This isolates "
            "one mathematical term for intuition—it is not a physically separate real-world state."
        ),
        "colormap": "plasma",
        "colormap_range": (0.32, 0.98)
    },
    "full": {
        "title": r"Scenario 4: Full LLG (Precession + Damping)",
        "title_tk": "Scenario 4: Full LLG (Precession + Damping)",
        "button_text": "4. Full LLG (precession + damping)",
        "formula": (
            r"$\frac{d\mathbf{M}}{dt} = -\frac{\gamma}{1+\alpha^2}(\mathbf{M}\times\mathbf{H}_{\mathrm{eff}}) "
            r"- \frac{\gamma\alpha}{(1+\alpha^2)M_s}\mathbf{M}\times(\mathbf{M}\times\mathbf{H}_{\mathrm{eff}})$"
        ),
        "formula_tk": "dM/dt = -[γ / (1+α²)] (M × H_eff) - [γ α / ((1+α²) Ms)] [M × (M × H_eff)]",
        "explanation": (
            "The full Landau-Lifshitz-Gilbert equation in action. The precession torque causes M to "
            "rotate around H_eff, while the Gilbert damping torque simultaneously pulls M inward. "
            "The result is a realistic spiral path decaying into alignment with H_eff along +z."
        ),
        "colormap": "plasma",
        "colormap_range": (0.32, 0.98)
    }
}


# =============================================================================
# MAIN GUI & ANIMATION APPLICATION CLASS
# =============================================================================

class LLGExplainerApp:
    """
    Main Tkinter desktop application encapsulating the 3D Matplotlib canvas,
    the animation loop, theme controls, and educational explanations.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Landau-Lifshitz-Gilbert (LLG) Equation Visual Explainer")
        self.root.geometry("1280x860")
        self.root.minsize(1024, 720)

        # Apply modern dark slate visual theme palette
        self.bg_dark = "#1E222A"
        self.bg_card = "#252B37"
        self.bg_plot = "#181B22"
        self.fg_text = "#E6EDF3"
        self.fg_muted = "#8B949E"
        self.accent_blue = "#2F81F7"
        self.accent_green = "#238636"
        self.accent_orange = "#D97706"
        self.accent_red = "#DA3633"

        self.root.configure(bg=self.bg_dark)

        # Animation & View state attributes
        self.current_scenario_key = "before"
        self.current_frame = 0
        self.animation_job = None
        self.is_animating = False
        self.current_zoom_limit = 1.3  # Dynamic 3D view zoom boundary limit

        self.t_eval = None
        self.M_traj = None
        self.traj_colors = None

        # Build UI layout
        self._setup_styles()
        self._build_gui_layout()

        # Initialize plot with Scenario 1
        self.select_scenario("before")

    def _setup_styles(self):
        """Configure ttk styles for a modern, polished window look."""
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=self.bg_dark)
        style.configure("Card.TFrame", background=self.bg_card, relief="flat")

        style.configure(
            "Header.TLabel",
            background=self.bg_dark,
            foreground=self.fg_text,
            font=("Segoe UI", 16, "bold")
        )
        style.configure(
            "SubHeader.TLabel",
            background=self.bg_dark,
            foreground=self.fg_muted,
            font=("Segoe UI", 10)
        )
        style.configure(
            "CardTitle.TLabel",
            background=self.bg_card,
            foreground=self.accent_blue,
            font=("Segoe UI", 12, "bold")
        )
        style.configure(
            "Formula.TLabel",
            background=self.bg_card,
            foreground="#A5D6FF",
            font=("Consolas", 11, "italic")
        )
        style.configure(
            "Explanation.TLabel",
            background=self.bg_card,
            foreground=self.fg_text,
            font=("Segoe UI", 10),
            wraplength=480
        )

    def _build_gui_layout(self):
        """Construct side panel, control buttons, status cards, and 3D canvas."""
        # Top Header Bar
        header_frame = ttk.Frame(self.root, padding=(20, 15, 20, 10))
        header_frame.pack(fill="x", side="top")

        ttk.Label(
            header_frame,
            text="Landau-Lifshitz-Gilbert (LLG) Equation 3D Visual Explainer",
            style="Header.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            header_frame,
            text="An interactive theoretical visual guide to magnetization dynamics in spintronics",
            style="SubHeader.TLabel"
        ).pack(anchor="w", pady=(2, 0))

        # Main Content Container (Left controls + Right 3D canvas)
        main_container = ttk.Frame(self.root, padding=(20, 10, 20, 20))
        main_container.pack(fill="both", expand=True, side="top")

        # Left Control & Explanation Panel
        left_panel = ttk.Frame(main_container, width=420)
        left_panel.pack(side="left", fill="y", padx=(0, 15))
        left_panel.pack_propagate(False)

        # Button Group Box
        btn_card = ttk.Frame(left_panel, style="Card.TFrame", padding=15)
        btn_card.pack(fill="x", pady=(0, 15))

        ttk.Label(
            btn_card,
            text="SELECT CONCEPTUAL SCENARIO",
            style="CardTitle.TLabel"
        ).pack(anchor="w", pady=(0, 12))

        self.buttons = {}
        for key, sc in SCENARIOS.items():
            btn = tk.Button(
                btn_card,
                text=sc["button_text"],
                font=("Segoe UI", 10, "bold"),
                bg="#2D333B",
                fg=self.fg_text,
                activebackground=self.accent_blue,
                activeforeground="#FFFFFF",
                bd=0,
                padx=12,
                pady=10,
                anchor="w",
                cursor="hand2",
                command=lambda k=key: self.select_scenario(k)
            )
            btn.pack(fill="x", pady=4)
            self.buttons[key] = btn

        # Animation Controls (Replay / Pause)
        ctrl_frame = tk.Frame(btn_card, bg=self.bg_card)
        ctrl_frame.pack(fill="x", pady=(10, 0))

        self.btn_replay = tk.Button(
            ctrl_frame,
            text="🔄 Replay Animation",
            font=("Segoe UI", 9, "bold"),
            bg=self.accent_blue,
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2",
            command=self.replay_animation
        )
        self.btn_replay.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_pause = tk.Button(
            ctrl_frame,
            text="⏸ Pause / Resume",
            font=("Segoe UI", 9, "bold"),
            bg="#374151",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2",
            command=self.toggle_pause
        )
        self.btn_pause.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # 3D View Zoom Controls (Zoom In / Zoom Out / Reset View)
        zoom_frame = tk.Frame(btn_card, bg=self.bg_card)
        zoom_frame.pack(fill="x", pady=(8, 0))

        self.btn_zoom_in = tk.Button(
            zoom_frame,
            text="🔍+ Zoom In",
            font=("Segoe UI", 9, "bold"),
            bg="#374151",
            fg="#FFFFFF",
            bd=0,
            padx=6,
            pady=5,
            cursor="hand2",
            command=self.zoom_in
        )
        self.btn_zoom_in.pack(side="left", expand=True, fill="x", padx=(0, 2))

        self.btn_zoom_out = tk.Button(
            zoom_frame,
            text="🔍- Zoom Out",
            font=("Segoe UI", 9, "bold"),
            bg="#374151",
            fg="#FFFFFF",
            bd=0,
            padx=6,
            pady=5,
            cursor="hand2",
            command=self.zoom_out
        )
        self.btn_zoom_out.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_zoom_reset = tk.Button(
            zoom_frame,
            text="🎯 Reset View",
            font=("Segoe UI", 9, "bold"),
            bg="#2D333B",
            fg=self.fg_text,
            bd=0,
            padx=6,
            pady=5,
            cursor="hand2",
            command=self.reset_view
        )
        self.btn_zoom_reset.pack(side="right", expand=True, fill="x", padx=(2, 0))

        # Scenario Explanation Card
        self.exp_card = ttk.Frame(left_panel, style="Card.TFrame", padding=15)
        self.exp_card.pack(fill="both", expand=True)
        self.exp_card.pack(fill="both", expand=True)

        self.lbl_card_title = ttk.Label(
            self.exp_card,
            text="SCENARIO EXPLANATION",
            style="CardTitle.TLabel"
        )
        self.lbl_card_title.pack(anchor="w", pady=(0, 8))

        self.lbl_sc_title = ttk.Label(
            self.exp_card,
            text="",
            font=("Segoe UI", 11, "bold"),
            background=self.bg_card,
            foreground="#58A6FF"
        )
        self.lbl_sc_title.pack(anchor="w", pady=(0, 6))

        # Text Explanation Widget with smooth auto-wrap
        self.lbl_exp_text = ttk.Label(
            self.exp_card,
            text="",
            style="Explanation.TLabel"
        )
        self.lbl_exp_text.pack(anchor="w", fill="x", expand=True, pady=(0, 10))

        # Legend & Color Coding Reference Card
        legend_frame = tk.Frame(self.exp_card, bg="#1C2128", bd=1, relief="solid")
        legend_frame.pack(fill="x", side="bottom", pady=(10, 0), ipady=6, ipadx=8)

        tk.Label(
            legend_frame,
            text="3D VISUAL LEGEND:",
            font=("Segoe UI", 8, "bold"),
            bg="#1C2128",
            fg=self.fg_muted
        ).pack(anchor="w", padx=6, pady=(4, 2))

        items = [
            ("🟢 Solid Green Arrow", "Effective Magnetic Field H_eff (+z)"),
            ("🟠 Glowing Orange Vector", "Live Magnetization Vector M(t)"),
            ("🔥 Magma Arc Gradient", "Continuous arc-length trajectory (t=0 to t_final)"),
            ("🟢 Green Point / 🔴 Red Point", "Initial Position (t=0) / Final Position")
        ]
        for symbol, desc in items:
            row = tk.Frame(legend_frame, bg="#1C2128")
            row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=symbol, font=("Segoe UI", 8, "bold"), bg="#1C2128", fg="#E6EDF3").pack(side="left")
            tk.Label(row, text=f" : {desc}", font=("Segoe UI", 8), bg="#1C2128", fg=self.fg_muted).pack(side="left")

        # Right Panel - Matplotlib 3D Figure Container
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side="right", fill="both", expand=True)

        # Matplotlib Dark Theme Figure setup
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(9, 7.5), facecolor=self.bg_plot, dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor=self.bg_plot)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Connect mouse scroll wheel listener for interactive 3D zooming
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def select_scenario(self, scenario_key):
        """Switch active scenario, stop existing animation, and start new one."""
        self._stop_animation()

        self.current_scenario_key = scenario_key
        self.current_frame = 0
        self.is_animating = True

        # Highlight active button
        for k, btn in self.buttons.items():
            if k == scenario_key:
                btn.config(bg=self.accent_blue, fg="#FFFFFF")
            else:
                btn.config(bg="#2D333B", fg=self.fg_text)

        # Update explanation panel text
        sc = SCENARIOS[scenario_key]
        self.lbl_sc_title.config(text=sc.get("title_tk", sc["title"]))
        self.lbl_exp_text.config(text=sc["explanation"])

        # Compute numerical ODE trajectory
        self.t_eval, self.M_traj = compute_trajectory(scenario_key)

        # Compute continuous cumulative arc-length along the 3D trajectory path
        # This provides a smooth, un-quantized spatial color parameterization.
        N = len(self.M_traj)
        if N > 1:
            ds = np.linalg.norm(np.diff(self.M_traj, axis=0), axis=1)
            s_cum = np.concatenate([[0.0], np.cumsum(ds)])
            total_s = s_cum[-1]
            
            if total_s > 1e-12:
                s_norm = s_cum / total_s
            else:
                s_norm = np.linspace(0.0, 1.0, N)

            # Midpoint parameter for each segment
            s_mid = 0.5 * (s_norm[:-1] + s_norm[1:])
        else:
            s_mid = np.array([0.0])

        # Map midpoint arc-length parameter continuously to the scenario colormap range
        sc_meta = SCENARIOS[scenario_key]
        cmap_name = sc_meta.get("colormap", TRAJECTORY_COLORMAP)
        c_min, c_max = sc_meta.get("colormap_range", COLORMAP_RANGE)
        
        color_sample_vals = c_min + s_mid * (c_max - c_min)

        cmap = plt.get_cmap(cmap_name)
        self.traj_colors = cmap(color_sample_vals)

        # Initial render & start animation frame loop
        self._render_static_background()
        self._animate_step()

    def _render_static_background(self):
        """Draw unit reference sphere, coordinate axes, and H_eff vector."""
        self.ax.clear()
        self.ax.set_facecolor(self.bg_plot)

        # 1. High-resolution shaded unit reference sphere (|M| = 1)
        # Slightly muted surface color/alpha for optimal trajectory contrast
        u = np.linspace(0, 2 * np.pi, MESH_RESOLUTION)
        v = np.linspace(0, np.pi, MESH_RESOLUTION)
        x_sp = np.outer(np.cos(u), np.sin(v))
        y_sp = np.outer(np.sin(u), np.sin(v))
        z_sp = np.outer(np.ones(np.size(u)), np.cos(v))

        # Render translucent sphere surface with smooth shading
        self.ax.plot_surface(
            x_sp, y_sp, z_sp,
            color="#2A3440",
            alpha=0.14,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True
        )

        # Wireframe equator and prime meridian circles for spatial depth
        theta_line = np.linspace(0, 2 * np.pi, 100)
        self.ax.plot(
            np.cos(theta_line), np.sin(theta_line), np.zeros_like(theta_line),
            color="#4E5D6C", alpha=0.25, linestyle="--", linewidth=1.0
        )
        self.ax.plot(
            np.cos(theta_line), np.zeros_like(theta_line), np.sin(theta_line),
            color="#4E5D6C", alpha=0.20, linestyle=":", linewidth=1.0
        )

        # 2. Bold Effective Field Arrow H_eff along +z (omitted in 'before' scenario)
        if self.current_scenario_key != "before":
            self.ax.quiver(
                0, 0, 0,
                0, 0, 1.45,
                color="#00E676",
                linewidth=3.8,
                arrow_length_ratio=0.12,
                alpha=0.95,
                pivot="tail"
            )
            self.ax.text(
                0.05, 0.05, 1.55,
                r"$\mathbf{H}_{\mathrm{eff}}$",
                color="#00E676",
                fontsize=13,
                fontweight="bold"
            )
        else:
            # For scenario 1 ('before'), display clear 3D LaTeX text annotation indicating H_eff = 0
            self.ax.text(
                0.05, 0.05, 1.55,
                r"$\mathbf{H}_{\mathrm{eff}} = 0$",
                color="#8B949E",
                fontsize=12,
                fontweight="bold"
            )

        # 3. Initial Position Marker (Green dot)
        M0 = self.M_traj[0]
        self.ax.scatter(
            [M0[0]], [M0[1]], [M0[2]],
            color="#00E676",
            s=80,
            edgecolors="black",
            linewidth=1.2,
            depthshade=False,
            zorder=10
        )
        self.ax.text(
            M0[0] * 1.12, M0[1] * 1.12, M0[2] * 1.12,
            r"$\mathbf{M}_0$",
            color="#00E676",
            fontsize=11,
            fontweight="bold"
        )

        # 4. Axes limits, labels, camera orientation & clean styling
        limit = self.current_zoom_limit
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])

        self.ax.set_xlabel("X", color=self.fg_muted, fontsize=10, labelpad=5)
        self.ax.set_ylabel("Y", color=self.fg_muted, fontsize=10, labelpad=5)
        self.ax.set_zlabel("Z", color=self.fg_muted, fontsize=10, labelpad=5)

        # Set optimal default camera perspective angle
        self.ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)

        # Custom grid line styling
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor(self.bg_plot)
        self.ax.yaxis.pane.set_edgecolor(self.bg_plot)
        self.ax.zaxis.pane.set_edgecolor(self.bg_plot)
        self.ax.grid(True, color="#2D333B", linestyle=":", linewidth=0.6, alpha=0.5)

        # Dynamic plot title
        self.ax.set_title(
            SCENARIOS[self.current_scenario_key]["title"],
            color=self.fg_text,
            fontsize=12,
            pad=12,
            fontweight="bold"
        )

        # Interactive elements placeholders
        self.quiver_M = None
        self.tip_marker = None
        self.glow_collection = None
        self.line_collection = None
        self.final_marker = None

    def _animate_step(self):
        """Frame update step driven by Tkinter event loop `after()`."""
        if not self.is_animating or self.M_traj is None:
            return

        idx = self.current_frame
        N = len(self.M_traj)

        # 1. Update Trajectory Curve with Continuous Perceptually Smooth Color Gradient
        if idx > 1:
            # Clean up previous collections to avoid memory or rendering leaks
            if self.line_collection is not None:
                try:
                    self.line_collection.remove()
                except Exception:
                    pass
            if self.glow_collection is not None:
                try:
                    self.glow_collection.remove()
                except Exception:
                    pass

            pts = self.M_traj[:idx].reshape(-1, 1, 3)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            sub_colors = self.traj_colors[:idx - 1]

            # Dual-pass line rendering for soft glowing depth and crisp anti-aliased trajectory:
            # Layer A: Soft background glow underlay (wider & semi-transparent)
            self.glow_collection = Line3DCollection(
                segments,
                colors=sub_colors,
                linewidth=6.5,
                alpha=0.24,
                antialiased=True
            )
            self.ax.add_collection(self.glow_collection)

            # Layer B: Main sharp crisp trajectory line (soft alpha blending = 0.80)
            self.line_collection = Line3DCollection(
                segments,
                colors=sub_colors,
                linewidth=3.0,
                alpha=0.80,
                antialiased=True
            )
            self.ax.add_collection(self.line_collection)

        # 2. Update Live Magnetization Vector Arrow M(t)
        if self.quiver_M is not None:
            try:
                self.quiver_M.remove()
            except Exception:
                pass

        Mx, My, Mz = self.M_traj[idx]
        self.quiver_M = self.ax.quiver(
            0, 0, 0,
            Mx, My, Mz,
            color="#FF5722",
            linewidth=4.0,
            arrow_length_ratio=0.14,
            pivot="tail",
            zorder=12
        )

        # 3. Update Live Tip Point Marker (Complementary luminous gold/yellow dot)
        if self.tip_marker is not None:
            try:
                self.tip_marker.remove()
            except Exception:
                pass

        self.tip_marker = self.ax.scatter(
            [Mx], [My], [Mz],
            color="#FFEA00",
            s=95,
            edgecolors="#181B22",
            linewidth=1.4,
            depthshade=False,
            zorder=15
        )

        # 4. Red Final Position Marker upon completion
        if idx == N - 1 and self.current_scenario_key != "before":
            if self.final_marker is not None:
                try:
                    self.final_marker.remove()
                except Exception:
                    pass
            self.final_marker = self.ax.scatter(
                [Mx], [My], [Mz],
                color="#E91E63",
                s=110,
                marker="*",
                edgecolors="white",
                linewidth=1.5,
                depthshade=False,
                zorder=16
            )
            self.ax.text(
                Mx * 1.15, My * 1.15, Mz * 1.15,
                r"$\mathbf{M}_{\mathrm{final}}$",
                color="#E91E63",
                fontsize=11,
                fontweight="bold"
            )

        # Refresh canvas
        self.canvas.draw_idle()

        # Advance frame or stop loop at end
        if idx < N - 1:
            self.current_frame += 1
            self.animation_job = self.root.after(ANIMATION_INTERVAL_MS, self._animate_step)
        else:
            self.is_animating = False

    def _stop_animation(self):
        """Cancel ongoing Tkinter `after` animation timer."""
        self.is_animating = False
        if self.animation_job is not None:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None

    def replay_animation(self):
        """Restart current scenario animation from t=0."""
        self.select_scenario(self.current_scenario_key)

    def toggle_pause(self):
        """Pause or resume the animation frame loop."""
        if self.is_animating:
            self._stop_animation()
            self.btn_pause.config(text="▶ Resume", bg=self.accent_green)
        else:
            if self.M_traj is not None and self.current_frame < len(self.M_traj) - 1:
                self.is_animating = True
                self.btn_pause.config(text="⏸ Pause", bg="#374151")
                self._animate_step()

    def zoom_in(self, step=0.15):
        """Zoom into the 3D plot scene."""
        self.current_zoom_limit = max(0.4, self.current_zoom_limit - step)
        self._apply_zoom_limits()

    def zoom_out(self, step=0.15):
        """Zoom out of the 3D plot scene."""
        self.current_zoom_limit = min(3.5, self.current_zoom_limit + step)
        self._apply_zoom_limits()

    def reset_view(self):
        """Reset camera zoom and orientation angles to defaults."""
        self.current_zoom_limit = 1.3
        self.ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
        self._apply_zoom_limits()

    def _apply_zoom_limits(self):
        """Enforce current zoom limit bounds on 3D plot axes."""
        lim = self.current_zoom_limit
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([-lim, lim])
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """Handle mouse scroll wheel events for dynamic interactive 3D zooming."""
        if event.inaxes == self.ax:
            if event.button == "up":
                self.zoom_in(step=0.10)
            elif event.button == "down":
                self.zoom_out(step=0.10)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Instantiate and run the Landau-Lifshitz-Gilbert desktop application."""
    root = tk.Tk()

    # Center window on desktop screen
    root.update_idletasks()
    width = 1280
    height = 860
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")

    app = LLGExplainerApp(root)

    # Handle window close cleanly
    def on_closing():
        app._stop_animation()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
