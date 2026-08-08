#
# This file is part of the CEMD distribution
# Copyright (c) 2022-2026 Jérôme Claverie.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PySide6 import QtCore, QtWidgets
from scipy.ndimage import gaussian_filter1d

from cemd.analysis.rdf import compute_rdf
from cemd.analysis.silicates import analyze_silicates

from .base_dialog import BaseBuilderDialog


class RDFDialog(BaseBuilderDialog):
    def __init__(self, parent, system):
        # Using the BaseBuilderDialog constructor (parent, title, width)
        super().__init__(parent, "Structural analysis", 900)
        self.setMinimumHeight(750)

        self.system = system
        self.u = system.to_mda()
        self.combo_list = system.atom_types

        self.current_r = None
        self.current_gr_raw = None

        self.setup_ui()

    def setup_ui(self):
        # Using BaseBuilderDialog's main_layout
        main_layout = self.main_layout

        # ---PANEL CONFIGURATION (3 Columns) ---
        config_group = QtWidgets.QGroupBox("Settings")
        config_layout = QtWidgets.QGridLayout()

        # Column 0 & 1: Type A
        config_layout.addWidget(QtWidgets.QLabel("Atom type A :"), 0, 0)
        self.combo_a = QtWidgets.QComboBox()
        self.combo_a.addItem("all")
        self.combo_a.insertSeparator(1)
        self.combo_a.addItems(self.combo_list)
        config_layout.addWidget(self.combo_a, 0, 1)

        # Column 2 & 3: Type B
        config_layout.addWidget(QtWidgets.QLabel("Atom type B :"), 0, 2)
        self.combo_b = QtWidgets.QComboBox()
        self.combo_b.addItem("all")
        self.combo_b.insertSeparator(1)
        self.combo_b.addItems(self.combo_list)
        config_layout.addWidget(self.combo_b, 0, 3)

        # Column 4 & 5: Parameters (Cutoff /Bin)
        config_layout.addWidget(QtWidgets.QLabel("Cutoff :"), 0, 4)
        self.spin_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_cutoff.setRange(5.0, 50.0)
        self.spin_cutoff.setValue(12.0)
        self.spin_cutoff.setSuffix(" Å")
        config_layout.addWidget(self.spin_cutoff, 0, 5)

        config_layout.addWidget(QtWidgets.QLabel("Bin width :"), 1, 4)
        self.spin_dr = QtWidgets.QDoubleSpinBox()
        self.spin_dr.setRange(0.05, 0.5)
        self.spin_dr.setValue(0.05)
        self.spin_dr.setSuffix(" Å")
        self.spin_cutoff.setSingleStep(0.05)
        config_layout.addWidget(self.spin_dr, 1, 5)

        # Smoothing
        config_layout.addWidget(QtWidgets.QLabel("Smoothing (σ) :"), 1, 0)
        smooth_hbox = QtWidgets.QHBoxLayout()
        self.slider_sigma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_sigma.setRange(0, 100)
        self.slider_sigma.setValue(15)
        self.slider_sigma.valueChanged.connect(self.update_plot_only)
        self.lbl_sigma = QtWidgets.QLabel("0.15 Å")
        self.lbl_sigma.setFixedWidth(50)
        smooth_hbox.addWidget(self.slider_sigma)
        smooth_hbox.addWidget(self.lbl_sigma)
        config_layout.addLayout(smooth_hbox, 1, 1, 1, 3)

        # ---COMPUTE BUTTON (BaseBuilderDialog Style) ---
        # Using create_icon_button for auto-adaptive icon
        self.btn_compute = self.create_icon_button("Compute", "play", primary=True)
        self.btn_compute.setMinimumHeight(30)
        self.btn_compute.clicked.connect(self.run_calculation)
        config_layout.addWidget(self.btn_compute, 2, 4, 1, 2)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # ---TABS & GRAPH ---
        self.tabs_view = QtWidgets.QTabWidget()
        self.tabs_view.setDocumentMode(True)
        self.tabs_view.addTab(QtWidgets.QWidget(), "g(r) - Radial")
        self.tabs_view.addTab(QtWidgets.QWidget(), "G(r) - Reduced")
        self.tabs_view.addTab(QtWidgets.QWidget(), "n(r) - Coordination")
        self.tabs_view.currentChanged.connect(self.update_plot_only)
        main_layout.addWidget(self.tabs_view)

        self.figure, self.ax = plt.subplots(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(NavigationToolbar(self.canvas, self))
        main_layout.addWidget(self.canvas)

    def run_calculation(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            t1 = self.combo_a.currentText()
            t2 = self.combo_b.currentText()
            cutoff = self.spin_cutoff.value()
            dr = self.spin_dr.value()

            self.rdf_results = compute_rdf(
                self.system, type1=t1, type2=t2, cutoff=cutoff, dr=dr
            )

            self.update_plot_only()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def update_plot_only(self):
        """Gère uniquement l'affichage et le lissage visuel à l'écran."""
        if self.rdf_results is None:
            return
        self.ax.clear()

        # 1. Récupération des données pures calculées par le backend
        res, rho_target, rho0 = self.rdf_results
        r = res.index
        dr = self.spin_dr.value()

        # 2. Gestion du lissage (Smoothing graphique temporaire)
        sigma_val = self.slider_sigma.value() / 100.0
        self.lbl_sigma.setText(f"{sigma_val:.2f} Å")

        if sigma_val < 0.001:
            gr_smooth = res["g_r"]
        else:
            gr_smooth = gaussian_filter1d(res["g_r"], sigma=sigma_val / dr)

        tab_idx = self.tabs_view.currentIndex()

        # --- Onglet 0 : g(r) - Radial ---
        if tab_idx == 0:
            self.ax.plot(r, res["g_r"], color="gray", alpha=0.2, label="Raw")
            self.ax.plot(r, gr_smooth, color="#d32f2f", lw=1.5, label="g(r)")
            self.ax.set_ylabel("g(r)")

        # --- Onglet 1 : G(r) - Reduced ---
        elif tab_idx == 1:
            # On recalcule la version lissée à partir du g(r) lissé et de rho0 du backend
            Gr_smooth = 4 * np.pi * r * rho0 * (gr_smooth - 1)

            self.ax.plot(r, res["G_r"], color="gray", alpha=0.2, label="Raw")
            self.ax.plot(r, Gr_smooth, color="#1976D2", lw=1.5, label="G(r)")
            self.ax.axhline(0, color="black", lw=0.5, ls="--")
            self.ax.set_ylabel("G(r) (Å⁻²)")

        # --- Onglet 2 : n(r) - Coordination ---
        elif tab_idx == 2:
            # On recalcule l'intégration cumulative à partir du g(r) lissé et de rho_target
            nr_smooth = np.cumsum(4 * np.pi * r**2 * rho_target * gr_smooth * dr)

            self.ax.plot(r, res["n_r"], color="gray", alpha=0.2, label="Raw")
            self.ax.plot(r, nr_smooth, color="#388E3C", lw=1.5, label="n(r)")
            self.ax.set_ylabel("Coordination Number N")

        # Configuration finale des axes Matplotlib
        self.ax.set_xlabel("r (Å)")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()


class SilicateDialog(BaseBuilderDialog):
    def __init__(self, parent, system):
        super().__init__(parent, "C-(A)-S-H Analysis", 650)
        self.setMinimumHeight(450)  # Slightly larger to accommodate both tables

        self.data = system
        self.u = system.to_mda()
        self.atom_types = system.atom_types

        self.setup_ui()
        self.auto_detect_types()

    def setup_ui(self):
        layout = self.main_layout

        # --- SECTION 1: CONFIGURATION ---
        type_group = QtWidgets.QGroupBox("Atom types selection")
        type_grid = QtWidgets.QGridLayout(type_group)

        type_grid.addWidget(QtWidgets.QLabel("Silicon type(s):"), 0, 0)
        self.edit_si = QtWidgets.QLineEdit()
        type_grid.addWidget(self.edit_si, 0, 1)

        type_grid.addWidget(QtWidgets.QLabel("Silicate oxygen type(s):"), 0, 2)
        self.edit_o = QtWidgets.QLineEdit()
        type_grid.addWidget(self.edit_o, 0, 3)

        type_grid.addWidget(QtWidgets.QLabel("Aluminum type(s):"), 1, 0)
        self.edit_al = QtWidgets.QLineEdit()
        type_grid.addWidget(self.edit_al, 1, 1)

        type_grid.addWidget(QtWidgets.QLabel("Calcium type(s):"), 1, 2)
        self.edit_ca = QtWidgets.QLineEdit()
        type_grid.addWidget(self.edit_ca, 1, 3)

        type_grid.addWidget(QtWidgets.QLabel("Si-O Cutoff:"), 2, 0)
        self.spin_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_cutoff.setRange(1.0, 3.0)
        self.spin_cutoff.setValue(1.85)
        self.spin_cutoff.setSuffix(" Å")
        type_grid.addWidget(self.spin_cutoff, 2, 1)

        layout.addWidget(type_group)

        # ---RUN BUTTON (Placed between sections) ---
        self.btn_run = self.create_icon_button("Run analysis", "play", primary=True)
        self.btn_run.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run)

        # --- SECTION 2: COMPOSITION RATIOS (TABLEAU) ---
        ratio_group = QtWidgets.QGroupBox("Chemical and structural properties")
        ratio_layout = QtWidgets.QVBoxLayout(ratio_group)

        # Remove margins around table in group
        ratio_layout.setContentsMargins(2, 2, 2, 2)  # Minimum margins

        self.table_ratios = QtWidgets.QTableWidget(1, 4)
        self.table_ratios.setHorizontalHeaderLabels(
            ["Ca/(Si+Al)", "Al/Si", "H₂O/(Si+Al)", "MCL"]
        )
        self.table_ratios.verticalHeader().setVisible(False)

        # ADJUSTMENT HERE: 50px or 52px to remove the bottom white bar
        self.table_ratios.setFixedHeight(52)

        self.table_ratios.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        # Removes the border from the table itself to fit it in better
        self.table_ratios.setFrameShape(QtWidgets.QFrame.NoFrame)

        ratio_layout.addWidget(self.table_ratios)
        layout.addWidget(ratio_group)

        # --- SECTION 3: Qn DISTRIBUTION (TABLEAU) ---
        qn_group = QtWidgets.QGroupBox("Polymerisation (Qⁿ units distribution)")
        qn_layout = QtWidgets.QVBoxLayout(qn_group)

        # Remove margins
        qn_layout.setContentsMargins(2, 2, 2, 2)

        self.table_qn = QtWidgets.QTableWidget(1, 5)
        self.table_qn.setHorizontalHeaderLabels(
            ["Q0 (%)", "Q1 (%)", "Q2 (%)", "Q3 (%)", "Q4 (%)"]
        )
        self.table_qn.verticalHeader().setVisible(False)

        # AJUSTEMENT ICI : 52px
        self.table_qn.setFixedHeight(52)

        self.table_qn.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.table_qn.setFrameShape(QtWidgets.QFrame.NoFrame)

        qn_layout.addWidget(self.table_qn)
        layout.addWidget(qn_group)

        # Optional: Add a spring at the end to push everything up
        layout.addStretch()

        # Status log
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.showMessage("Ready to analyze.")
        layout.addWidget(self.status_bar)

    def auto_detect_types(self):
        """Detects types based on atomic masses."""
        mass_dict = {"Si": 28.085, "O": 15.999, "Al": 26.982, "Ca": 40.078, "H": 1.008}
        found = {"Si": [], "O": [], "Al": [], "Ca": []}
        data_masses = getattr(self.data, "masses", [])

        for i, t_mass in enumerate(data_masses):
            if i < len(self.atom_types):
                t_name = self.atom_types[i]
                for element, target_mass in mass_dict.items():
                    if abs(t_mass - target_mass) < 0.5:
                        if element in found:
                            found[element].append(t_name)
                        break

        self.edit_si.setText(" ".join(found["Si"]))
        self.edit_o.setText(" ".join(found["O"]))
        self.edit_al.setText(" ".join(found["Al"]))
        self.edit_ca.setText(" ".join(found["Ca"]))

        detected_list = [f"{k}: {', '.join(v)}" for k, v in found.items() if v]
        if detected_list:
            msg = f"🔍 Auto-detected: {' | '.join(detected_list)}"
            self.status_bar.showMessage(msg, 5000)  # Displays for 5 seconds

    def run_analysis(self):
        try:
            self.status_bar.showMessage("⏱ Analyzing connectivity...", 0)

            # Récupération des paramètres de l'UI
            si = self.edit_si.text().strip()
            o = self.edit_o.text().strip()
            al = self.edit_al.text().strip()
            ca = self.edit_ca.text().strip()
            cutoff = self.spin_cutoff.value()

            if not si or not o:
                self.status_bar.showMessage(
                    "✕ Error: Si and O types must be defined.", 5000
                )
                return

            # APPEL UNIQUE AU BACKEND (On passe directement self.data qui est l'AtomicSystem)
            res = analyze_silicates(
                self.data,
                si_types=si,
                o_types=o,
                al_types=al,
                ca_types=ca,
                cutoff=cutoff,
            )

            # --- MISE EN FORME DES RAPPORTS CHIMIQUES ---
            ratios = [
                f"{res['Ca/(Si+Al)']:.3f}",
                f"{res['Al/Si']:.3f}" if res["Al/Si"] > 0 else "-",
                f"{res['H2O/(Si+Al)']:.3f}",
                f"{res['MCL']:.2f}" if not np.isnan(res["MCL"]) else "N/A",
            ]

            # Remplissage du tableau des rapports
            for i, val in enumerate(ratios):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table_ratios.setItem(0, i, item)

            # --- REMPLISSAGE DU TABLEAU Qn ---
            pqsi = res["Qn_distribution"]
            for i in range(5):
                item = QtWidgets.QTableWidgetItem(f"{pqsi[i]:.1f}%")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table_qn.setItem(0, i, item)

            self.status_bar.showMessage(
                f"✓ Success: {res['n_si_analyzed']} Si atoms analyzed.", 7000
            )

        except Exception as e:
            self.status_bar.showMessage(f"⚠ Error: {str(e)}", 10000)
