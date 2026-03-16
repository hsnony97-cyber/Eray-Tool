"""
Eray-Tool: NX Nastran Thickness Optimization Tool

BDF dosyasındaki PSHELL'lerin kalınlıklarını bağımsız olarak optimize eder.
Amaç: Minimum ağırlık ile displacement ve stress sınırlarını sağlamak.

Algoritmalar:
  1) Sensitivity-Based
  2) SciPy Minimize (SLSQP)
  3) Hybrid (Bisection + Sensitivity)
  4) DOE + Surrogate Model
"""

import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def select_file(entry, title, filetypes):
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def select_directory(entry, title):
    path = filedialog.askdirectory(title=title)
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def format_nastran_real(value):
    """Nastran real format: integer değerlerde bile ondalık nokta olmalı (2 -> 2.0)"""
    s = f"{value:g}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


# ---------------------------------------------------------------------------
# BDF okuma: PSHELL, malzeme density, eleman alanları
# ---------------------------------------------------------------------------

def read_bdf_model(bdf_path):
    """pyNastran ile BDF'yi okur ve PSHELL bilgilerini çıkarır.
    Returns:
        pids: PSHELL PID listesi
        pid_mid_map: {pid: mid} PSHELL -> malzeme eşlemesi
        mid_density_map: {mid: rho} malzeme density
        pid_area_map: {pid: total_area} her PSHELL'e ait elemanların toplam alanı
    """
    model = BDF()
    model.read_bdf(bdf_path, xref=True)

    pids = []
    pid_mid_map = {}
    mid_density_map = {}
    pid_area_map = {}

    # PSHELL kartlarını oku
    for pid, prop in model.properties.items():
        if prop.type == "PSHELL":
            pids.append(pid)
            mid = prop.mid1_ref.mid if hasattr(prop, "mid1_ref") and prop.mid1_ref else prop.Mid1()
            pid_mid_map[pid] = mid

    # MAT1 density değerlerini oku
    for mid, mat in model.materials.items():
        if hasattr(mat, "rho"):
            mid_density_map[mid] = mat.rho

    # Her PID'e ait elemanların toplam alanını hesapla
    for pid in pids:
        pid_area_map[pid] = 0.0

    for eid, elem in model.elements.items():
        if elem.type in ("CQUAD4", "CTRIA3", "CQUAD8", "CTRIA6"):
            elem_pid = elem.pid
            if elem_pid in pid_area_map:
                try:
                    area = elem.Area()
                    pid_area_map[elem_pid] += area
                except Exception:
                    pass

    return pids, pid_mid_map, mid_density_map, pid_area_map


def compute_real_mass(thickness_map, pid_mid_map, mid_density_map, pid_area_map):
    """Gerçek kütle hesabı: mass = sum(thickness * area * density) for each PID"""
    total_mass = 0.0
    pid_mass = {}
    for pid, t in thickness_map.items():
        area = pid_area_map.get(pid, 0.0)
        mid = pid_mid_map.get(pid)
        rho = mid_density_map.get(mid, 0.0) if mid else 0.0
        m = t * area * rho
        pid_mass[pid] = m
        total_mass += m
    return total_mass, pid_mass


# ---------------------------------------------------------------------------
# BDF PSHELL yazma (pyNastran kullanmadan, hızlı text işlemi)
# ---------------------------------------------------------------------------

def read_pshell_pids(bdf_path):
    """BDF dosyasından PSHELL PID listesini okur (hızlı text tarama)."""
    pids = []
    with open(bdf_path, "r") as f:
        for line in f:
            if line.upper().startswith("PSHELL"):
                if "," in line:
                    parts = line.split(",")
                    pids.append(int(parts[1].strip()))
                else:
                    pid_field = line[8:16].strip()
                    if pid_field:
                        pids.append(int(pid_field))
    return pids


def modify_pshell_thicknesses(bdf_path, output_bdf_path, thickness_map, log_callback=None):
    """BDF dosyasındaki PSHELL kalınlıklarını thickness_map'e göre değiştirir."""
    with open(bdf_path, "r") as f:
        lines = f.readlines()

    modified_count = 0
    new_lines = []

    for line in lines:
        if line.upper().startswith("PSHELL") and "," in line:
            parts = line.split(",")
            if len(parts) >= 4:
                pid = int(parts[1].strip())
                if pid in thickness_map:
                    parts[3] = format_nastran_real(thickness_map[pid])
                    new_lines.append(",".join(parts))
                    modified_count += 1
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        elif line.upper().startswith("PSHELL"):
            if len(line) >= 32:
                pid_field = line[8:16].strip()
                if pid_field:
                    pid = int(pid_field)
                    if pid in thickness_map:
                        t_field = format_nastran_real(thickness_map[pid]).rjust(8)
                        new_line = line[:24] + t_field + line[32:]
                        new_lines.append(new_line)
                        modified_count += 1
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(output_bdf_path, "w") as f:
        f.writelines(new_lines)

    if log_callback:
        log_callback(f"  {modified_count} PSHELL güncellendi.")
    return modified_count


# ---------------------------------------------------------------------------
# Nastran çalıştırma
# ---------------------------------------------------------------------------

def run_nastran(nastran_exe, bdf_path, output_dir, log_callback):
    bdf_basename = os.path.splitext(os.path.basename(bdf_path))[0]
    op2_path = os.path.join(output_dir, bdf_basename + ".op2")

    cmd = [nastran_exe, bdf_path, f"out={output_dir}{os.sep}{bdf_basename}"]
    log_callback(f"  Nastran: {bdf_basename}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    if result.returncode != 0:
        log_callback(f"  Nastran return code: {result.returncode}")

    if not os.path.isfile(op2_path):
        raise FileNotFoundError(
            f"OP2 bulunamadı: {op2_path}\n"
            "BDF'de 'PARAM,POST,-1' olduğundan emin olunuz."
        )
    return op2_path


# ---------------------------------------------------------------------------
# OP2 sonuç okuma
# ---------------------------------------------------------------------------

def get_von_mises_stresses(op2_model):
    stress_map = {}
    for attr_name in ("cquad4_stress", "ctria3_stress"):
        stress_dict = getattr(op2_model, attr_name, None)
        if not stress_dict:
            continue
        for subcase_id, stress_obj in stress_dict.items():
            if hasattr(stress_obj, "element_node"):
                eids = stress_obj.element_node[:, 0]
                center_mask = stress_obj.element_node[:, 1] == 0
            elif hasattr(stress_obj, "element"):
                eids = stress_obj.element
                center_mask = None
            else:
                continue
            ovm = stress_obj.data
            for t_idx in range(ovm.shape[0]):
                for i in range(ovm.shape[1]):
                    if center_mask is not None and not center_mask[i]:
                        continue
                    eid = int(eids[i])
                    vm_val = float(ovm[t_idx, i, -1])
                    if eid not in stress_map or vm_val > stress_map[eid]:
                        stress_map[eid] = vm_val
    return stress_map


def get_max_absolute_displacement(op2_model):
    max_disp = 0.0
    if not op2_model.displacements:
        return 0.0
    for subcase_id, disp_obj in op2_model.displacements.items():
        translations = disp_obj.data[:, :, :3]
        magnitudes = np.sqrt(np.sum(translations ** 2, axis=2))
        current_max = float(np.max(magnitudes))
        if current_max > max_disp:
            max_disp = current_max
    return max_disp


def extract_results_csv(op2_model, output_dir, label, log_callback):
    stress_map = get_von_mises_stresses(op2_model)
    if stress_map:
        rows = [{"ElementID": eid, "VonMises": vm} for eid, vm in sorted(stress_map.items())]
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"von_mises_{label}.csv")
        df.to_csv(csv_path, index=False)
        log_callback(f"  Stress CSV: {csv_path} ({len(df)} eleman)")

    if op2_model.displacements:
        disp_data = []
        for subcase_id, disp_obj in op2_model.displacements.items():
            node_ids = disp_obj.node_gridtype[:, 0]
            data = disp_obj.data
            for t_idx in range(data.shape[0]):
                for i, nid in enumerate(node_ids):
                    r = data[t_idx, i, :]
                    disp_data.append({
                        "NodeID": int(nid),
                        "T1": r[0], "T2": r[1], "T3": r[2],
                        "R1": r[3], "R2": r[4], "R3": r[5],
                        "Magnitude": float(np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)),
                    })
        df = pd.DataFrame(disp_data)
        csv_path = os.path.join(output_dir, f"displacements_{label}.csv")
        df.to_csv(csv_path, index=False)
        log_callback(f"  Disp CSV: {csv_path} ({len(df)} satır)")


def load_allowable_excel(excel_path):
    df = pd.read_excel(excel_path, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    eid_col = df.columns[0]
    allow_col = df.columns[1]
    return {int(row[eid_col]): float(row[allow_col]) for _, row in df.iterrows()}


def check_stress_constraints(stress_map, allowable_map):
    failed = []
    max_vm = 0.0
    for eid, vm in stress_map.items():
        if vm > max_vm:
            max_vm = vm
        if eid in allowable_map and vm > allowable_map[eid]:
            failed.append((eid, vm, allowable_map[eid]))
    return len(failed) == 0, failed, max_vm


def log_mass_summary(total_mass, max_disp, max_disp_limit, max_vm, stress_ok, log_callback):
    """Mass Summary: toplam mass, max displacement, max stress"""
    disp_status = "OK" if max_disp <= max_disp_limit else "FAIL"
    stress_status = "OK" if stress_ok else "FAIL"
    log_callback("")
    log_callback("  ╔══════════════════════════════════════════╗")
    log_callback("  ║           MASS SUMMARY                   ║")
    log_callback("  ╠══════════════════════════════════════════╣")
    log_callback(f"  ║  Total Mass:        {total_mass:>12.4f}  kg     ║")
    log_callback(f"  ║  Max Displacement:  {max_disp:>12.6f}  [{disp_status:>4}] ║")
    log_callback(f"  ║  Max Von Mises:     {max_vm:>12.2f}  [{stress_status:>4}] ║")
    log_callback("  ╚══════════════════════════════════════════╝")
    log_callback("")


def solve_and_evaluate(nastran_exe, bdf_path, thickness_map, work_dir, log_callback):
    """BDF'yi değiştir, çöz, sonuçları oku ve CSV yaz."""
    os.makedirs(work_dir, exist_ok=True)
    bdf_basename = os.path.basename(bdf_path)
    modified_bdf = os.path.join(work_dir, bdf_basename)
    modify_pshell_thicknesses(bdf_path, modified_bdf, thickness_map, log_callback)

    op2_path = run_nastran(nastran_exe, modified_bdf, work_dir, log_callback)
    op2_model = OP2()
    op2_model.read_op2(op2_path)

    max_disp = get_max_absolute_displacement(op2_model)
    stress_map = get_von_mises_stresses(op2_model)

    iter_label = os.path.basename(work_dir)
    extract_results_csv(op2_model, work_dir, iter_label, log_callback)

    return max_disp, stress_map, op2_model


# ---------------------------------------------------------------------------
# İterasyon Tracker (grafik + mass summary için)
# ---------------------------------------------------------------------------

class IterationTracker:
    """Her iterasyonun mass, displacement, stress bilgisini tutar ve grafiğe çizer."""

    def __init__(self, pid_mid_map, mid_density_map, pid_area_map,
                 max_disp_limit, plot_callback):
        self.pid_mid_map = pid_mid_map
        self.mid_density_map = mid_density_map
        self.pid_area_map = pid_area_map
        self.max_disp_limit = max_disp_limit
        self.plot_callback = plot_callback

        self.iterations = []
        self.masses = []
        self.displacements = []
        self.stresses = []

    def record(self, iteration_label, thickness_map, max_disp, max_vm, stress_ok,
               allowable_map, log_callback):
        """Bir iterasyonu kaydet, mass summary bas, grafiği güncelle."""
        total_mass, _ = compute_real_mass(
            thickness_map, self.pid_mid_map, self.mid_density_map, self.pid_area_map
        )

        self.iterations.append(len(self.iterations) + 1)
        self.masses.append(total_mass)
        self.displacements.append(max_disp)
        self.stresses.append(max_vm)

        log_mass_summary(total_mass, max_disp, self.max_disp_limit, max_vm, stress_ok, log_callback)

        # Grafiği güncelle
        self.plot_callback(self.iterations, self.masses, self.displacements, self.max_disp_limit)

        return total_mass


# ---------------------------------------------------------------------------
# ALGORİTMA 1: Sensitivity-Based
# ---------------------------------------------------------------------------

def optimize_sensitivity(bdf_path, nastran_exe, output_dir, pids, allowable_map,
                         min_t, max_t, step, max_disp_limit, max_iter,
                         log_callback, progress_callback, tracker):
    n = len(pids)
    log_callback(f"\n  Toplam {n} PSHELL optimize edilecek.")
    current_t = {pid: max_t for pid in pids}

    for iteration in range(max_iter):
        log_callback(f"\n{'='*50}")
        log_callback(f"SENSITIVITY İTERASYON {iteration + 1}/{max_iter}")
        progress_callback((iteration / max_iter) * 100)

        ref_dir = os.path.join(output_dir, f"sens_iter{iteration}_ref")
        log_callback("  Referans çözüm...")
        ref_disp, ref_stress, _ = solve_and_evaluate(
            nastran_exe, bdf_path, current_t, ref_dir, log_callback
        )

        stress_ok, failed, max_vm = check_stress_constraints(ref_stress, allowable_map)
        tracker.record(f"Sens_{iteration}", current_t, ref_disp, max_vm, stress_ok,
                       allowable_map, log_callback)

        if not stress_ok:
            log_callback(f"  {len(failed)} eleman stress aşıyor, kalınlaştırılıyor...")
            for eid, vm, allow in failed:
                for pid in pids:
                    if current_t[pid] < max_t:
                        current_t[pid] = min(current_t[pid] + step, max_t)
            continue

        if ref_disp > max_disp_limit:
            log_callback("  Displacement limiti aşılıyor!")
            break

        log_callback("  Hassasiyet analizi yapılıyor...")
        sensitivities = {}

        for i, pid in enumerate(pids):
            if current_t[pid] <= min_t + 1e-9:
                sensitivities[pid] = float('inf')
                continue

            perturbed_t = dict(current_t)
            perturbed_t[pid] = current_t[pid] - step

            pert_dir = os.path.join(output_dir, f"sens_iter{iteration}_p{pid}")
            try:
                pert_disp, pert_stress, _ = solve_and_evaluate(
                    nastran_exe, bdf_path, perturbed_t, pert_dir, log_callback
                )
                sensitivity = (pert_disp - ref_disp) / step
                sensitivities[pid] = sensitivity

                s_ok, _, _ = check_stress_constraints(pert_stress, allowable_map)
                if not s_ok:
                    sensitivities[pid] = float('inf')
            except Exception:
                sensitivities[pid] = float('inf')

            log_callback(f"    PID {pid}: dDisp/dT = {sensitivities[pid]:.6f} ({i+1}/{n})")

        sorted_pids = sorted(
            [p for p in pids if sensitivities[p] < float('inf')],
            key=lambda p: abs(sensitivities[p])
        )

        if not sorted_pids:
            log_callback("  Hiçbir PSHELL daha fazla inceltilemez.")
            break

        n_reduce = max(1, len(sorted_pids) // 2)
        reduced_any = False
        for pid in sorted_pids[:n_reduce]:
            new_val = current_t[pid] - step
            if new_val >= min_t:
                current_t[pid] = round(new_val, 8)
                reduced_any = True

        if not reduced_any:
            log_callback("  Minimum kalınlığa ulaşıldı.")
            break

        log_callback(f"  {n_reduce} PSHELL incelendi.")

    log_callback(f"\n{'='*50}")
    log_callback("SON ÇÖZÜM")
    final_dir = os.path.join(output_dir, "final_result")
    final_disp, final_stress, final_op2 = solve_and_evaluate(
        nastran_exe, bdf_path, current_t, final_dir, log_callback
    )
    extract_results_csv(final_op2, final_dir, "final", log_callback)
    stress_ok, _, max_vm = check_stress_constraints(final_stress, allowable_map)
    tracker.record("Final", current_t, final_disp, max_vm, stress_ok, allowable_map, log_callback)

    progress_callback(100)
    return current_t, final_disp, final_stress


# ---------------------------------------------------------------------------
# ALGORİTMA 2: SciPy Minimize (SLSQP)
# ---------------------------------------------------------------------------

def optimize_scipy(bdf_path, nastran_exe, output_dir, pids, allowable_map,
                   min_t, max_t, step, max_disp_limit, max_iter,
                   log_callback, progress_callback, tracker):
    from scipy.optimize import minimize

    n = len(pids)
    eval_count = [0]
    log_callback(f"\n  SciPy SLSQP - {n} PSHELL")

    def objective(x):
        return np.sum(x)

    def disp_constraint(x):
        eval_count[0] += 1
        t_map = {pid: float(x[i]) for i, pid in enumerate(pids)}
        work_dir = os.path.join(output_dir, f"scipy_eval{eval_count[0]}")
        log_callback(f"\n  Eval #{eval_count[0]}: sum(t) = {np.sum(x):.4f}")
        progress_callback(min(eval_count[0] / max_iter * 100, 99))

        try:
            max_disp, stress_map, _ = solve_and_evaluate(
                nastran_exe, bdf_path, t_map, work_dir, log_callback
            )
            stress_ok, _, max_vm = check_stress_constraints(stress_map, allowable_map)
            tracker.record(f"SciPy_{eval_count[0]}", t_map, max_disp, max_vm, stress_ok,
                           allowable_map, log_callback)
            return max_disp_limit - max_disp
        except Exception as exc:
            log_callback(f"  Nastran hatası: {exc}")
            return -1.0

    x0 = np.full(n, (min_t + max_t) / 2.0)
    bounds = [(min_t, max_t)] * n
    constraints = [{"type": "ineq", "fun": disp_constraint}]

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": step / 10, "eps": step},
    )

    log_callback(f"\n  SciPy sonucu: {result.message}")
    final_t = {pid: round(float(result.x[i]), 8) for i, pid in enumerate(pids)}

    final_dir = os.path.join(output_dir, "final_result")
    final_disp, final_stress, final_op2 = solve_and_evaluate(
        nastran_exe, bdf_path, final_t, final_dir, log_callback
    )
    extract_results_csv(final_op2, final_dir, "final", log_callback)
    stress_ok, _, max_vm = check_stress_constraints(final_stress, allowable_map)
    tracker.record("Final", final_t, final_disp, max_vm, stress_ok, allowable_map, log_callback)

    progress_callback(100)
    return final_t, final_disp, final_stress


# ---------------------------------------------------------------------------
# ALGORİTMA 3: Hybrid
# ---------------------------------------------------------------------------

def optimize_hybrid(bdf_path, nastran_exe, output_dir, pids, allowable_map,
                    min_t, max_t, step, max_disp_limit, max_iter,
                    log_callback, progress_callback, tracker):
    log_callback("\n  AŞAMA 1: Bisection ile başlangıç kalınlığı...")

    low, high = min_t, max_t
    best_uniform_t = max_t
    bisect_iter = 0
    max_bisect = 20

    while high - low > step / 2 and bisect_iter < max_bisect:
        bisect_iter += 1
        mid = round((low + high) / 2, 8)
        t_map = {pid: mid for pid in pids}

        work_dir = os.path.join(output_dir, f"bisect_{bisect_iter}")
        log_callback(f"\n  Bisection #{bisect_iter}: t = {mid}")
        progress_callback(bisect_iter / (max_bisect + max_iter) * 100)

        try:
            max_disp, stress_map, _ = solve_and_evaluate(
                nastran_exe, bdf_path, t_map, work_dir, log_callback
            )
            stress_ok, _, max_vm = check_stress_constraints(stress_map, allowable_map)
            tracker.record(f"Bisect_{bisect_iter}", t_map, max_disp, max_vm, stress_ok,
                           allowable_map, log_callback)

            if max_disp <= max_disp_limit and stress_ok:
                best_uniform_t = mid
                high = mid
            else:
                low = mid
        except Exception as exc:
            log_callback(f"    Nastran hatası: {exc}")
            low = mid

    log_callback(f"\n  Bisection sonucu: t = {best_uniform_t}")
    log_callback(f"\n  AŞAMA 2: Sensitivity ile ince ayar...")

    remaining_iter = max(3, max_iter - bisect_iter)
    return optimize_sensitivity(
        bdf_path, nastran_exe, os.path.join(output_dir, "sensitivity_phase"),
        pids, allowable_map, min_t, best_uniform_t, step, max_disp_limit, remaining_iter,
        log_callback, progress_callback, tracker
    )


# ---------------------------------------------------------------------------
# ALGORİTMA 4: DOE + Surrogate Model
# ---------------------------------------------------------------------------

def latin_hypercube_sampling(n_samples, n_vars, min_vals, max_vals, seed=42):
    rng = np.random.RandomState(seed)
    result = np.zeros((n_samples, n_vars))
    for j in range(n_vars):
        cut = np.linspace(0, 1, n_samples + 1)
        uniform_samples = rng.uniform(low=cut[:-1], high=cut[1:])
        rng.shuffle(uniform_samples)
        result[:, j] = min_vals[j] + uniform_samples * (max_vals[j] - min_vals[j])
    return result


def optimize_doe_surrogate(bdf_path, nastran_exe, output_dir, pids, allowable_map,
                           min_t, max_t, step, max_disp_limit, max_iter,
                           log_callback, progress_callback, tracker):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from scipy.optimize import minimize

    n_vars = len(pids)
    min_vals = np.full(n_vars, min_t)
    max_vals = np.full(n_vars, max_t)
    n_doe_samples = min(max(2 * n_vars + 1, 10), max(max_iter // 2, 5))

    log_callback(f"\n  DOE + Surrogate Model")
    log_callback(f"  {n_vars} PSHELL, {n_doe_samples} DOE numune")
    log_callback(f"\n  AŞAMA 1: Latin Hypercube Sampling...")

    samples = latin_hypercube_sampling(n_doe_samples, n_vars, min_vals, max_vals)

    X_train = []
    y_disp = []

    for i in range(n_doe_samples):
        t_map = {pid: round(float(samples[i, j]), 8) for j, pid in enumerate(pids)}
        work_dir = os.path.join(output_dir, f"doe_sample_{i}")
        log_callback(f"\n  DOE Numune {i+1}/{n_doe_samples}")
        progress_callback((i / n_doe_samples) * 40)

        try:
            max_disp, stress_map, _ = solve_and_evaluate(
                nastran_exe, bdf_path, t_map, work_dir, log_callback
            )
            stress_ok, _, max_vm = check_stress_constraints(stress_map, allowable_map)
            tracker.record(f"DOE_{i+1}", t_map, max_disp, max_vm, stress_ok,
                           allowable_map, log_callback)

            X_train.append(samples[i])
            y_disp.append(max_disp)
        except Exception as exc:
            log_callback(f"    Nastran hatası: {exc}")

    if len(X_train) < 3:
        raise RuntimeError("Yeterli DOE numunesi çözülemedi!")

    X_train = np.array(X_train)
    y_disp = np.array(y_disp)

    # DOE CSV
    doe_rows = []
    for i in range(len(X_train)):
        row = {"Sample": i + 1, "MaxDisp": y_disp[i]}
        for j, pid in enumerate(pids):
            row[f"PID_{pid}"] = X_train[i, j]
        doe_rows.append(row)
    pd.DataFrame(doe_rows).to_csv(os.path.join(output_dir, "doe_samples.csv"), index=False)

    # AŞAMA 2: Surrogate
    log_callback(f"\n  AŞAMA 2: Surrogate model ile optimizasyon...")
    remaining_iter = max_iter - n_doe_samples
    best_t = None
    best_weight = float('inf')

    for surr_iter in range(max(remaining_iter, 3)):
        log_callback(f"\n{'='*50}")
        log_callback(f"SURROGATE İTERASYON {surr_iter + 1}")
        progress_callback(40 + (surr_iter / max(remaining_iter, 3)) * 50)

        degree = 1 if n_vars > 20 else 2
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X_train)

        model = Ridge(alpha=1.0)
        model.fit(X_poly, y_disp)
        log_callback(f"  Surrogate R²: {model.score(X_poly, y_disp):.4f}")

        def surrogate_objective(x):
            return np.sum(x)

        def surrogate_disp_constraint(x):
            x_poly = poly.transform(x.reshape(1, -1))
            return max_disp_limit - model.predict(x_poly)[0]

        opt_result = minimize(
            surrogate_objective, np.mean(X_train, axis=0), method="SLSQP",
            bounds=[(min_t, max_t)] * n_vars,
            constraints=[{"type": "ineq", "fun": surrogate_disp_constraint}],
            options={"maxiter": 200},
        )

        candidate_t = {pid: round(float(opt_result.x[j]), 8) for j, pid in enumerate(pids)}
        predicted_disp = model.predict(poly.transform(opt_result.x.reshape(1, -1)))[0]
        log_callback(f"  Surrogate tahmin: Disp={predicted_disp:.6f}, Weight={np.sum(opt_result.x):.4f}")

        verify_dir = os.path.join(output_dir, f"surrogate_verify_{surr_iter}")
        try:
            real_disp, real_stress, _ = solve_and_evaluate(
                nastran_exe, bdf_path, candidate_t, verify_dir, log_callback
            )
            stress_ok, _, max_vm = check_stress_constraints(real_stress, allowable_map)
            total_mass = tracker.record(f"Surr_{surr_iter+1}", candidate_t, real_disp, max_vm,
                                        stress_ok, allowable_map, log_callback)

            error = abs(real_disp - predicted_disp)
            log_callback(f"  Tahmin hatası: {error:.6f} ({error/max(real_disp,1e-9)*100:.1f}%)")

            X_train = np.vstack([X_train, opt_result.x.reshape(1, -1)])
            y_disp = np.append(y_disp, real_disp)

            if real_disp <= max_disp_limit and stress_ok and total_mass < best_weight:
                best_t = dict(candidate_t)
                best_weight = total_mass
                log_callback(f"  *** YENİ EN İYİ: Mass={best_weight:.4f} kg")

            if error < step / 10 and real_disp <= max_disp_limit and stress_ok:
                log_callback(f"  Yakınsadı!")
                break
        except Exception as exc:
            log_callback(f"  Doğrulama hatası: {exc}")

    if best_t is None:
        best_idx = np.argmin(y_disp)
        best_t = {pid: round(float(X_train[best_idx, j]), 8) for j, pid in enumerate(pids)}

    log_callback(f"\n{'='*50}")
    log_callback("SON ÇÖZÜM")
    final_dir = os.path.join(output_dir, "final_result")
    final_disp, final_stress, final_op2 = solve_and_evaluate(
        nastran_exe, bdf_path, best_t, final_dir, log_callback
    )
    extract_results_csv(final_op2, final_dir, "final", log_callback)
    stress_ok, _, max_vm = check_stress_constraints(final_stress, allowable_map)
    tracker.record("Final", best_t, final_disp, max_vm, stress_ok, allowable_map, log_callback)

    progress_callback(100)
    return best_t, final_disp, final_stress


# ---------------------------------------------------------------------------
# Ana GUI Uygulaması
# ---------------------------------------------------------------------------

class NastranToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eray-Tool | NX Nastran Thickness Optimizasyon Aracı")
        self.root.geometry("1100x750")
        self.root.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        # Ana yatay bölme: sol (kontroller+log) / sağ (grafik)
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sol panel
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        main_frame = ttk.Frame(left_frame, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(main_frame, text="BDF Dosyası:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.bdf_entry = ttk.Entry(main_frame, width=45)
        self.bdf_entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=4)
        ttk.Button(main_frame, text="Seç...", command=lambda: select_file(
            self.bdf_entry, "BDF Dosyası Seçiniz",
            [("BDF Dosyaları", "*.bdf *.dat *.nas"), ("Tüm Dosyalar", "*.*")]
        )).grid(row=row, column=3)

        row += 1
        ttk.Label(main_frame, text="NX Nastran:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.nastran_entry = ttk.Entry(main_frame, width=45)
        self.nastran_entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=4)
        ttk.Button(main_frame, text="Seç...", command=lambda: select_file(
            self.nastran_entry, "NX Nastran Seçiniz",
            [("Çalıştırılabilir", "*.exe *.bat"), ("Tüm Dosyalar", "*.*")]
        )).grid(row=row, column=3)

        row += 1
        ttk.Label(main_frame, text="Çıktı Klasörü:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.output_entry = ttk.Entry(main_frame, width=45)
        self.output_entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=4)
        ttk.Button(main_frame, text="Seç...", command=lambda: select_directory(
            self.output_entry, "Çıktı Klasörü Seçiniz"
        )).grid(row=row, column=3)

        row += 1
        ttk.Label(main_frame, text="Allowable Excel:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.excel_entry = ttk.Entry(main_frame, width=45)
        self.excel_entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=4)
        ttk.Button(main_frame, text="Seç...", command=lambda: select_file(
            self.excel_entry, "Stress Allowable Excel Seçiniz",
            [("Excel Dosyaları", "*.xlsx *.xls"), ("Tüm Dosyalar", "*.*")]
        )).grid(row=row, column=3)

        row += 1
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=4, sticky=tk.EW, pady=6
        )

        # Parametreler
        row += 1
        param_frame = ttk.LabelFrame(main_frame, text="Optimizasyon Parametreleri", padding=6)
        param_frame.grid(row=row, column=0, columnspan=4, sticky=tk.EW, pady=2)

        ttk.Label(param_frame, text="Min T:").grid(row=0, column=0, sticky=tk.W, padx=2)
        self.min_t_entry = ttk.Entry(param_frame, width=8)
        self.min_t_entry.grid(row=0, column=1, padx=2)
        self.min_t_entry.insert(0, "0.5")

        ttk.Label(param_frame, text="Max T:").grid(row=0, column=2, sticky=tk.W, padx=2)
        self.max_t_entry = ttk.Entry(param_frame, width=8)
        self.max_t_entry.grid(row=0, column=3, padx=2)
        self.max_t_entry.insert(0, "5.0")

        ttk.Label(param_frame, text="Step:").grid(row=0, column=4, sticky=tk.W, padx=2)
        self.step_entry = ttk.Entry(param_frame, width=8)
        self.step_entry.grid(row=0, column=5, padx=2)
        self.step_entry.insert(0, "0.1")

        ttk.Label(param_frame, text="Max Disp:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=(4, 0))
        self.max_disp_entry = ttk.Entry(param_frame, width=8)
        self.max_disp_entry.grid(row=1, column=1, padx=2, pady=(4, 0))
        self.max_disp_entry.insert(0, "10.0")

        ttk.Label(param_frame, text="Max Iter:").grid(row=1, column=2, sticky=tk.W, padx=2, pady=(4, 0))
        self.max_iter_entry = ttk.Entry(param_frame, width=8)
        self.max_iter_entry.grid(row=1, column=3, padx=2, pady=(4, 0))
        self.max_iter_entry.insert(0, "20")

        ttk.Label(param_frame, text="Algoritma:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=(4, 0))
        self.algo_var = tk.StringVar()
        algo_combo = ttk.Combobox(param_frame, textvariable=self.algo_var, state="readonly", width=32)
        algo_combo["values"] = (
            "Sensitivity-Based",
            "SciPy Minimize (SLSQP)",
            "Hybrid (Bisection + Sensitivity)",
            "DOE + Surrogate Model (Önerilen)",
        )
        algo_combo.current(3)
        algo_combo.grid(row=2, column=1, columnspan=4, sticky=tk.W, padx=2, pady=(4, 0))

        main_frame.columnconfigure(1, weight=1)

        # Butonlar
        row += 1
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=6)

        self.run_btn = ttk.Button(btn_frame, text="Optimizasyonu Başlat", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT, padx=6)

        self.single_btn = ttk.Button(btn_frame, text="Tek Çözüm", command=self._on_single_run)
        self.single_btn.pack(side=tk.LEFT, padx=6)

        # İlerleme
        row += 1
        self.progress = ttk.Progressbar(main_frame, mode="determinate")
        self.progress.grid(row=row, column=0, columnspan=4, sticky=tk.EW, pady=(0, 4))

        # Log
        row += 1
        ttk.Label(main_frame, text="İşlem Günlüğü:").grid(row=row, column=0, sticky=tk.W)
        row += 1
        self.log_text = tk.Text(main_frame, height=12, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW)
        main_frame.rowconfigure(row, weight=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=row, column=3, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Sağ panel: Grafik
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)

        self.fig, (self.ax_mass, self.ax_disp) = plt.subplots(2, 1, figsize=(5, 5))
        self.fig.tight_layout(pad=3.0)

        self.ax_mass.set_title("Total Mass vs Iteration", fontsize=10)
        self.ax_mass.set_xlabel("Iteration")
        self.ax_mass.set_ylabel("Mass (kg)")
        self.ax_mass.grid(True, alpha=0.3)

        self.ax_disp.set_title("Max Displacement vs Iteration", fontsize=10)
        self.ax_disp.set_xlabel("Iteration")
        self.ax_disp.set_ylabel("Max Disp")
        self.ax_disp.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def _update_plot(self, iterations, masses, displacements, max_disp_limit):
        """Thread-safe grafik güncelleme."""
        def _do_update():
            self.ax_mass.clear()
            self.ax_mass.set_title("Total Mass vs Iteration", fontsize=10)
            self.ax_mass.set_xlabel("Iteration")
            self.ax_mass.set_ylabel("Mass (kg)")
            self.ax_mass.grid(True, alpha=0.3)
            self.ax_mass.plot(iterations, masses, "b-o", markersize=4, label="Mass")
            self.ax_mass.legend(fontsize=8)

            self.ax_disp.clear()
            self.ax_disp.set_title("Max Displacement vs Iteration", fontsize=10)
            self.ax_disp.set_xlabel("Iteration")
            self.ax_disp.set_ylabel("Max Disp")
            self.ax_disp.grid(True, alpha=0.3)
            self.ax_disp.plot(iterations, displacements, "r-o", markersize=4, label="Max Disp")
            self.ax_disp.axhline(y=max_disp_limit, color="green", linestyle="--",
                                 linewidth=1.5, label=f"Limit ({max_disp_limit})")
            self.ax_disp.legend(fontsize=8)

            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()
        self.root.after(0, _do_update)

    def _log(self, message):
        def _append():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(0, _append)

    def _set_progress(self, value):
        self.root.after(0, lambda v=value: self.progress.configure(value=v))

    def _validate_common(self):
        bdf_path = self.bdf_entry.get().strip()
        nastran_exe = self.nastran_entry.get().strip()
        output_dir = self.output_entry.get().strip()

        if not bdf_path or not os.path.isfile(bdf_path):
            messagebox.showerror("Hata", "Geçerli bir BDF dosyası seçiniz.")
            return None
        if not nastran_exe or not os.path.isfile(nastran_exe):
            messagebox.showerror("Hata", "Geçerli bir NX Nastran dosyası seçiniz.")
            return None
        if not output_dir:
            messagebox.showerror("Hata", "Çıktı klasörü seçiniz.")
            return None
        os.makedirs(output_dir, exist_ok=True)
        return bdf_path, nastran_exe, output_dir

    def _disable_buttons(self):
        self.run_btn.configure(state=tk.DISABLED)
        self.single_btn.configure(state=tk.DISABLED)

    def _finish(self):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.run_btn.configure(state=tk.NORMAL)
        self.single_btn.configure(state=tk.NORMAL)

    def _on_single_run(self):
        vals = self._validate_common()
        if not vals:
            return
        bdf_path, nastran_exe, output_dir = vals
        self._disable_buttons()
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)
        threading.Thread(
            target=self._single_worker, args=(bdf_path, nastran_exe, output_dir), daemon=True
        ).start()

    def _single_worker(self, bdf_path, nastran_exe, output_dir):
        try:
            self._log("=" * 50)
            self._log("TEK ÇÖZÜM MODU")
            self._log("=" * 50)

            op2_path = run_nastran(nastran_exe, bdf_path, output_dir, self._log)
            op2_model = OP2()
            op2_model.read_op2(op2_path)
            extract_results_csv(op2_model, output_dir, "result", self._log)

            max_disp = get_max_absolute_displacement(op2_model)
            max_vm = max(get_von_mises_stresses(op2_model).values()) if get_von_mises_stresses(op2_model) else 0.0

            # BDF'den mass bilgisi
            self._log("\nBDF okunuyor (mass hesabı)...")
            pids, pid_mid_map, mid_density_map, pid_area_map = read_bdf_model(bdf_path)
            # Mevcut kalınlıkları oku
            bdf_model = BDF()
            bdf_model.read_bdf(bdf_path, xref=False)
            current_t = {}
            for pid, prop in bdf_model.properties.items():
                if prop.type == "PSHELL":
                    current_t[pid] = prop.t
            total_mass, _ = compute_real_mass(current_t, pid_mid_map, mid_density_map, pid_area_map)

            log_mass_summary(total_mass, max_disp, 0, max_vm, True, self._log)

            self._log("İŞLEM TAMAMLANDI!")
            self.root.after(0, lambda: messagebox.showinfo("Başarılı", "Sonuçlar CSV olarak yazıldı!"))
        except Exception as exc:
            error_msg = str(exc)
            self._log(f"\nHATA: {error_msg}")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Hata", msg))
        finally:
            self.root.after(0, self._finish)

    def _on_run(self):
        vals = self._validate_common()
        if not vals:
            return
        bdf_path, nastran_exe, output_dir = vals

        excel_path = self.excel_entry.get().strip()
        if not excel_path or not os.path.isfile(excel_path):
            messagebox.showerror("Hata", "Stress allowable Excel dosyası seçiniz.")
            return

        try:
            min_t = float(self.min_t_entry.get().strip())
            max_t = float(self.max_t_entry.get().strip())
            step = float(self.step_entry.get().strip())
            max_disp_limit = float(self.max_disp_entry.get().strip())
            max_iter = int(self.max_iter_entry.get().strip())
        except ValueError:
            messagebox.showerror("Hata", "Parametre değerleri sayısal olmalıdır.")
            return

        if min_t >= max_t or step <= 0:
            messagebox.showerror("Hata", "Min < Max ve Step > 0 olmalıdır.")
            return

        algo_name = self.algo_var.get()
        self._disable_buttons()
        self.progress.configure(mode="determinate", value=0)

        threading.Thread(
            target=self._opt_worker,
            args=(bdf_path, nastran_exe, output_dir, excel_path,
                  min_t, max_t, step, max_disp_limit, max_iter, algo_name),
            daemon=True,
        ).start()

    def _opt_worker(self, bdf_path, nastran_exe, output_dir, excel_path,
                    min_t, max_t, step, max_disp_limit, max_iter, algo_name):
        try:
            self._log("=" * 60)
            self._log(f"OPTİMİZASYON: {algo_name}")
            self._log(f"  Min: {min_t}  Max: {max_t}  Step: {step}")
            self._log(f"  Max Disp: {max_disp_limit}  Max İter: {max_iter}")
            self._log("=" * 60)

            # BDF'den PSHELL, malzeme ve alan bilgisi oku
            self._log("\nBDF modeli okunuyor...")
            pids, pid_mid_map, mid_density_map, pid_area_map = read_bdf_model(bdf_path)
            if not pids:
                raise ValueError("BDF dosyasında PSHELL kartı bulunamadı!")
            self._log(f"  {len(pids)} PSHELL bulundu")
            self._log(f"  {len(mid_density_map)} malzeme density yüklendi")
            total_area = sum(pid_area_map.values())
            self._log(f"  Toplam eleman alanı: {total_area:.4f}")

            allowable_map = load_allowable_excel(excel_path)
            self._log(f"  {len(allowable_map)} element allowable yüklendi.")

            # Tracker oluştur
            tracker = IterationTracker(
                pid_mid_map, mid_density_map, pid_area_map,
                max_disp_limit, self._update_plot
            )

            # Algoritma çalıştır
            if "DOE" in algo_name:
                result_t, final_disp, final_stress = optimize_doe_surrogate(
                    bdf_path, nastran_exe, output_dir, pids, allowable_map,
                    min_t, max_t, step, max_disp_limit, max_iter,
                    self._log, self._set_progress, tracker
                )
            elif "Sensitivity" in algo_name and "Hybrid" not in algo_name:
                result_t, final_disp, final_stress = optimize_sensitivity(
                    bdf_path, nastran_exe, output_dir, pids, allowable_map,
                    min_t, max_t, step, max_disp_limit, max_iter,
                    self._log, self._set_progress, tracker
                )
            elif "SciPy" in algo_name:
                result_t, final_disp, final_stress = optimize_scipy(
                    bdf_path, nastran_exe, output_dir, pids, allowable_map,
                    min_t, max_t, step, max_disp_limit, max_iter,
                    self._log, self._set_progress, tracker
                )
            else:
                result_t, final_disp, final_stress = optimize_hybrid(
                    bdf_path, nastran_exe, output_dir, pids, allowable_map,
                    min_t, max_t, step, max_disp_limit, max_iter,
                    self._log, self._set_progress, tracker
                )

            # Sonuç raporu
            self._log(f"\n{'='*60}")
            self._log("OPTİMİZASYON SONUÇLARI")
            stress_ok, failed, max_vm = check_stress_constraints(final_stress, allowable_map)
            total_mass, pid_mass = compute_real_mass(
                result_t, pid_mid_map, mid_density_map, pid_area_map
            )
            log_mass_summary(total_mass, final_disp, max_disp_limit, max_vm, stress_ok, self._log)

            # Kalınlık + mass CSV
            t_rows = []
            for pid in sorted(result_t.keys()):
                t_rows.append({
                    "PID": pid,
                    "Thickness": result_t[pid],
                    "Area": pid_area_map.get(pid, 0),
                    "Mass": pid_mass.get(pid, 0),
                })
            t_df = pd.DataFrame(t_rows)
            t_csv = os.path.join(output_dir, "optimized_thicknesses.csv")
            t_df.to_csv(t_csv, index=False)
            self._log(f"  Kalınlık dağılımı: {t_csv}")

            # İterasyon geçmişi CSV
            hist_df = pd.DataFrame({
                "Iteration": tracker.iterations,
                "Mass_kg": tracker.masses,
                "MaxDisplacement": tracker.displacements,
                "MaxVonMises": tracker.stresses,
            })
            hist_csv = os.path.join(output_dir, "iteration_history.csv")
            hist_df.to_csv(hist_csv, index=False)

            # Grafiği kaydet
            self.fig.savefig(os.path.join(output_dir, "optimization_plot.png"), dpi=150)

            self._log("\nİŞLEM TAMAMLANDI!")
            self.root.after(0, lambda: messagebox.showinfo(
                "Optimizasyon Tamamlandı",
                f"Total Mass: {total_mass:.4f} kg\n"
                f"Max Disp: {final_disp:.4f}\n"
                f"Max VM: {max_vm:.2f}\n\n"
                f"Detaylar: {t_csv}"
            ))

        except Exception as exc:
            error_msg = str(exc)
            self._log(f"\nHATA: {error_msg}")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Hata", msg))
        finally:
            self.root.after(0, self._finish)


def main():
    root = tk.Tk()
    NastranToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
