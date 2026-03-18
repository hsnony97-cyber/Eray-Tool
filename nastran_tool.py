"""
Eray-Tool: NX Nastran Thickness Optimization Tool

BDF dosyasındaki PSHELL'lerin kalınlıklarını bağımsız olarak optimize eder.
Amaç: Minimum ağırlık ile displacement ve stress sınırlarını sağlamak.

Algoritmalar:
  1) FD + SQP (Forward Difference)
  2) Genetic Algorithm (GA)
"""

import os
import glob as globmod
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def run_nastran(nastran_exe, bdf_path, output_dir, log_callback, memory_mb=0):
    bdf_basename = os.path.splitext(os.path.basename(bdf_path))[0]
    op2_path = os.path.join(output_dir, bdf_basename + ".op2")

    cmd = [nastran_exe, bdf_path, f"out={output_dir}{os.sep}{bdf_basename}"]
    if memory_mb > 0:
        cmd.append(f"mem={memory_mb}mb")
    log_callback(f"  Nastran: {bdf_basename}" + (f" (mem={memory_mb}mb)" if memory_mb > 0 else ""))

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


def load_allowable_excel(excel_path, sheet_name="Stress Allowable(Element Based)"):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]
    id_col = df.columns[0]
    allow_col = df.columns[1]
    return {int(row[id_col]): float(row[allow_col]) for _, row in df.iterrows()}


def load_thickness_range_excel(excel_path):
    """Thickness Range sayfasından PID bazlı tmin, tmax, step okur.
    Returns: {pid: (tmin, tmax, step)}
    """
    df = pd.read_excel(excel_path, sheet_name="Thickness Range")
    df.columns = [c.strip() for c in df.columns]
    result = {}
    for _, row in df.iterrows():
        pid = int(row[df.columns[0]])
        tmin = float(row[df.columns[1]])
        tmax = float(row[df.columns[2]])
        step = float(row[df.columns[3]])
        result[pid] = (tmin, tmax, step)
    return result


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


def solve_and_evaluate(nastran_exe, bdf_path, thickness_map, work_dir, log_callback, memory_mb=0):
    """BDF'yi değiştir, çöz, sonuçları oku, CSV yaz, OP2 ve büyük dosyaları sil."""
    os.makedirs(work_dir, exist_ok=True)
    bdf_basename = os.path.basename(bdf_path)
    modified_bdf = os.path.join(work_dir, bdf_basename)
    modify_pshell_thicknesses(bdf_path, modified_bdf, thickness_map, log_callback)

    op2_path = run_nastran(nastran_exe, modified_bdf, work_dir, log_callback, memory_mb=memory_mb)
    op2_model = OP2()
    op2_model.read_op2(op2_path)

    max_disp = get_max_absolute_displacement(op2_model)
    stress_map = get_von_mises_stresses(op2_model)

    iter_label = os.path.basename(work_dir)
    extract_results_csv(op2_model, work_dir, iter_label, log_callback)

    # OP2 ve diğer büyük Nastran çıktı dosyalarını sil (CSV'ler korunur)
    del op2_model
    cleanup_nastran_outputs(work_dir, log_callback)

    return max_disp, stress_map


def cleanup_nastran_outputs(work_dir, log_callback):
    """OP2, f06, f04, log, DBALL, MASTER gibi büyük Nastran dosyalarını siler."""
    extensions = ("*.op2", "*.f06", "*.f04", "*.log", "*.DBALL", "*.MASTER",
                  "*.IFPDAT", "*.nx_pre_prc", "*.pch")
    deleted = 0
    for ext in extensions:
        for fpath in globmod.glob(os.path.join(work_dir, ext)):
            try:
                os.remove(fpath)
                deleted += 1
            except OSError:
                pass
    if deleted:
        log_callback(f"  {deleted} Nastran çıktı dosyası silindi.")


def solve_batch(nastran_exe, bdf_path, batch_items, n_parallel, log_callback, memory_mb=0):
    """
    Birden fazla thickness kombinasyonunu paralel olarak çözer.
    batch_items: [(thickness_map, work_dir), ...]
    Returns: [(thickness_map, max_disp, stress_map), ...] - aynı sırada
    """
    results = [None] * len(batch_items)

    def _solve_one(idx, thickness_map, work_dir):
        max_disp, stress_map = solve_and_evaluate(
            nastran_exe, bdf_path, thickness_map, work_dir, log_callback, memory_mb=memory_mb
        )
        return idx, thickness_map, max_disp, stress_map

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = []
        for idx, (t_map, w_dir) in enumerate(batch_items):
            futures.append(executor.submit(_solve_one, idx, t_map, w_dir))

        for future in as_completed(futures):
            idx, t_map, max_disp, stress_map = future.result()
            results[idx] = (t_map, max_disp, stress_map)

    return results


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
# ALGORİTMA 1: Forward Difference + SciPy SQP (trust-constr)
# ---------------------------------------------------------------------------

def optimize_fd_sqp(bdf_path, nastran_exe, output_dir, pids, allowable_map,
                    min_t, max_t, step, max_disp_limit, max_iter,
                    log_callback, progress_callback, tracker, n_parallel=1, memory_mb=0,
                    pid_bounds=None):
    """Forward Difference ile gradyan hesaplayıp SciPy trust-constr (SQP) ile optimize eder."""
    from scipy.optimize import minimize, NonlinearConstraint

    n = len(pids)
    eval_count = [0]
    cache = {}

    # Per-PID bounds
    if pid_bounds:
        bounds_list = [pid_bounds[pid] for pid in pids]  # [(tmin, tmax, step), ...]
        pid_min = np.array([b[0] for b in bounds_list])
        pid_max = np.array([b[1] for b in bounds_list])
        pid_step = np.array([b[2] for b in bounds_list])
    else:
        pid_min = np.full(n, min_t)
        pid_max = np.full(n, max_t)
        pid_step = np.full(n, step)

    log_callback(f"\n  FD + SQP - {n} PSHELL, Paralel: {n_parallel}")

    def _eval_point(x, label_prefix="fd"):
        """Bir noktayı çözer, cache'e bakar."""
        key = tuple(round(v, 8) for v in x)
        if key in cache:
            return cache[key]
        eval_count[0] += 1
        t_map = {pid: float(x[i]) for i, pid in enumerate(pids)}
        work_dir = os.path.join(output_dir, f"{label_prefix}_eval{eval_count[0]}")
        try:
            max_disp, stress_map = solve_and_evaluate(
                nastran_exe, bdf_path, t_map, work_dir, log_callback, memory_mb=memory_mb
            )
            stress_ok, _, max_vm = check_stress_constraints(stress_map, allowable_map)
            tracker.record(f"FD_{eval_count[0]}", t_map, max_disp, max_vm, stress_ok,
                           allowable_map, log_callback)
            cache[key] = (max_disp, stress_map)
            return max_disp, stress_map
        except Exception as exc:
            log_callback(f"  Nastran hatası: {exc}")
            cache[key] = (max_disp_limit * 10, {})
            return max_disp_limit * 10, {}

    def objective(x):
        """Amaç: toplam ağırlığı minimize et (proxy: sum of thicknesses)."""
        return np.sum(x)

    def objective_grad(x):
        """Gradyan sabit: her kalınlığın ağırlığa katkısı 1."""
        return np.ones(n)

    def disp_constraint_fun(x):
        """Displacement constraint: max_disp döndürür."""
        max_disp, _ = _eval_point(x, "sqp")
        progress_callback(min(eval_count[0] / max_iter * 100, 99))
        log_callback(f"  SQP Eval #{eval_count[0]}: sum(t)={np.sum(x):.4f}, disp={max_disp:.6f}")
        return max_disp

    def disp_constraint_jac(x):
        """Forward difference ile displacement gradyanı hesaplar."""
        log_callback(f"  FD gradyan hesaplanıyor...")
        x0 = np.array(x, dtype=float)
        f0, _ = _eval_point(x0, "fd_ref")

        if n_parallel > 1:
            # Paralel FD
            batch_items = []
            for i in range(n):
                x_pert = x0.copy()
                x_pert[i] += pid_step[i]
                x_pert[i] = min(x_pert[i], pid_max[i])
                t_map = {pid: float(x_pert[j]) for j, pid in enumerate(pids)}
                w_dir = os.path.join(output_dir, f"fd_grad_{eval_count[0]}_p{i}")
                batch_items.append((t_map, w_dir))

            batch_results = solve_batch(nastran_exe, bdf_path, batch_items, n_parallel, log_callback, memory_mb=memory_mb)
            grad = np.zeros(n)
            for i, (_, pert_disp, _) in enumerate(batch_results):
                grad[i] = (pert_disp - f0) / pid_step[i]
                log_callback(f"    PID {pids[i]}: dDisp/dT = {grad[i]:.6f}")
        else:
            # Seri FD
            grad = np.zeros(n)
            for i in range(n):
                x_pert = x0.copy()
                x_pert[i] += pid_step[i]
                x_pert[i] = min(x_pert[i], pid_max[i])
                fi, _ = _eval_point(x_pert, "fd_pert")
                grad[i] = (fi - f0) / pid_step[i]
                log_callback(f"    PID {pids[i]}: dDisp/dT = {grad[i]:.6f}")

        return grad

    x0 = np.array([(pid_min[i] + pid_max[i]) / 2.0 for i in range(n)])
    bounds = [(pid_min[i], pid_max[i]) for i in range(n)]

    disp_constr = NonlinearConstraint(
        disp_constraint_fun, -np.inf, max_disp_limit,
        jac=disp_constraint_jac
    )

    log_callback(f"  SQP optimizasyon başlıyor...")
    result = minimize(
        objective, x0, method="trust-constr",
        jac=objective_grad,
        bounds=bounds,
        constraints=[disp_constr],
        options={"maxiter": max_iter, "verbose": 0, "gtol": float(np.min(pid_step)) / 10},
    )

    log_callback(f"\n  SQP sonucu: {result.message}")
    final_t = {pid: round(float(result.x[i]), 8) for i, pid in enumerate(pids)}

    final_dir = os.path.join(output_dir, "final_result")
    final_disp, final_stress = solve_and_evaluate(
        nastran_exe, bdf_path, final_t, final_dir, log_callback, memory_mb=memory_mb
    )
    stress_ok, _, max_vm = check_stress_constraints(final_stress, allowable_map)
    tracker.record("Final", final_t, final_disp, max_vm, stress_ok, allowable_map, log_callback)

    progress_callback(100)
    return final_t, final_disp, final_stress


# ---------------------------------------------------------------------------
# ALGORİTMA 2: Genetic Algorithm (GA)
# ---------------------------------------------------------------------------

def optimize_genetic(bdf_path, nastran_exe, output_dir, pids, allowable_map,
                     min_t, max_t, step, max_disp_limit, max_iter,
                     log_callback, progress_callback, tracker, n_parallel=1, memory_mb=0,
                     pid_mid_map=None, mid_density_map=None, pid_area_map=None,
                     pid_bounds=None):
    """Genetik Algoritma ile kalınlık optimizasyonu."""
    n = len(pids)
    eval_count = [0]

    # Per-PID bounds
    if pid_bounds:
        bounds_list = [pid_bounds[pid] for pid in pids]
        pid_min = np.array([b[0] for b in bounds_list])
        pid_max = np.array([b[1] for b in bounds_list])
        pid_step = np.array([b[2] for b in bounds_list])
    else:
        pid_min = np.full(n, min_t)
        pid_max = np.full(n, max_t)
        pid_step = np.full(n, step)

    # GA parametreleri
    pop_size = max(10, 2 * n)
    n_generations = max_iter
    crossover_rate = 0.8
    mutation_rate = 0.2
    elite_count = max(2, pop_size // 5)

    # Gerçek kütle hesabı kullanılabilir mi?
    use_real_mass = (pid_mid_map is not None and mid_density_map is not None
                     and pid_area_map is not None)

    log_callback(f"\n  Genetic Algorithm - {n} PSHELL")
    log_callback(f"  Popülasyon: {pop_size}, Jenerasyon: {n_generations}, Paralel: {n_parallel}")
    log_callback(f"  Crossover: {crossover_rate}, Mutation: {mutation_rate}, Elite: {elite_count}")
    if use_real_mass:
        log_callback(f"  Fitness: Gerçek kütle (t x area x density)")
    else:
        log_callback(f"  Fitness: sum(thicknesses)")

    rng = np.random.RandomState(42)

    def _calc_mass(t_map):
        """Birey için kütle hesabı."""
        if use_real_mass:
            mass, _ = compute_real_mass(t_map, pid_mid_map, mid_density_map, pid_area_map)
            return mass
        return sum(t_map.values())

    def _fitness(t_map, max_disp, stress_ok):
        """Fitness: düşük gerçek kütle + constraint ihlali penaltisi."""
        mass = _calc_mass(t_map)
        penalty = 0.0
        if max_disp > max_disp_limit:
            penalty += 1000.0 * (max_disp - max_disp_limit)
        if not stress_ok:
            penalty += 1000.0
        return mass + penalty

    # İlk popülasyonu oluştur (Latin Hypercube benzeri)
    population = np.zeros((pop_size, n))
    for j in range(n):
        vals = np.linspace(pid_min[j], pid_max[j], pop_size)
        rng.shuffle(vals)
        population[:, j] = vals

    best_t = None
    best_fitness = float('inf')
    best_disp = float('inf')
    best_stress = {}

    for gen in range(n_generations):
        log_callback(f"\n{'='*50}")
        log_callback(f"GA JENERASYON {gen + 1}/{n_generations}")
        progress_callback((gen / n_generations) * 100)

        # Popülasyonu değerlendir
        fitness_scores = np.full(pop_size, float('inf'))
        disp_results = np.zeros(pop_size)
        stress_results = [None] * pop_size
        stress_ok_results = [False] * pop_size
        vm_results = np.zeros(pop_size)
        mass_results = np.zeros(pop_size)
        t_maps = [None] * pop_size

        # Her birey için t_map oluştur
        for i in range(pop_size):
            t_maps[i] = {pid: round(float(population[i, j]), 8) for j, pid in enumerate(pids)}

        if n_parallel > 1:
            # Paralel değerlendirme
            batch_items = []
            for i in range(pop_size):
                w_dir = os.path.join(output_dir, f"ga_gen{gen}_ind{i}")
                batch_items.append((t_maps[i], w_dir))

            log_callback(f"  {pop_size} birey paralel değerlendiriliyor...")
            try:
                batch_results = solve_batch(nastran_exe, bdf_path, batch_items, n_parallel, log_callback, memory_mb=memory_mb)
                for i, (t_map, max_disp, stress_map) in enumerate(batch_results):
                    stress_ok, _, max_vm = check_stress_constraints(stress_map, allowable_map)
                    disp_results[i] = max_disp
                    stress_results[i] = stress_map
                    stress_ok_results[i] = stress_ok
                    vm_results[i] = max_vm
                    mass_results[i] = _calc_mass(t_maps[i])
                    fitness_scores[i] = _fitness(t_maps[i], max_disp, stress_ok)

                    tracker.record(f"GA_G{gen+1}_{i+1}", t_maps[i], max_disp, max_vm, stress_ok,
                                   allowable_map, log_callback)
            except Exception as exc:
                log_callback(f"  Paralel GA hatası: {exc}")
                continue
        else:
            # Seri değerlendirme
            for i in range(pop_size):
                eval_count[0] += 1
                w_dir = os.path.join(output_dir, f"ga_gen{gen}_ind{i}")
                try:
                    max_disp, stress_map = solve_and_evaluate(
                        nastran_exe, bdf_path, t_maps[i], w_dir, log_callback, memory_mb=memory_mb
                    )
                    stress_ok, _, max_vm = check_stress_constraints(stress_map, allowable_map)
                except Exception as exc:
                    log_callback(f"    GA Eval hatası: {exc}")
                    max_disp = max_disp_limit * 10
                    stress_ok = False
                    stress_map = {}
                    max_vm = 0.0

                disp_results[i] = max_disp
                stress_results[i] = stress_map
                stress_ok_results[i] = stress_ok
                vm_results[i] = max_vm
                mass_results[i] = _calc_mass(t_maps[i])
                fitness_scores[i] = _fitness(t_maps[i], max_disp, stress_ok)

                tracker.record(f"GA_G{gen+1}_{i+1}", t_maps[i], max_disp, max_vm, stress_ok,
                               allowable_map, log_callback)

        # ---- Jenerasyon Özeti (satır satır) ----
        log_callback(f"\n  --- Gen {gen + 1} Özet ---")
        mass_unit = "kg" if use_real_mass else "sum(t)"
        log_callback(f"  {'No':>4}  {'Mass':>12}  {'Max Disp':>12}  {'Stress':>8}  {'Feasible':>8}")
        log_callback(f"  {'----':>4}  {'--------':>12}  {'--------':>12}  {'------':>8}  {'--------':>8}")
        for i in range(pop_size):
            feasible = "OK" if (disp_results[i] <= max_disp_limit and stress_ok_results[i]) else "FAIL"
            log_callback(f"  {i+1:>4}  {mass_results[i]:>12.4f}  {disp_results[i]:>12.6f}  {vm_results[i]:>8.1f}  {feasible:>8}")

        # En iyi bireyi güncelle (feasible + en hafif gerçek kütle)
        gen_best_idx = np.argmin(fitness_scores)
        if fitness_scores[gen_best_idx] < best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_t = dict(t_maps[gen_best_idx])
            best_disp = disp_results[gen_best_idx]
            best_stress = stress_results[gen_best_idx] if stress_results[gen_best_idx] else {}

        gen_best_mass = mass_results[gen_best_idx]
        gen_feasible_count = sum(1 for i in range(pop_size)
                                 if disp_results[i] <= max_disp_limit and stress_ok_results[i])
        log_callback(f"\n  Gen {gen+1} en iyi: Birey #{gen_best_idx+1}, "
                     f"Mass={gen_best_mass:.4f} {mass_unit}, "
                     f"Disp={disp_results[gen_best_idx]:.6f}")
        log_callback(f"  Feasible birey: {gen_feasible_count}/{pop_size}")

        # Yeni popülasyon oluştur
        sorted_indices = np.argsort(fitness_scores)
        new_population = np.zeros_like(population)

        # Elitizm: en iyi bireyleri koru
        for i in range(elite_count):
            new_population[i] = population[sorted_indices[i]]

        # Crossover ve mutasyon ile kalan bireyleri oluştur
        for i in range(elite_count, pop_size):
            # Tournament selection
            p1_idx = sorted_indices[rng.randint(0, max(pop_size // 2, 2))]
            p2_idx = sorted_indices[rng.randint(0, max(pop_size // 2, 2))]
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # Crossover (BLX-alpha)
            child = parent1.copy()
            if rng.random() < crossover_rate:
                alpha = 0.5
                for j in range(n):
                    lo = min(parent1[j], parent2[j])
                    hi = max(parent1[j], parent2[j])
                    span = hi - lo
                    child[j] = rng.uniform(lo - alpha * span, hi + alpha * span)

            # Mutasyon
            if rng.random() < mutation_rate:
                mut_idx = rng.randint(0, n)
                child[mut_idx] += rng.normal(0, (pid_max[mut_idx] - pid_min[mut_idx]) * 0.1)

            # Sınırları uygula ve step'e yuvarla
            child = np.clip(child, pid_min, pid_max)
            child = np.round(child / pid_step) * pid_step
            child = np.clip(child, pid_min, pid_max)
            new_population[i] = child

        population = new_population

    if best_t is None:
        best_t = {pid: round(float(population[0, j]), 8) for j, pid in enumerate(pids)}

    log_callback(f"\n{'='*50}")
    log_callback("SON ÇÖZÜM")
    final_dir = os.path.join(output_dir, "final_result")
    final_disp, final_stress = solve_and_evaluate(
        nastran_exe, bdf_path, best_t, final_dir, log_callback, memory_mb=memory_mb
    )
    stress_ok, _, max_vm = check_stress_constraints(final_stress, allowable_map)
    tracker.record("Final", best_t, final_disp, max_vm, stress_ok, allowable_map, log_callback)

    progress_callback(100)
    return best_t, final_disp, final_stress


# ---------------------------------------------------------------------------
# Çıktı dosyası temizleme (saklama moduna göre)
# ---------------------------------------------------------------------------

def cleanup_iteration_outputs(output_dir, retain_mode, tracker, max_disp_limit,
                              allowable_map, log_callback):
    """Saklama moduna göre iterasyon çıktılarını temizler.
    retain_mode:
      'all'      - Hiçbir şey silme
      'feasible' - Sadece kısıtları sağlayan iterasyonların dosyalarını tut
      'optimum'  - Sadece en iyi (en düşük mass, feasible) sonucu tut
    """
    import shutil

    if retain_mode == "all":
        return

    # Çıktı dizinindeki alt klasörleri listele (final_result hariç)
    subdirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item != "final_result":
            subdirs.append(item_path)

    if retain_mode == "optimum":
        # final_result dışındaki tüm alt klasörleri sil
        deleted = 0
        for d in subdirs:
            try:
                shutil.rmtree(d)
                deleted += 1
            except OSError:
                pass
        if deleted:
            log_callback(f"  {deleted} ara iterasyon klasoru silindi (sadece en optimum sonuc saklandi).")

    elif retain_mode == "feasible":
        # Tracker'daki feasible olmayan iterasyonlara ait klasorleri sil
        feasible_indices = set()
        for i in range(len(tracker.iterations)):
            disp = tracker.displacements[i]
            if disp <= max_disp_limit:
                feasible_indices.add(i)

        deleted = 0
        for d in subdirs:
            dirname = os.path.basename(d)
            # Iterasyon numarasini dirname'den cikar
            keep = False
            for idx in feasible_indices:
                iter_num = idx + 1
                if str(iter_num) in dirname:
                    keep = True
                    break
            if not keep:
                try:
                    shutil.rmtree(d)
                    deleted += 1
                except OSError:
                    pass
        if deleted:
            log_callback(f"  {deleted} ara iterasyon klasoru silindi (sadece feasible sonuclar saklandi).")


# ---------------------------------------------------------------------------
# Ana GUI Uygulaması
# ---------------------------------------------------------------------------

class NastranToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eray-Tool | NX Nastran Thickness Optimizasyon Aracı")
        self.root.geometry("1100x750")
        self.root.resizable(True, True)
        self.use_excel_thickness = False
        self.thickness_range_data = None
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

        self.excel_thickness_btn = ttk.Button(
            param_frame, text="Excel'den T Oku",
            command=self._toggle_excel_thickness
        )
        self.excel_thickness_btn.grid(row=0, column=6, padx=4)

        ttk.Label(param_frame, text="Max Disp:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=(4, 0))
        self.max_disp_entry = ttk.Entry(param_frame, width=8)
        self.max_disp_entry.grid(row=1, column=1, padx=2, pady=(4, 0))
        self.max_disp_entry.insert(0, "10.0")

        ttk.Label(param_frame, text="Max Iter:").grid(row=1, column=2, sticky=tk.W, padx=2, pady=(4, 0))
        self.max_iter_entry = ttk.Entry(param_frame, width=8)
        self.max_iter_entry.grid(row=1, column=3, padx=2, pady=(4, 0))
        self.max_iter_entry.insert(0, "20")

        ttk.Label(param_frame, text="Paralel Run:").grid(row=1, column=4, sticky=tk.W, padx=2, pady=(4, 0))
        self.n_parallel_entry = ttk.Entry(param_frame, width=8)
        self.n_parallel_entry.grid(row=1, column=5, padx=2, pady=(4, 0))
        self.n_parallel_entry.insert(0, "1")

        ttk.Label(param_frame, text="Memory (MB):").grid(row=2, column=0, sticky=tk.W, padx=2, pady=(4, 0))
        self.memory_entry = ttk.Entry(param_frame, width=8)
        self.memory_entry.grid(row=2, column=1, padx=2, pady=(4, 0))
        self.memory_entry.insert(0, "0")
        ttk.Label(param_frame, text="(0 = varsayılan)").grid(row=2, column=2, sticky=tk.W, padx=2, pady=(4, 0))

        ttk.Label(param_frame, text="Total Run:").grid(row=2, column=3, sticky=tk.W, padx=2, pady=(4, 0))
        self.total_run_var = tk.StringVar(value="-")
        ttk.Label(param_frame, textvariable=self.total_run_var, width=8,
                  relief="sunken", anchor=tk.CENTER).grid(row=2, column=4, padx=2, pady=(4, 0))
        ttk.Button(param_frame, text="Hesapla", width=8,
                   command=self._calc_total_run).grid(row=2, column=5, padx=2, pady=(4, 0))

        ttk.Label(param_frame, text="Algoritma:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=(4, 0))
        self.algo_var = tk.StringVar()
        algo_combo = ttk.Combobox(param_frame, textvariable=self.algo_var, state="readonly", width=32)
        algo_combo["values"] = (
            "FD + SQP (Forward Difference)",
            "Genetic Algorithm (GA)",
        )
        algo_combo.current(0)
        algo_combo.grid(row=3, column=1, columnspan=4, sticky=tk.W, padx=2, pady=(4, 0))

        ttk.Label(param_frame, text="Sonuç Saklama:").grid(row=4, column=0, sticky=tk.W, padx=2, pady=(4, 0))
        self.output_retain_var = tk.StringVar()
        retain_combo = ttk.Combobox(param_frame, textvariable=self.output_retain_var, state="readonly", width=32)
        retain_combo["values"] = (
            "Sadece En Optimum Sonuc",
            "Kriterleri Saglayan Sonuclar",
            "Butun Sonuclar",
        )
        retain_combo.current(2)
        retain_combo.grid(row=4, column=1, columnspan=4, sticky=tk.W, padx=2, pady=(4, 0))

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

    def _toggle_excel_thickness(self):
        """Excel'den Thickness Range okuma modunu aç/kapat."""
        excel_path = self.excel_entry.get().strip()
        if not excel_path or not os.path.isfile(excel_path):
            messagebox.showerror("Hata", "Oncelikle Allowable Excel dosyasini seciniz.")
            return

        if not self.use_excel_thickness:
            try:
                self.thickness_range_data = load_thickness_range_excel(excel_path)
                if not self.thickness_range_data:
                    messagebox.showerror("Hata", "Thickness Range sayfasi bos!")
                    return
                self.use_excel_thickness = True
                self.min_t_entry.configure(state=tk.DISABLED)
                self.max_t_entry.configure(state=tk.DISABLED)
                self.step_entry.configure(state=tk.DISABLED)
                self.excel_thickness_btn.configure(text="Manuel T'ye Don")
                self._log(f"  Excel'den {len(self.thickness_range_data)} PID icin thickness range yuklendi.")
            except Exception as exc:
                messagebox.showerror("Hata", f"Thickness Range okunamadi: {exc}")
        else:
            self.use_excel_thickness = False
            self.thickness_range_data = None
            self.min_t_entry.configure(state=tk.NORMAL)
            self.max_t_entry.configure(state=tk.NORMAL)
            self.step_entry.configure(state=tk.NORMAL)
            self.excel_thickness_btn.configure(text="Excel'den T Oku")
            self._log("  Manuel thickness parametrelerine geri donuldu.")

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

    def _calc_total_run(self):
        # BDF dosyasından n_pids al
        bdf_path = self.bdf_entry.get().strip()
        if not bdf_path or not os.path.isfile(bdf_path):
            messagebox.showerror("Hata", "Total Run hesabı için BDF dosyası gereklidir.")
            return

        try:
            min_t = float(self.min_t_entry.get().strip())
            max_t = float(self.max_t_entry.get().strip())
            step = float(self.step_entry.get().strip())
            max_iter = int(self.max_iter_entry.get().strip())
            if step <= 0 or min_t >= max_t:
                messagebox.showerror("Hata", "Min < Max ve Step > 0 olmalıdır.")
                return
        except ValueError:
            messagebox.showerror("Hata", "Parametre değerleri sayısal olmalıdır.")
            return

        algo_name = self.algo_var.get()

        # BDF'den PSHELL sayısını oku
        try:
            self._log("BDF okunuyor (PSHELL sayısı için)...")
            pids, _, _, _ = read_bdf_model(bdf_path)
            n_pids = len(pids)
            if n_pids == 0:
                messagebox.showerror("Hata", "BDF dosyasında PSHELL bulunamadı.")
                return
            self._log(f"  {n_pids} PSHELL bulundu.")
        except Exception as exc:
            messagebox.showerror("Hata", f"BDF okuma hatası: {exc}")
            return

        # Thickness aralığına göre yardımcı değerler
        import math
        t_range = max_t - min_t
        n_steps = int(round(t_range / step))  # bir PSHELL'in max-min arası adım sayısı

        # Algoritma bazlı tahmini run hesabı
        if "FD" in algo_name or "SQP" in algo_name:
            # Her iterasyonda: 1 ref + n_pids FD pertürbasyon
            # gtol=step/10: küçük step -> daha hassas -> daha çok iter
            # Ancak optimizer max_iter'den fazla çalışmaz
            total_run = (1 + n_pids) * max_iter + 1
            detail = (f"({1 + n_pids}) x {max_iter} iter + 1 final"
                      f"  [gtol={step/10:.4f}, cache ile daha az olabilir]")

        elif "Genetic" in algo_name or "GA" in algo_name:
            # pop_size * n_generations + 1 final
            # step sadece yuvarlama için kullanılır, run sayısını değiştirmez
            pop_size = max(10, 2 * n_pids)
            total_run = pop_size * max_iter + 1
            detail = f"{pop_size} pop x {max_iter} gen + 1 final  [step={step} yuvarlama]"

        else:
            total_run = max_iter + 1
            detail = "bilinmeyen algoritma"

        self.total_run_var.set(str(total_run))
        self._log(f"Total Run hesaplandı: {total_run}")
        self._log(f"  Algoritma: {algo_name}")
        self._log(f"  n_pids={n_pids}, max_iter={max_iter}, min_t={min_t}, max_t={max_t}, step={step}")
        self._log(f"  Thickness aralığı: {n_steps} adım ({min_t} -> {max_t}, step={step})")
        self._log(f"  Detay: {detail}")

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

        thickness_range_data = None
        if self.use_excel_thickness and self.thickness_range_data:
            thickness_range_data = self.thickness_range_data
            # Excel modunda dummy değerler (kullanılmayacak, per-PID'den alınacak)
            min_t = 0.0
            max_t = 1.0
            step = 0.1
        else:
            try:
                min_t = float(self.min_t_entry.get().strip())
                max_t = float(self.max_t_entry.get().strip())
                step = float(self.step_entry.get().strip())
            except ValueError:
                messagebox.showerror("Hata", "Parametre değerleri sayısal olmalıdır.")
                return
            if min_t >= max_t or step <= 0:
                messagebox.showerror("Hata", "Min < Max ve Step > 0 olmalıdır.")
                return

        try:
            max_disp_limit = float(self.max_disp_entry.get().strip())
            max_iter = int(self.max_iter_entry.get().strip())
            n_parallel = int(self.n_parallel_entry.get().strip())
            if n_parallel < 1:
                n_parallel = 1
            memory_mb = int(self.memory_entry.get().strip())
            if memory_mb < 0:
                memory_mb = 0
        except ValueError:
            messagebox.showerror("Hata", "Parametre değerleri sayısal olmalıdır.")
            return

        algo_name = self.algo_var.get()

        retain_text = self.output_retain_var.get()
        if "Optimum" in retain_text:
            retain_mode = "optimum"
        elif "Kriterleri" in retain_text:
            retain_mode = "feasible"
        else:
            retain_mode = "all"

        self._disable_buttons()
        self.progress.configure(mode="determinate", value=0)

        threading.Thread(
            target=self._opt_worker,
            args=(bdf_path, nastran_exe, output_dir, excel_path,
                  min_t, max_t, step, max_disp_limit, max_iter, algo_name, n_parallel, memory_mb,
                  retain_mode, thickness_range_data),
            daemon=True,
        ).start()

    def _opt_worker(self, bdf_path, nastran_exe, output_dir, excel_path,
                    min_t, max_t, step, max_disp_limit, max_iter, algo_name, n_parallel, memory_mb,
                    retain_mode="all", thickness_range_data=None):
        try:
            self._log("=" * 60)
            self._log(f"OPTİMİZASYON: {algo_name}")
            if thickness_range_data:
                self._log(f"  Thickness: Excel'den PID bazli (Thickness Range)")
            else:
                self._log(f"  Min: {min_t}  Max: {max_t}  Step: {step}")
            self._log(f"  Max Disp: {max_disp_limit}  Max Iter: {max_iter}  Paralel: {n_parallel}  Memory: {memory_mb}mb" if memory_mb > 0 else f"  Max Disp: {max_disp_limit}  Max Iter: {max_iter}  Paralel: {n_parallel}")
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

            # Excel'den PID bazli thickness range varsa, per-PID bounds olustur
            if thickness_range_data:
                pid_bounds = {}
                for pid in pids:
                    if pid in thickness_range_data:
                        pid_bounds[pid] = thickness_range_data[pid]
                    else:
                        pid_bounds[pid] = (min_t, max_t, step)
                        self._log(f"  UYARI: PID {pid} Excel'de yok, varsayilan degerler kullanilacak.")
                # Global min/max/step algoritmaya geçmek için ortalamaları al
                all_mins = [pid_bounds[p][0] for p in pids]
                all_maxs = [pid_bounds[p][1] for p in pids]
                all_steps = [pid_bounds[p][2] for p in pids]
                min_t = min(all_mins)
                max_t = max(all_maxs)
                step = min(all_steps)
            else:
                pid_bounds = {pid: (min_t, max_t, step) for pid in pids}

            allowable_map = load_allowable_excel(excel_path)
            self._log(f"  {len(allowable_map)} element allowable yüklendi.")

            # Tracker oluştur
            tracker = IterationTracker(
                pid_mid_map, mid_density_map, pid_area_map,
                max_disp_limit, self._update_plot
            )

            # Algoritma çalıştır
            algo_kwargs = dict(
                bdf_path=bdf_path, nastran_exe=nastran_exe, output_dir=output_dir,
                pids=pids, allowable_map=allowable_map,
                min_t=min_t, max_t=max_t, step=step,
                max_disp_limit=max_disp_limit, max_iter=max_iter,
                log_callback=self._log, progress_callback=self._set_progress,
                tracker=tracker, n_parallel=n_parallel, memory_mb=memory_mb,
                pid_bounds=pid_bounds,
            )

            if "FD" in algo_name or "SQP" in algo_name:
                result_t, final_disp, final_stress = optimize_fd_sqp(**algo_kwargs)
            else:
                algo_kwargs["pid_mid_map"] = pid_mid_map
                algo_kwargs["mid_density_map"] = mid_density_map
                algo_kwargs["pid_area_map"] = pid_area_map
                result_t, final_disp, final_stress = optimize_genetic(**algo_kwargs)

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

            # Çıktı dosyalarını saklama moduna göre temizle
            cleanup_iteration_outputs(
                output_dir, retain_mode, tracker, max_disp_limit,
                allowable_map, self._log
            )

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
