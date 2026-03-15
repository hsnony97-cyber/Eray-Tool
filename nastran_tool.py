"""
Eray-Tool: NX Nastran Thickness Iteration & OP2 Results Extraction Tool

BDF dosyasındaki tüm PSHELL kalınlıklarını min'den max'a iterasyon ile değiştirerek
NX Nastran ile çözer. Her iterasyonda:
  - Max absolute displacement kontrolü
  - Element bazlı Von Mises stress allowable kontrolü
yaparak en uygun kalınlığı bulur. Sonuçları CSV olarak dışarı aktarır.
"""

import os
import re
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

import numpy as np
import pandas as pd
from pyNastran.op2.op2 import OP2


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


# ---------------------------------------------------------------------------
# BDF PSHELL thickness değiştirme
# ---------------------------------------------------------------------------

def modify_pshell_thickness(bdf_path, output_bdf_path, new_thickness, log_callback):
    """
    BDF dosyasındaki tüm PSHELL kartlarının kalınlık (T) alanını değiştirir.
    Nastran fixed-format (8 karakter alan) ve free-format (virgüllü) destekler.
    """
    with open(bdf_path, "r") as f:
        lines = f.readlines()

    modified_count = 0
    new_lines = []
    thickness_str = f"{new_thickness:.6g}"

    for line in lines:
        # Free-format: PSHELL ile başlayan ve virgül içeren satırlar
        if line.upper().startswith("PSHELL") and "," in line:
            parts = line.split(",")
            # PSHELL, PID, MID1, T, ...
            if len(parts) >= 4:
                parts[3] = thickness_str
                new_lines.append(",".join(parts))
                modified_count += 1
            else:
                new_lines.append(line)
        # Fixed-format: 8 karakter genişliğinde alanlar
        elif line.upper().startswith("PSHELL"):
            # Alan 1: PSHELL (0:8), Alan 2: PID (8:16), Alan 3: MID1 (16:24), Alan 4: T (24:32)
            if len(line) >= 32:
                t_field = f"{new_thickness:8g}"
                new_line = line[:24] + t_field + line[32:]
                new_lines.append(new_line)
                modified_count += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(output_bdf_path, "w") as f:
        f.writelines(new_lines)

    log_callback(f"  {modified_count} PSHELL kartının kalınlığı {new_thickness} olarak değiştirildi.")
    return modified_count


# ---------------------------------------------------------------------------
# Nastran çalıştırma
# ---------------------------------------------------------------------------

def run_nastran(nastran_exe, bdf_path, output_dir, log_callback):
    bdf_basename = os.path.splitext(os.path.basename(bdf_path))[0]
    op2_path = os.path.join(output_dir, bdf_basename + ".op2")

    cmd = [
        nastran_exe,
        bdf_path,
        f"out={output_dir}{os.sep}{bdf_basename}",
    ]

    log_callback(f"  Nastran komutu: {' '.join(cmd)}")
    log_callback("  Nastran çalışıyor...")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    if result.stdout:
        log_callback(result.stdout[-300:])

    if result.returncode != 0:
        log_callback(f"  Nastran return code: {result.returncode}")
        if result.stderr:
            log_callback(f"  Nastran stderr: {result.stderr[-300:]}")

    if not os.path.isfile(op2_path):
        raise FileNotFoundError(
            f"OP2 dosyası bulunamadı: {op2_path}\n"
            "BDF dosyanızda 'PARAM,POST,-1' olduğundan emin olunuz."
        )

    log_callback(f"  OP2 oluşturuldu: {op2_path}")
    return op2_path


# ---------------------------------------------------------------------------
# OP2 sonuç okuma
# ---------------------------------------------------------------------------

def get_von_mises_stresses(op2_model):
    """OP2'den element bazlı Von Mises streslerini dict olarak döndürür: {eid: max_vm}"""
    stress_map = {}

    plate_stress_attrs = [
        ("cquad4_stress", "CQUAD4"),
        ("ctria3_stress", "CTRIA3"),
    ]

    for attr_name, elem_type in plate_stress_attrs:
        stress_dict = getattr(op2_model, attr_name, None)
        if not stress_dict:
            continue
        for subcase_id, stress_obj in stress_dict.items():
            if hasattr(stress_obj, "element_node"):
                eids = stress_obj.element_node[:, 0]
                node_ids = stress_obj.element_node[:, 1]
                center_mask = node_ids == 0
            elif hasattr(stress_obj, "element"):
                eids = stress_obj.element
                center_mask = None
            else:
                continue

            ovm = stress_obj.data  # (ntimes, nelements, nresults)

            for t_idx in range(ovm.shape[0]):
                for i in range(ovm.shape[1]):
                    if center_mask is not None and not center_mask[i]:
                        continue
                    eid = int(eids[i])
                    vm_val = float(ovm[t_idx, i, -1])
                    # Her eleman için maksimum VM değerini tut
                    if eid not in stress_map or vm_val > stress_map[eid]:
                        stress_map[eid] = vm_val

    return stress_map


def get_max_absolute_displacement(op2_model):
    """OP2'den tüm nodeların max absolute displacement değerini döndürür."""
    max_disp = 0.0

    if not op2_model.displacements:
        return 0.0

    for subcase_id, disp_obj in op2_model.displacements.items():
        data = disp_obj.data  # (ntimes, nnodes, 6) -> T1,T2,T3,R1,R2,R3
        # Translasyonel displacement: T1, T2, T3 (ilk 3 sütun)
        translations = data[:, :, :3]
        magnitudes = np.sqrt(np.sum(translations ** 2, axis=2))
        current_max = float(np.max(magnitudes))
        if current_max > max_disp:
            max_disp = current_max

    return max_disp


def extract_von_mises_stress_csv(op2_model, output_dir, thickness, log_callback):
    """Von Mises streslerini CSV'ye yazar."""
    stress_map = get_von_mises_stresses(op2_model)
    if not stress_map:
        log_callback("  UYARI: Stress verisi bulunamadı.")
        return None

    rows = [{"ElementID": eid, "VonMises": vm, "Thickness": thickness}
            for eid, vm in sorted(stress_map.items())]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"von_mises_stress_t{thickness:.4g}.csv")
    df.to_csv(csv_path, index=False)
    log_callback(f"  Stress CSV yazıldı: {csv_path} ({len(df)} eleman)")
    return csv_path


def extract_displacements_csv(op2_model, output_dir, thickness, log_callback):
    """Displacement sonuçlarını CSV'ye yazar."""
    disp_data = []

    if not op2_model.displacements:
        log_callback("  UYARI: Displacement verisi bulunamadı.")
        return None

    for subcase_id, disp_obj in op2_model.displacements.items():
        node_ids = disp_obj.node_gridtype[:, 0]
        data = disp_obj.data

        for t_idx in range(data.shape[0]):
            for i, nid in enumerate(node_ids):
                row = data[t_idx, i, :]
                disp_data.append({
                    "Subcase": subcase_id,
                    "NodeID": int(nid),
                    "T1": row[0], "T2": row[1], "T3": row[2],
                    "R1": row[3], "R2": row[4], "R3": row[5],
                    "Magnitude": float(np.sqrt(row[0]**2 + row[1]**2 + row[2]**2)),
                })

    df = pd.DataFrame(disp_data)
    csv_path = os.path.join(output_dir, f"displacements_t{thickness:.4g}.csv")
    df.to_csv(csv_path, index=False)
    log_callback(f"  Displacement CSV yazıldı: {csv_path} ({len(df)} satır)")
    return csv_path


def load_allowable_excel(excel_path):
    """Stress allowable Excel dosyasını okur. {element_id: allowable} dict döndürür."""
    df = pd.read_excel(excel_path, sheet_name=0)
    # Sütun isimlerini normalize et
    df.columns = [c.strip() for c in df.columns]

    # İlk sütun Element ID, ikinci sütun Allowable
    eid_col = df.columns[0]
    allow_col = df.columns[1]

    allowable_map = {}
    for _, row in df.iterrows():
        eid = int(row[eid_col])
        allow = float(row[allow_col])
        allowable_map[eid] = allow

    return allowable_map


# ---------------------------------------------------------------------------
# Ana GUI Uygulaması
# ---------------------------------------------------------------------------

class NastranToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eray-Tool | NX Nastran Thickness Iterasyon Aracı")
        self.root.geometry("800x650")
        self.root.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        row = 0
        # --- BDF Dosyası ---
        ttk.Label(main_frame, text="BDF Dosyası:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.bdf_entry = ttk.Entry(main_frame, width=55)
        self.bdf_entry.grid(row=row, column=1, columnspan=3, sticky=tk.EW, padx=4)
        ttk.Button(main_frame, text="Seç...", command=lambda: select_file(
            self.bdf_entry, "BDF Dosyası Seçiniz",
            [("BDF Dosyaları", "*.bdf *.dat *.nas"), ("Tüm Dosyalar", "*.*")]
        )).grid(row=row, column=4)

        # --- NX Nastran Exe ---
        row += 1
        ttk.Label(main_frame, text="NX Nastran:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.nastran_entry = ttk.Entry(main_frame, width=55)
        self.nastran_entry.grid(row=row, column=1, columnspan=3, sticky=tk.EW, padx=4)
        ttk.Button(main_frame, text="Seç...", command=lambda: select_file(
            self.nastran_entry, "NX Nastran Seçiniz",
            [("Çalıştırılabilir", "*.exe *.bat"), ("Tüm Dosyalar", "*.*")]
        )).grid(row=row, column=4)

        # --- Çıktı Klasörü ---
        row += 1
        ttk.Label(main_frame, text="Çıktı Klasörü:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.output_entry = ttk.Entry(main_frame, width=55)
        self.output_entry.grid(row=row, column=1, columnspan=3, sticky=tk.EW, padx=4)
        ttk.Button(main_frame, text="Seç...", command=lambda: select_directory(
            self.output_entry, "Çıktı Klasörü Seçiniz"
        )).grid(row=row, column=4)

        # --- Stress Allowable Excel ---
        row += 1
        ttk.Label(main_frame, text="Allowable Excel:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.excel_entry = ttk.Entry(main_frame, width=55)
        self.excel_entry.grid(row=row, column=1, columnspan=3, sticky=tk.EW, padx=4)
        ttk.Button(main_frame, text="Seç...", command=lambda: select_file(
            self.excel_entry, "Stress Allowable Excel Seçiniz",
            [("Excel Dosyaları", "*.xlsx *.xls"), ("Tüm Dosyalar", "*.*")]
        )).grid(row=row, column=4)

        # --- Separator ---
        row += 1
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=5, sticky=tk.EW, pady=8
        )

        # --- Iterasyon Parametreleri ---
        row += 1
        param_frame = ttk.LabelFrame(main_frame, text="İterasyon Parametreleri", padding=8)
        param_frame.grid(row=row, column=0, columnspan=5, sticky=tk.EW, pady=4)

        # Min Thickness
        ttk.Label(param_frame, text="Min Kalınlık:").grid(row=0, column=0, sticky=tk.W, padx=4)
        self.min_t_entry = ttk.Entry(param_frame, width=10)
        self.min_t_entry.grid(row=0, column=1, padx=4)
        self.min_t_entry.insert(0, "0.5")

        # Max Thickness
        ttk.Label(param_frame, text="Max Kalınlık:").grid(row=0, column=2, sticky=tk.W, padx=4)
        self.max_t_entry = ttk.Entry(param_frame, width=10)
        self.max_t_entry.grid(row=0, column=3, padx=4)
        self.max_t_entry.insert(0, "5.0")

        # Step
        ttk.Label(param_frame, text="Adım (Step):").grid(row=0, column=4, sticky=tk.W, padx=4)
        self.step_entry = ttk.Entry(param_frame, width=10)
        self.step_entry.grid(row=0, column=5, padx=4)
        self.step_entry.insert(0, "0.1")

        # Max Displacement
        ttk.Label(param_frame, text="Max Disp:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=(6, 0))
        self.max_disp_entry = ttk.Entry(param_frame, width=10)
        self.max_disp_entry.grid(row=1, column=1, padx=4, pady=(6, 0))
        self.max_disp_entry.insert(0, "10.0")

        main_frame.columnconfigure(1, weight=1)

        # --- Butonlar ---
        row += 1
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=5, pady=10)

        self.run_btn = ttk.Button(btn_frame, text="Iterasyonu Başlat", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT, padx=8)

        self.single_btn = ttk.Button(btn_frame, text="Tek Çözüm (Mevcut BDF)", command=self._on_single_run)
        self.single_btn.pack(side=tk.LEFT, padx=8)

        # --- İlerleme ---
        row += 1
        self.progress = ttk.Progressbar(main_frame, mode="determinate")
        self.progress.grid(row=row, column=0, columnspan=5, sticky=tk.EW, pady=(0, 4))

        # --- Log ---
        row += 1
        ttk.Label(main_frame, text="İşlem Günlüğü:").grid(row=row, column=0, sticky=tk.W)
        row += 1
        self.log_text = tk.Text(main_frame, height=14, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.grid(row=row, column=0, columnspan=4, sticky=tk.NSEW)
        main_frame.rowconfigure(row, weight=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=row, column=4, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scrollbar.set)

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
            messagebox.showerror("Hata", "Geçerli bir NX Nastran çalıştırılabilir dosyası seçiniz.")
            return None
        if not output_dir:
            messagebox.showerror("Hata", "Çıktı klasörü seçiniz.")
            return None
        os.makedirs(output_dir, exist_ok=True)
        return bdf_path, nastran_exe, output_dir

    # --- Tek çözüm (iterasyonsuz, eski davranış) ---
    def _on_single_run(self):
        vals = self._validate_common()
        if not vals:
            return
        bdf_path, nastran_exe, output_dir = vals
        self.run_btn.configure(state=tk.DISABLED)
        self.single_btn.configure(state=tk.DISABLED)
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)
        thread = threading.Thread(
            target=self._single_worker, args=(bdf_path, nastran_exe, output_dir), daemon=True
        )
        thread.start()

    def _single_worker(self, bdf_path, nastran_exe, output_dir):
        try:
            self._log("=" * 50)
            self._log("TEK ÇÖZÜM MODU")
            self._log("=" * 50)

            op2_path = run_nastran(nastran_exe, bdf_path, output_dir, self._log)

            self._log("\nOP2 okunuyor...")
            op2_model = OP2()
            op2_model.read_op2(op2_path)

            self._log("\nStress çıkarılıyor...")
            extract_von_mises_stress_csv(op2_model, output_dir, 0, self._log)

            self._log("\nDisplacement çıkarılıyor...")
            extract_displacements_csv(op2_model, output_dir, 0, self._log)

            max_disp = get_max_absolute_displacement(op2_model)
            self._log(f"\nMax Absolute Displacement: {max_disp:.6f}")

            self._log("\nİŞLEM TAMAMLANDI!")
            self.root.after(0, lambda: messagebox.showinfo("Başarılı", "Sonuçlar CSV olarak yazıldı!"))
        except Exception as exc:
            error_msg = str(exc)
            self._log(f"\nHATA: {error_msg}")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Hata", msg))
        finally:
            self.root.after(0, self._finish)

    # --- İterasyon modu ---
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
        except ValueError:
            messagebox.showerror("Hata", "Min/Max kalınlık, adım ve max disp değerleri sayısal olmalıdır.")
            return

        if min_t >= max_t or step <= 0:
            messagebox.showerror("Hata", "Min < Max ve Step > 0 olmalıdır.")
            return

        self.run_btn.configure(state=tk.DISABLED)
        self.single_btn.configure(state=tk.DISABLED)
        self.progress.configure(mode="determinate", value=0)

        thread = threading.Thread(
            target=self._iteration_worker,
            args=(bdf_path, nastran_exe, output_dir, excel_path, min_t, max_t, step, max_disp_limit),
            daemon=True,
        )
        thread.start()

    def _iteration_worker(self, bdf_path, nastran_exe, output_dir, excel_path,
                          min_t, max_t, step, max_disp_limit):
        try:
            self._log("=" * 60)
            self._log("THICKNESS İTERASYON MODU")
            self._log(f"  Min: {min_t}  Max: {max_t}  Step: {step}")
            self._log(f"  Max Displacement Limiti: {max_disp_limit}")
            self._log("=" * 60)

            # Allowable Excel oku
            self._log("\nAllowable Excel okunuyor...")
            allowable_map = load_allowable_excel(excel_path)
            self._log(f"  {len(allowable_map)} element için allowable değeri yüklendi.")

            # Thickness değerlerini oluştur
            thicknesses = []
            t = min_t
            while t <= max_t + 1e-9:
                thicknesses.append(round(t, 8))
                t += step
            total_steps = len(thicknesses)
            self._log(f"  Toplam {total_steps} iterasyon yapılacak.\n")

            # Sonuç özeti
            summary_rows = []
            passed_thickness = None

            for idx, thickness in enumerate(thicknesses):
                self._log("-" * 50)
                self._log(f"İTERASYON {idx + 1}/{total_steps} | Kalınlık = {thickness}")
                self._set_progress((idx / total_steps) * 100)

                # 1) BDF'yi kopyala ve kalınlığı değiştir
                iter_dir = os.path.join(output_dir, f"iter_t{thickness:.4g}")
                os.makedirs(iter_dir, exist_ok=True)

                bdf_basename = os.path.basename(bdf_path)
                modified_bdf = os.path.join(iter_dir, bdf_basename)
                modify_pshell_thickness(bdf_path, modified_bdf, thickness, self._log)

                # 2) Nastran çalıştır
                try:
                    op2_path = run_nastran(nastran_exe, modified_bdf, iter_dir, self._log)
                except Exception as run_exc:
                    self._log(f"  UYARI: Nastran hatası, bu iterasyon atlanıyor: {run_exc}")
                    summary_rows.append({
                        "Thickness": thickness,
                        "MaxDisplacement": None,
                        "MaxVonMises": None,
                        "DispOK": False,
                        "StressOK": False,
                        "Status": "NASTRAN HATASI",
                    })
                    continue

                # 3) OP2 oku
                op2_model = OP2()
                op2_model.read_op2(op2_path)

                # 4) CSV'leri yaz
                extract_von_mises_stress_csv(op2_model, iter_dir, thickness, self._log)
                extract_displacements_csv(op2_model, iter_dir, thickness, self._log)

                # 5) Displacement kontrolü
                max_disp = get_max_absolute_displacement(op2_model)
                disp_ok = max_disp <= max_disp_limit
                self._log(f"  Max Disp: {max_disp:.6f} (Limit: {max_disp_limit}) -> {'OK' if disp_ok else 'FAIL'}")

                # 6) Stress kontrolü
                stress_map = get_von_mises_stresses(op2_model)
                stress_ok = True
                max_vm = 0.0
                failed_elements = []

                for eid, vm_val in stress_map.items():
                    if vm_val > max_vm:
                        max_vm = vm_val
                    if eid in allowable_map:
                        if vm_val > allowable_map[eid]:
                            stress_ok = False
                            failed_elements.append((eid, vm_val, allowable_map[eid]))

                if failed_elements:
                    self._log(f"  Stress FAIL: {len(failed_elements)} eleman allowable'ı aşıyor.")
                    # İlk 5 tanesini göster
                    for eid, vm, allow in failed_elements[:5]:
                        self._log(f"    Element {eid}: VM={vm:.2f} > Allowable={allow:.2f}")
                    if len(failed_elements) > 5:
                        self._log(f"    ... ve {len(failed_elements) - 5} eleman daha")
                else:
                    self._log(f"  Stress: OK (Max VM: {max_vm:.2f})")

                status = "PASS" if (disp_ok and stress_ok) else "FAIL"
                self._log(f"  SONUÇ: {status}")

                summary_rows.append({
                    "Thickness": thickness,
                    "MaxDisplacement": max_disp,
                    "MaxVonMises": max_vm,
                    "DispOK": disp_ok,
                    "StressOK": stress_ok,
                    "Status": status,
                })

                if disp_ok and stress_ok and passed_thickness is None:
                    passed_thickness = thickness

            # Sonuç özeti CSV
            self._set_progress(100)
            self._log("\n" + "=" * 60)

            summary_df = pd.DataFrame(summary_rows)
            summary_csv = os.path.join(output_dir, "iteration_summary.csv")
            summary_df.to_csv(summary_csv, index=False)
            self._log(f"İterasyon özeti yazıldı: {summary_csv}")

            if passed_thickness is not None:
                self._log(f"\nSONUÇ: Tüm koşulları sağlayan minimum kalınlık = {passed_thickness}")
                self.root.after(0, lambda t=passed_thickness: messagebox.showinfo(
                    "İterasyon Tamamlandı",
                    f"Koşulları sağlayan minimum kalınlık: {t}\n\n"
                    f"Detaylar: {summary_csv}"
                ))
            else:
                self._log("\nSONUÇ: Hiçbir kalınlık tüm koşulları sağlayamadı!")
                self.root.after(0, lambda: messagebox.showwarning(
                    "İterasyon Tamamlandı",
                    "Hiçbir kalınlık değeri tüm koşulları sağlayamadı.\n"
                    "Max kalınlığı artırmayı deneyiniz."
                ))

            self._log("İŞLEM TAMAMLANDI!")

        except Exception as exc:
            error_msg = str(exc)
            self._log(f"\nHATA: {error_msg}")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Hata", msg))
        finally:
            self.root.after(0, self._finish)

    def _finish(self):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.run_btn.configure(state=tk.NORMAL)
        self.single_btn.configure(state=tk.NORMAL)


def main():
    root = tk.Tk()
    NastranToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
