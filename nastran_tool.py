"""
Eray-Tool: NX Nastran OP2 Results Extraction Tool

BDF dosyasını NX Nastran ile çözer, OP2 sonuçlarından:
  - PSHELL elemanlarının Von Mises streslerini
  - Tüm nodeların displacement değerlerini
CSV olarak dışarı aktarır.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

import pandas as pd
from pyNastran.op2.op2 import OP2


def select_file(entry, title, filetypes):
    """Dosya seçme dialogu açar ve sonucu entry widget'a yazar."""
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def select_directory(entry, title):
    """Klasör seçme dialogu açar ve sonucu entry widget'a yazar."""
    path = filedialog.askdirectory(title=title)
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def run_nastran(nastran_exe, bdf_path, output_dir, log_callback):
    """NX Nastran'ı çalıştırır ve OP2 dosya yolunu döndürür."""
    bdf_basename = os.path.splitext(os.path.basename(bdf_path))[0]
    op2_path = os.path.join(output_dir, bdf_basename + ".op2")

    cmd = [
        nastran_exe,
        bdf_path,
        f"out={output_dir}{os.sep}{bdf_basename}",
    ]

    log_callback(f"Nastran komutu: {' '.join(cmd)}")
    log_callback("Nastran çalışıyor, lütfen bekleyiniz...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=output_dir,
    )

    if result.stdout:
        log_callback(result.stdout[-500:])
    if result.returncode != 0:
        stderr_msg = result.stderr[-500:] if result.stderr else "Bilinmeyen hata"
        raise RuntimeError(f"Nastran hata kodu {result.returncode}: {stderr_msg}")

    if not os.path.isfile(op2_path):
        raise FileNotFoundError(
            f"OP2 dosyası bulunamadı: {op2_path}\n"
            "Nastran çalıştı ancak OP2 üretmemiş olabilir. "
            "BDF dosyanızda 'PARAM,POST,-1' olduğundan emin olunuz."
        )

    log_callback(f"OP2 dosyası oluşturuldu: {op2_path}")
    return op2_path


def extract_von_mises_stress(op2_model, output_dir, log_callback):
    """PSHELL elemanlarının Von Mises streslerini CSV'ye yazar."""
    stress_data = []

    # CQUAD4 streslerini kontrol et
    if hasattr(op2_model, "cquad4_stress") and op2_model.cquad4_stress:
        for subcase_id, stress_obj in op2_model.cquad4_stress.items():
            log_callback(f"  CQUAD4 stress bulundu (subcase {subcase_id})")
            eids = stress_obj.element
            ovm = stress_obj.data  # (ntimes, nelements, nresults)
            for t_idx in range(ovm.shape[0]):
                for i, eid in enumerate(eids):
                    # Von Mises index genelde son sütun
                    vm_val = ovm[t_idx, i, -1]
                    stress_data.append({
                        "Subcase": subcase_id,
                        "TimeStep": t_idx,
                        "ElementID": eid,
                        "ElementType": "CQUAD4",
                        "VonMises": vm_val,
                    })

    # CTRIA3 streslerini kontrol et
    if hasattr(op2_model, "ctria3_stress") and op2_model.ctria3_stress:
        for subcase_id, stress_obj in op2_model.ctria3_stress.items():
            log_callback(f"  CTRIA3 stress bulundu (subcase {subcase_id})")
            eids = stress_obj.element
            ovm = stress_obj.data
            for t_idx in range(ovm.shape[0]):
                for i, eid in enumerate(eids):
                    vm_val = ovm[t_idx, i, -1]
                    stress_data.append({
                        "Subcase": subcase_id,
                        "TimeStep": t_idx,
                        "ElementID": eid,
                        "ElementType": "CTRIA3",
                        "VonMises": vm_val,
                    })

    if not stress_data:
        log_callback("UYARI: OP2 dosyasında PSHELL (CQUAD4/CTRIA3) stress verisi bulunamadı.")
        return None

    df = pd.DataFrame(stress_data)
    csv_path = os.path.join(output_dir, "von_mises_stress.csv")
    df.to_csv(csv_path, index=False)
    log_callback(f"Von Mises stress CSV yazıldı: {csv_path} ({len(df)} satır)")
    return csv_path


def extract_displacements(op2_model, output_dir, log_callback):
    """Tüm nodeların displacement sonuçlarını CSV'ye yazar."""
    disp_data = []

    if not op2_model.displacements:
        log_callback("UYARI: OP2 dosyasında displacement verisi bulunamadı.")
        return None

    for subcase_id, disp_obj in op2_model.displacements.items():
        log_callback(f"  Displacement bulundu (subcase {subcase_id})")
        node_ids = disp_obj.node_gridtype[:, 0]
        data = disp_obj.data  # (ntimes, nnodes, 6) -> T1,T2,T3,R1,R2,R3

        for t_idx in range(data.shape[0]):
            for i, nid in enumerate(node_ids):
                row = data[t_idx, i, :]
                disp_data.append({
                    "Subcase": subcase_id,
                    "TimeStep": t_idx,
                    "NodeID": int(nid),
                    "T1": row[0],
                    "T2": row[1],
                    "T3": row[2],
                    "R1": row[3],
                    "R2": row[4],
                    "R3": row[5],
                })

    df = pd.DataFrame(disp_data)
    csv_path = os.path.join(output_dir, "displacements.csv")
    df.to_csv(csv_path, index=False)
    log_callback(f"Displacement CSV yazıldı: {csv_path} ({len(df)} satır)")
    return csv_path


class NastranToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eray-Tool | NX Nastran Sonuç Çıkarma Aracı")
        self.root.geometry("750x550")
        self.root.resizable(True, True)

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- BDF Dosyası ---
        row = 0
        ttk.Label(main_frame, text="BDF Dosyası:").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        self.bdf_entry = ttk.Entry(main_frame, width=60)
        self.bdf_entry.grid(row=row, column=1, sticky=tk.EW, padx=4)
        ttk.Button(
            main_frame, text="Seç...",
            command=lambda: select_file(
                self.bdf_entry, "BDF Dosyası Seçiniz",
                [("BDF Dosyaları", "*.bdf *.dat *.nas"), ("Tüm Dosyalar", "*.*")]
            ),
        ).grid(row=row, column=2)

        # --- NX Nastran Exe ---
        row = 1
        ttk.Label(main_frame, text="NX Nastran:").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        self.nastran_entry = ttk.Entry(main_frame, width=60)
        self.nastran_entry.grid(row=row, column=1, sticky=tk.EW, padx=4)
        ttk.Button(
            main_frame, text="Seç...",
            command=lambda: select_file(
                self.nastran_entry, "NX Nastran Çalıştırılabilir Dosyası Seçiniz",
                [("Çalıştırılabilir", "*.exe *.bat"), ("Tüm Dosyalar", "*.*")]
            ),
        ).grid(row=row, column=2)

        # --- Çıktı Klasörü ---
        row = 2
        ttk.Label(main_frame, text="Çıktı Klasörü:").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        self.output_entry = ttk.Entry(main_frame, width=60)
        self.output_entry.grid(row=row, column=1, sticky=tk.EW, padx=4)
        ttk.Button(
            main_frame, text="Seç...",
            command=lambda: select_directory(
                self.output_entry, "Çıktı Klasörü Seçiniz"
            ),
        ).grid(row=row, column=2)

        main_frame.columnconfigure(1, weight=1)

        # --- Çalıştır Butonu ---
        row = 3
        self.run_btn = ttk.Button(
            main_frame, text="Çöz ve Sonuçları Çıkar",
            command=self._on_run,
        )
        self.run_btn.grid(row=row, column=0, columnspan=3, pady=12)

        # --- İlerleme Çubuğu ---
        row = 4
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(0, 6))

        # --- Log Alanı ---
        row = 5
        ttk.Label(main_frame, text="İşlem Günlüğü:").grid(
            row=row, column=0, sticky=tk.W
        )
        row = 6
        self.log_text = tk.Text(main_frame, height=16, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW)
        main_frame.rowconfigure(row, weight=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=row, column=3, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _log(self, message):
        """Thread-safe log mesajı ekler."""
        def _append():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(0, _append)

    def _on_run(self):
        bdf_path = self.bdf_entry.get().strip()
        nastran_exe = self.nastran_entry.get().strip()
        output_dir = self.output_entry.get().strip()

        if not bdf_path or not os.path.isfile(bdf_path):
            messagebox.showerror("Hata", "Geçerli bir BDF dosyası seçiniz.")
            return
        if not nastran_exe or not os.path.isfile(nastran_exe):
            messagebox.showerror("Hata", "Geçerli bir NX Nastran çalıştırılabilir dosyası seçiniz.")
            return
        if not output_dir:
            messagebox.showerror("Hata", "Çıktı klasörü seçiniz.")
            return
        os.makedirs(output_dir, exist_ok=True)

        self.run_btn.configure(state=tk.DISABLED)
        self.progress.start(10)

        thread = threading.Thread(target=self._worker, args=(bdf_path, nastran_exe, output_dir), daemon=True)
        thread.start()

    def _worker(self, bdf_path, nastran_exe, output_dir):
        try:
            # 1) Nastran çalıştır
            self._log("=" * 50)
            self._log("ADIM 1: NX Nastran çalıştırılıyor...")
            op2_path = run_nastran(nastran_exe, bdf_path, output_dir, self._log)

            # 2) OP2 oku
            self._log("\nADIM 2: OP2 dosyası okunuyor...")
            op2_model = OP2()
            op2_model.read_op2(op2_path)
            self._log("OP2 dosyası başarıyla okundu.")

            # 3) Von Mises stress çıkar
            self._log("\nADIM 3: Von Mises stress çıkarılıyor...")
            extract_von_mises_stress(op2_model, output_dir, self._log)

            # 4) Displacement çıkar
            self._log("\nADIM 4: Displacement sonuçları çıkarılıyor...")
            extract_displacements(op2_model, output_dir, self._log)

            self._log("\n" + "=" * 50)
            self._log("İŞLEM TAMAMLANDI!")
            self.root.after(0, lambda: messagebox.showinfo("Başarılı", "Sonuçlar CSV olarak yazıldı!"))

        except Exception as e:
            self._log(f"\nHATA: {e}")
            self.root.after(0, lambda: messagebox.showerror("Hata", str(e)))
        finally:
            self.root.after(0, self._finish)

    def _finish(self):
        self.progress.stop()
        self.run_btn.configure(state=tk.NORMAL)


def main():
    root = tk.Tk()
    NastranToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
