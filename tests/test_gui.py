"""Kiểm tra luồng dữ liệu giao diện mà không mở cửa sổ hay hộp thoại."""

import tkinter as tk
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from spatial_denoise.core import RunConfig, run_pipeline
from spatial_denoise.gui import DenoiseApp, NAMES


class GuiFlowTests(unittest.TestCase):
    def setUp(self):
        self.tcl = tk.Tcl()
        self.app = SimpleNamespace()
        for name, initial in {
            "mode": "Khử nhiễu ảnh có sẵn", "noise": "Muối tiêu", "sigma": "bad",
            "amount": "bad", "seed": "bad", "kernel": "3", "threshold": "20",
            "reference_info": "", "mode_hint": "", "metric_note": "", "snapshot": "",
            "info": "", "insight": "",
        }.items():
            setattr(self.app, name, tk.StringVar(master=self.tcl, value=initial))
        for name in ("reference_box", "open_image_button", "run_button", "filter_heading",
                     "noise_combo", "seed_entry", "sigma_entry", "amount_entry", "demo_button",
                     "clear_reference_button", "save_button", "table",
                     "sigma_label", "amount_label"):
            setattr(self.app, name, Mock())
        self.app.table.get_children.return_value = ()
        self.app.noise_fields = [Mock() for _ in range(8)]
        self.app.cards = {(name, row): Mock() for name in NAMES for row in (0, 1)}
        self.app.original = np.full((9, 11), 128, np.uint8)
        self.app.source = "noisy.png"
        self.app.reference = None
        self.app.reference_source = None
        self.app.result = None
        self.app.busy = False
        self.app._work = Mock()
        self.app.view = Mock()
        self.app._arrange_images = Mock()
        for name in ("_noise_state", "_clear_result", "_clear_reference", "_mode_changed",
                     "_read_config", "_show_result", "_load", "_run", "_save_image"):
            setattr(self.app, name, MethodType(getattr(DenoiseApp, name), self.app))

    def test_direct_run_ignores_disabled_noise_values_and_passes_input_unchanged(self):
        self.app._run()
        task = self.app._work.call_args.args[0]
        result = task()
        self.assertEqual(result.config.input_mode, "existing")
        np.testing.assert_array_equal(result.images["noisy"], self.app.original)
        self.assertFalse(result.has_reference)

    def test_mode_switch_clears_old_reference_scores_and_disables_save(self):
        self.app.reference = self.app.original.copy()
        self.app.reference_source = "old-clean.png"
        self.app.result = object()
        self.app._mode_changed()
        self.assertIsNone(self.app.reference)
        self.assertIsNone(self.app.reference_source)
        self.assertIsNone(self.app.result)
        self.app.save_button.configure.assert_called_with(state="disabled")
        for field in self.app.noise_fields:
            field.grid_remove.assert_called_once()
        self.assertIn("Không có ảnh chuẩn", self.app.metric_note.get())
        self.app.mode.set("Thử nghiệm tạo nhiễu")
        self.app._mode_changed()
        for field in self.app.noise_fields:
            field.grid.assert_called_once()
        self.app.reference_box.grid_remove.assert_called_once()

    def test_new_input_forgets_old_reference_and_result(self):
        self.app.reference = self.app.original.copy()
        self.app.reference_source = "old.png"
        self.app.result = object()
        new_image = np.full((7, 8), 99, np.uint8)
        self.app._load(new_image, "new.png")
        self.assertIsNone(self.app.reference)
        self.assertIsNone(self.app.reference_source)
        self.assertIsNone(self.app.result)
        np.testing.assert_array_equal(self.app.cards[("noisy", 0)].set_array.call_args.args[0], new_image)
        self.assertIsNone(self.app.cards[("original", 0)].set_array.call_args.args[0])

    def test_no_reference_scores_are_unavailable_and_no_best_row(self):
        result = run_pipeline(self.app.original, RunConfig(input_mode="existing"))
        self.app._show_result(result)
        self.assertEqual(self.app.table.insert.call_count, 3)
        for call in self.app.table.insert.call_args_list:
            self.assertEqual(call.kwargs["values"][1:5], ("—",) * 4)
            self.assertEqual(call.kwargs["tags"], ())
        self.assertIn("Không có ảnh chuẩn", self.app.metric_note.get())
        self.app.save_button.configure.assert_called_with(state="normal")

    def test_save_generated_noise_full_resolution(self):
        self.app.result = run_pipeline(self.app.original, RunConfig(noise="salt_pepper", amount=0.5))
        self.assertFalse(np.array_equal(self.app.original, self.app.result.images["noisy"]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            target = Path(folder) / "anh_nhieu.png"
            with patch("spatial_denoise.gui.filedialog.asksaveasfilename", return_value=str(target)):
                self.app._save_image("noisy")
            self.assertEqual(self.app._work.call_args.args[0](), target)
            with Image.open(target) as image:
                np.testing.assert_array_equal(np.asarray(image), self.app.result.images["noisy"])
        np.testing.assert_array_equal(self.app.original, np.full((9, 11), 128, np.uint8))

    def test_save_selected_filter_full_resolution_without_modifying_input(self):
        self.app.result = run_pipeline(self.app.original, RunConfig(noise="salt_pepper", amount=0.5))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            target = Path(folder) / "result.png"
            with patch("spatial_denoise.gui.filedialog.asksaveasfilename", return_value=str(target)):
                self.app._save_image("mean")
            saved = self.app._work.call_args.args[0]()
            self.assertEqual(saved, target)
            with Image.open(target) as image:
                np.testing.assert_array_equal(np.asarray(image), self.app.result.images["mean"])
            # Chọn lại tên tệp đầu vào không được phép ghi đè.
            self.app.source = str(target)
            self.app._work.reset_mock()
            with patch("spatial_denoise.gui.filedialog.asksaveasfilename", return_value=str(target)), \
                 patch("spatial_denoise.gui.messagebox.showerror") as error:
                self.app._save_image("median")
            error.assert_called_once()
            self.app._work.assert_not_called()

    def test_equal_psnr_does_not_claim_filter_improved_clean_image(self):
        result = run_pipeline(self.app.original, RunConfig(sigma=0))
        self.app._show_result(result)
        for call in self.app.table.insert.call_args_list:
            self.assertEqual(call.kwargs["tags"], ())
        self.assertIn("chưa cải thiện", self.app.insight.get())


if __name__ == "__main__":
    unittest.main()
