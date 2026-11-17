"""Kiểm tra vùng xem và hành vi Tkinter; không gửi input tới Windows."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spatial_denoise.core import RunConfig, run_pipeline
from spatial_denoise.gui import DenoiseApp
from spatial_denoise.imaging import demo_image
from spatial_denoise.viewer import image_transform


class TransformTests(unittest.TestCase):
    def test_fit_preserves_aspect_and_centers_portrait(self):
        scale, x, y = image_transform((200, 400), (600, 300), None, (0.9, 0.1))
        self.assertEqual((scale, x, y), (0.75, 225.0, 0.0))

    def test_zoom_and_pan_do_not_expose_blank_area_at_edges(self):
        self.assertEqual(image_transform((1000, 800), (400, 300), 2, (0, 0)), (2, 0, 0))
        self.assertEqual(image_transform((1000, 800), (400, 300), 2, (1, 1)), (2, -1600, -1300))
        self.assertEqual(image_transform((1000, 800), (400, 300), 2, (0.5, 0.5)), (2, -800, -650))


class ViewerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(DenoiseApp, "_run"):
            cls.app = DenoiseApp()
            cls.app.geometry("1280x800")
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        for timer in cls.app.tk.call("after", "info"):
            cls.app.after_cancel(timer)
        cls.app.destroy()

    def setUp(self):
        self.app._exit_fullscreen()
        self.app.original = demo_image()
        self.app.source = "Ảnh tổng hợp tự tạo"
        self.app.tabs.select(0)
        self.app.mode.set("Thử nghiệm tạo nhiễu")
        self.app._mode_changed()
        self.app.kernel.set("3")
        self.app._arrange_images()
        self.app._show_result(run_pipeline(self.app.original, self.app._read_config()))
        self.app.update()

    def test_large_preview_and_actions_fit_small_window_and_both_filters(self):
        app = self.app
        result = app.result
        for size in ("1000x640", "1280x800"):
            app.geometry(size)
            app._arrange_images()
            app.update()
            before = app.cards[("noisy", 0)]
            after = app.cards[("median", 0)]
            self.assertTrue(before.winfo_ismapped() and after.winfo_ismapped())
            self.assertTrue(app.cards[("mean", 0)].winfo_ismapped())
            self.assertTrue(app.cards[("original", 0)].winfo_ismapped())
            self.assertGreater(after.canvas.winfo_width(), 280)
            self.assertGreater(after.canvas.winfo_height(), 80)
            for button in (app.run_button, app.save_button):
                self.assertLessEqual(button.winfo_rooty() + button.winfo_height(),
                                     app.winfo_rooty() + app.winfo_height())
            self.assertIs(app.result, result)  # Đổi kích thước không xử lý lại ảnh.

    def test_zoom_anchor_pan_and_sobel_share_one_view(self):
        app = self.app
        app.geometry("1280x800")
        app.update()
        before, after = app.cards[("noisy", 0)], app.cards[("median", 0)]
        app.view.actual_size()
        point = (before.canvas.winfo_width() / 2 + 20, before.canvas.winfo_height() / 2 + 10)
        scale, ox, oy = before.transform()
        pixel = ((point[0] - ox) / scale, (point[1] - oy) / scale)
        app.view.zoom(2, before, point)
        scale, ox, oy = before.transform()
        self.assertAlmostEqual((point[0] - ox) / scale, pixel[0])
        self.assertAlmostEqual((point[1] - oy) / scale, pixel[1])
        before._pan_start(SimpleNamespace(x=100, y=100))
        before._pan(SimpleNamespace(x=70, y=80))
        self.assertEqual(before.transform()[0], after.transform()[0])
        # Grid rows can differ by one pixel; both views retain the same image center.
        for card in (before, after):
            scale, x, y = card.transform()
            width, height = card.viewport_size()
            iw, ih = card.image_size()
            self.assertAlmostEqual((width / 2 - x) / (iw * scale), app.view.center[0])
            self.assertAlmostEqual((height / 2 - y) / (ih * scale), app.view.center[1])
        self.assertGreater(app.view.center[0], 0.5)
        app.tabs.select(1)
        app.update()
        app.view.redraw()
        for name in ("original", "noisy", "mean", "median"):
            card = app.cards[(name, 1)]
            self.assertTrue(card.winfo_viewable())
            self.assertEqual(card.transform()[0], 2)
            self.assertIsNotNone(card.photo)
        app.view.fit()
        self.assertIsNone(app.view.scale)
        self.assertEqual(app.view.center, (0.5, 0.5))

    def test_direct_mode_hides_unavailable_reference_and_clears_metrics(self):
        app = self.app
        app.mode.set("Khử nhiễu ảnh có sẵn")
        app._mode_changed()
        self.assertEqual(app.table.get_children(), ())
        self.assertEqual(str(app.save_button.cget("state")), "disabled")
        self.assertIsNone(app.cards[("median", 0)].array)
        app._show_result(run_pipeline(app.original, RunConfig(input_mode="existing")))
        self.assertIn("Không có ảnh chuẩn", app.cards[("median", 0)].detail.cget("text"))

    def test_fullscreen_hides_controls_and_escape_restores_them(self):
        app = self.app
        app._toggle_fullscreen()
        app.update()
        self.assertFalse(app.sidebar_host.winfo_ismapped())
        self.assertFalse(hasattr(app, "header"))
        self.assertTrue(app.tabs.winfo_ismapped())
        app._exit_fullscreen()
        app.update()
        self.assertTrue(app.sidebar_host.winfo_ismapped())
        self.assertFalse(hasattr(app, "header"))

    def test_parameter_changes_mark_existing_result_until_run_or_reverted(self):
        app = self.app
        result = app.result
        app.kernel.set("7")
        self.assertIn("CHƯA ÁP DỤNG", app.snapshot.get())
        self.assertIs(app.result, result)
        app.kernel.set("3")
        self.assertNotIn("CHƯA ÁP DỤNG", app.snapshot.get())

    def test_new_clean_image_is_visible_before_processing(self):
        app = self.app
        app._load(np.full((80, 120), 100, np.uint8), "new-clean.png")
        app.update()
        self.assertTrue(app.cards[("original", 0)].winfo_ismapped())
        self.assertIsNotNone(app.cards[("original", 0)].array)
        app._show_result(run_pipeline(app.original, app._read_config()))

    def test_empty_and_large_portrait_render_only_visible_pixels(self):
        card = self.app.cards[("median", 0)]
        card.set_array(None, "Chưa xử lý")
        self.app.view.zoom(1.25, card)
        self.assertIsNone(card.photo)
        image = np.arange(2_000_000, dtype=np.uint8).reshape(2000, 1000)
        card.set_array(image)
        self.app.view.scale = 8
        card.draw()
        self.assertLessEqual(card.photo.width(), card.canvas.winfo_width() + 1)
        self.assertLessEqual(card.photo.height(), card.canvas.winfo_height() + 1)
