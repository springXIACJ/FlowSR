import tempfile
import unittest
from pathlib import Path

from PIL import Image

from flowsr.metrics import (
    collect_image_pairs,
    split_metric_names,
    summarize_metric_rows,
)


class MetricsUtilityTests(unittest.TestCase):
    def test_collect_image_pairs_matches_by_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr_dir = root / "sr"
            gt_dir = root / "gt"
            sr_dir.mkdir()
            gt_dir.mkdir()
            Image.new("RGB", (2, 2)).save(sr_dir / "b.png")
            Image.new("RGB", (2, 2)).save(sr_dir / "a.jpg")
            Image.new("RGB", (2, 2)).save(gt_dir / "a.png")
            Image.new("RGB", (2, 2)).save(gt_dir / "b.png")

            pairs = collect_image_pairs(sr_dir, gt_dir)

            self.assertEqual([pair.name for pair in pairs], ["a", "b"])
            self.assertEqual(pairs[0].sr_path, sr_dir / "a.jpg")
            self.assertEqual(pairs[0].gt_path, gt_dir / "a.png")

    def test_collect_image_pairs_reports_missing_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr_dir = root / "sr"
            gt_dir = root / "gt"
            sr_dir.mkdir()
            gt_dir.mkdir()
            Image.new("RGB", (2, 2)).save(sr_dir / "only_sr.png")
            Image.new("RGB", (2, 2)).save(gt_dir / "only_gt.png")

            with self.assertRaises(ValueError) as ctx:
                collect_image_pairs(sr_dir, gt_dir)

            message = str(ctx.exception)
            self.assertIn("missing GT", message)
            self.assertIn("missing SR", message)

    def test_split_metric_names_groups_metric_types(self):
        plan = split_metric_names(["psnr", "clipiqa", "fid", "maniqa"])

        self.assertEqual(plan.paired, ["psnr"])
        self.assertEqual(plan.no_reference, ["clipiqa", "maniqa-pipal"])
        self.assertTrue(plan.fid)

    def test_summarize_metric_rows_averages_numeric_values(self):
        summary = summarize_metric_rows([
            {"psnr": 20.0, "ssim": 0.8},
            {"psnr": 22.0, "ssim": 0.6},
        ])

        self.assertEqual(summary, {"psnr": 21.0, "ssim": 0.7})


if __name__ == "__main__":
    unittest.main()
