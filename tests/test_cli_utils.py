import tempfile
import unittest
from pathlib import Path

from PIL import Image

from flowsr.infer import (
    CheckpointValidationError,
    build_parser,
    collect_image_paths,
    make_output_path,
    validate_checkpoint,
)


class CliUtilityTests(unittest.TestCase):
    def test_collect_image_paths_accepts_file_or_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "b.JPG"
            image_b = root / "a.png"
            ignored = root / "notes.txt"

            Image.new("RGB", (2, 2)).save(image_a)
            Image.new("RGB", (2, 2)).save(image_b)
            ignored.write_text("not an image", encoding="utf-8")

            self.assertEqual(collect_image_paths(image_a), [image_a])
            self.assertEqual(collect_image_paths(root), [image_b, image_a])

    def test_make_output_path_uses_png_extension(self):
        out_dir = Path("outputs")
        image_path = Path("inputs") / "sample.jpeg"

        self.assertEqual(make_output_path(image_path, out_dir), out_dir / "sample.png")

    def test_default_checkpoint_name_uses_safetensors(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.checkpoint, Path("checkpoints/flowsr.safetensors"))

    def test_validate_checkpoint_reports_unloadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "bad.safetensors"
            checkpoint.write_bytes(b"not a safetensors checkpoint")

            with self.assertRaises(CheckpointValidationError) as ctx:
                validate_checkpoint(checkpoint)

            self.assertIn("Could not load checkpoint", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
