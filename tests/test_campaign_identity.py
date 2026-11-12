"""Test campaign identity without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

from pathlib import Path
import re
import sys
import tempfile
import unittest


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RUNTIME_ROOT / "src"
for source_path in (RUNTIME_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from modules.campaign_identity import validate_campaign_id
from modules.config_loader import ConfigLoader
from run_optimization import _parse_arguments


class CampaignIdentityTests(unittest.TestCase):
    """Check direct, CLI, and configuration campaign-ID validation."""

    def test_accepts_portable_unique_identifier(self) -> None:
        """Return an unchanged identifier that is portable across supported filesystems."""
        campaign_id = "TRIGA2000-NSGAII_20260908.A"
        self.assertEqual(validate_campaign_id(campaign_id), campaign_id)
        self.assertEqual(_parse_arguments(["--campaign-id", campaign_id]).campaign_id, campaign_id)

    def test_rejects_placeholders_path_characters_and_reserved_names(self) -> None:
        """Reject placeholder variants, unsafe separators, whitespace, and Windows devices."""
        invalid_values = (
            "",
            " G000",
            "G000 ",
            "GANTI_DENGAN_CAMPAIGN_ID_UNIK",
            "GANTI|_DENGAN_CAMPAIGN_ID_UNIK",
            "A/B",
            "A\\B",
            "A|B",
            ".hidden",
            "trailing-",
            "CON",
            "NUL.txt",
        )
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value), self.assertRaises(ValueError):
                validate_campaign_id(invalid_value)

    def test_cli_rejects_invalid_identifier_before_optimizer_construction(self) -> None:
        """Make argparse stop on an unsafe campaign ID without constructing a campaign."""
        with self.assertRaises(SystemExit):
            _parse_arguments(["--campaign-id", "GANTI|_DENGAN_CAMPAIGN_ID_UNIK"])

    def test_config_loader_applies_the_same_campaign_validator(self) -> None:
        """Reject an unsafe configured identifier through the shared validator."""
        config_text = (RUNTIME_ROOT / "config" / "config.ini").read_text(encoding="utf-8")
        invalid_config, replacement_count = re.subn(
            r"(?m)^campaign_id\s*=.*$",
            "campaign_id = BAD|ID",
            config_text,
            count=1,
        )
        self.assertEqual(replacement_count, 1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.ini"
            config_path.write_text(invalid_config, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "campaign_id"):
                ConfigLoader(config_path)


if __name__ == "__main__":
    unittest.main()
