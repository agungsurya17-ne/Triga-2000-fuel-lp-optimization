"""Read inventory U-235 masses for initialization heuristics.

FuelInventoryLoader.load_reactivity_proxy maps fuel IDs to u_235_current in
grams. These masses only order initialization heuristics; they are neither keff
values nor replacements for OpenMC fitness evaluations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
from pathlib import Path


class FuelInventoryLoader:
    """Convert a validated movable-fuel inventory into a reactivity mapping."""

    @staticmethod
    def load_reactivity_proxy(
        fuel_inventory_path: str | Path,
        fuel_ids: Iterable[str],
    ) -> dict[str, float]:
        """Load current U-235 mass for exactly the configured movable fuels.

        Parameters are the JSON inventory path and movable fuel IDs. The return
        value maps each configured fuel ID to its positive finite U-235 mass in
        grams."""
        inventory_path = Path(fuel_inventory_path)
        try:
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Fuel inventory tidak ditemukan: {inventory_path}.") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"Fuel inventory bukan JSON valid: {exc}.") from exc

        if not isinstance(payload, Mapping):
            raise ValueError("Fuel inventory harus berupa object JSON.")
        if payload.get("units") != "g":
            raise ValueError("Satuan fuel inventory harus 'g'.")
        records = payload.get("fuels")
        if not isinstance(records, Mapping):
            raise ValueError("Fuel inventory harus memiliki mapping 'fuels'.")

        configured_ids = tuple(fuel_ids)
        serial_to_id = {fuel_id.removeprefix("fuel_"): fuel_id for fuel_id in configured_ids}
        if len(serial_to_id) != len(configured_ids):
            raise ValueError("fuel_ids tidak boleh duplikat setelah normalisasi.")

        missing = set(serial_to_id).difference(records)
        extra = set(records).difference(serial_to_id)
        if missing or extra:
            raise ValueError(
                "Fuel inventory dan fuel_ids tidak cocok; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )

        reactivity_proxy: dict[str, float] = {}
        for serial, fuel_id in serial_to_id.items():
            record = records[serial]
            if not isinstance(record, Mapping):
                raise ValueError(f"Record fuel {serial} harus berupa object JSON.")
            if record.get("fuel_type") != "fuel":
                raise ValueError(f"fuel_type untuk {serial} harus 'fuel'.")

            current_mass = record.get("u_235_current")
            if isinstance(current_mass, bool) or not isinstance(current_mass, (int, float)):
                raise ValueError(f"u_235_current untuk {serial} harus numerik.")
            mass = float(current_mass)
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError(f"u_235_current untuk {serial} harus finite dan positif.")
            reactivity_proxy[fuel_id] = mass

        return reactivity_proxy
