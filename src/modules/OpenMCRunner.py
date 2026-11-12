"""Run one OpenMC model in a unique attempt directory.

OpenMCRunner validates the model and run ID, optionally exports XML, invokes
openmc.Model.run, and verifies the resulting statepoint. Cleanup occurs only
when explicitly requested."""

import shutil
from pathlib import Path

import openmc


class OpenMCRunner:
    """Run an OpenMC model in an isolated directory for each transport attempt."""

    def __init__(
        self,
        work_root="openmc_runs",
        threads=None,
        output=True,
        export_model_xml=True,
        geometry_debug=False,
        openmc_exec=None,
    ):
        """Store the run root and explicit OpenMC execution options.

        Parameters select the artifact root, thread count, console output, model
        export, overlap checking, and executable path. The constructor returns no value."""
        if not isinstance(geometry_debug, bool):
            raise TypeError("geometry_debug harus berupa bool.")
        if not isinstance(openmc_exec, (str, Path)) or not str(openmc_exec).strip():
            raise TypeError(
                "openmc_exec harus berupa path executable absolut yang sudah di-resolve."
            )
        executable_path = Path(openmc_exec)
        if not executable_path.is_absolute():
            raise ValueError(
                "openmc_exec harus berupa path executable absolut yang sudah di-resolve."
            )
        self.work_root = Path(work_root)
        self.threads = threads
        self.output = output
        self.export_model_xml = export_model_xml
        self.geometry_debug = geometry_debug
        self.openmc_exec = executable_path

    def _validate_model(self, model):
        """Validate one model parameter and return no value."""
        if not isinstance(model, openmc.Model):
            raise TypeError("model harus berupa instance openmc.Model.")

    def _prepare_workdir(self, run_id):
        """Create and return a unique working directory for one run ID."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id harus berupa string yang tidak kosong.")
        work_dir = self.work_root / run_id
        work_dir.mkdir(
            parents=True,
            exist_ok=False,
        )
        return work_dir

    def _validate_statepoint(self, statepoint_path):
        """Validate and return the statepoint path produced by OpenMC."""
        if statepoint_path is None:
            raise RuntimeError("OpenMC selesai tetapi statepoint tidak diperoleh.")
        statepoint_path = Path(statepoint_path)
        if not statepoint_path.is_file():
            raise FileNotFoundError(f"Statepoint tidak ditemukan: {statepoint_path}")
        return statepoint_path

    def run(self, model, run_id):
        """Run one model under ``run_id`` and return its paths as a mapping."""
        self._validate_model(model)
        work_dir = self._prepare_workdir(run_id)
        try:
            statepoint_path = model.run(
                cwd=work_dir,
                threads=self.threads,
                geometry_debug=self.geometry_debug,
                output=self.output,
                export_model_xml=self.export_model_xml,
                openmc_exec=self.openmc_exec,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Simulasi OpenMC gagal untuk {run_id}.") from exc
        statepoint_path = self._validate_statepoint(statepoint_path)
        return {"run_id": run_id, "work_dir": work_dir, "statepoint_path": statepoint_path}

    def cleanup(self, work_dir):
        """Remove the requested work directory and return no value."""
        work_dir = Path(work_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)
