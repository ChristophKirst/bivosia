# Set up BiVOSIA on a remote workstation or server

1. Install BiVOSIA and Python on the server.
2. Register the installation in the local app.
3. Install SAM2/Cutie on the selected host, then run predictions or VOS.

The examples use a Linux server named `workstation`. Substitute your SSH alias
or `user@host` and your own software/environment paths.

## 1. Check SSH access

From your local machine:

```bash
ssh workstation
```

- Verify the host key against your server's fingerprint.
- Configure key or SSH-agent authentication; put custom ports in your SSH configuration.
- BiVOSIA uses batch mode with host-key checking. It cannot prompt for a password during a job.

Exit the remote shell and check noninteractive access:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes workstation "python3 --version"
```

- BiVOSIA needs **Python 3.10+**; a manual SAM3 installation needs **Python 3.12+**.
- GPU inference also needs a working driver and compatible PyTorch in the selected environment.

## 2. Copy the BiVOSIA code

From the local directory containing the outer `bivosia/` project folder:

```bash
ssh workstation "mkdir -p ~/software"
rsync -a --exclude external/ --exclude .git/ bivosia/ workstation:~/software/bivosia/
```

- The outer `~/software/bivosia/` folder contains `pyproject.toml`, `docs/` and the inner
  `bivosia/` package. That inner folder contains `config.py` and `processing/`.
- The copy excludes local model downloads and environments in `external/`; install
  the required models in the remote environment as described below.
- Keep local and remote code in sync. Registering an installation does not copy or update it.

## 3. Prepare the remote Python environment

Create an environment, or use an existing one:

```bash
ssh workstation
python3 -m venv ~/venvs/bivosia
source ~/venvs/bivosia/bin/activate
cd ~/software/bivosia
python -m pip install -e .
```

- The editable installation uses the copied project. The remote account needs write
  access to `~/software/bivosia/external/` for model sources and weights.
- Cutie jobs also need write access to the remote dataset directory.

Check the installed package from any directory:

```bash
python -c "from bivosia.processing.remote.server import installation_info; print(installation_info())"
```

- Copy the reported **python** and **code** paths for registration.
- This checks core imports without a dataset or model weights.
- No remote desktop window is needed. On Linux with systemd, background jobs run as
  temporary user services so they survive the submitting SSH session.
- Check the user service manager with `systemctl --user status`. To keep it running
  after logout, enable lingering with `loginctl enable-linger "$USER"` if permitted,
  or ask the server administrator. Other hosts use detached worker processes.

## 4. Register the installation locally

Open **Settings → Remote installations… → New**:

| Field | Example |
|---|---|
| Host | `workstation` or `user@host` |
| Python | `/home/you/venvs/bivosia/bin/python` |
| Code | `/home/you/software/bivosia/bivosia` |
| Default data folder | `/home/you/data` (optional) |

- Use the server's actual absolute paths.
- **Python** is the executable, not a shell activation command.
- **Code** is the inner `bivosia/` package folder, not the project root or a dataset.
  Copy the `code` value returned by `installation_info()` to avoid guessing the path.
- **Save** checks SSH, code, protocol and imports. Model weights and a dataset are unnecessary.
- The saved installation appears in both the key-frame and VOS windows.
- The default data folder supplies `<folder>/<dataset-name>` for new datasets.
  Use **View files → Locations…** to choose a different destination for a dataset.

## 5. Install the models you need

Choose **Run on → SSH**, select the installation, then use **Install** beside
**Model** for SAM2 or Cutie.

Alternatively, run these commands from any directory in the activated remote environment:

```bash
python -m bivosia.run.run_install sam2
python -m bivosia.run.run_install cutie
```

- Each command installs only its named model into that Python environment.
- Sources and weights go under the remote `external/` directory.
- Existing core package versions are preserved; resolve reported dependency conflicts manually.
- **SAM3 requires a manual package and authorized checkpoint. No automatic SAM3 downloads.**

## 6. Run on the remote machine

| Task | Remote inputs | Local result |
|---|---|---|
| SAM key-frame prediction | Model/checkpoint; the app transfers the current image and prompts. No remote dataset path. | Candidate masks return to the editor. |
| Cutie VOS | Choose the location in **View files → Locations…**. Copy or consolidate the video and key-frame files there before running. | Progress and logs; masks are read frame by frame from the remote location. |
| Mask analysis / features / keypoint alignment | Select **Processing → SSH**. Required files must exist on that host. | Remote console logs; recording viewers load plot arrays and requested mask frames. |

- Transfer dataset files in **View files**, using file **Copy / Move** or **Consolidate…**.
- Consolidation compares existing copies and asks which version to use for each conflict.
  Its destination description uses relative paths; source copies are retained.
- **Check** validates VOS inputs and model availability. **Run** opens the dataset viewer to sync missing inputs, then submits the job automatically.
- Video symlinks are copied as actual files. Use absolute server folder paths, not `sftp://` URLs.
- On a cluster, use an allocated compute node or an exported script with your scheduler.
  BiVOSIA does not request scheduler resources.

## If setup fails

| Symptom | Check |
|---|---|
| Permission denied / host-key failure | Same SSH account, key/agent and known-host entry; retry the batch-mode command in step 1. |
| BiVOSIA was not found | **Code** must contain `config.py` and `processing/`. |
| Missing module or shared library | Install requirements in the exact **Python** environment; resolve reported system dependencies. |
| Incompatible protocol / image-session response | Match the remote code to the local copy. |
| Missing model or checkpoint | Install on the selected SSH target or supply its checkpoint path. Install SAM3 manually. |
| Dataset or external file missing | Check paths with the same SSH account; local paths are not automatically remote paths. |
| Job becomes unresponsive immediately | Update BiVOSIA and check `systemctl --user status` and `loginctl show-user "$USER" -p Linger`. Servers with `KillUserProcesses=yes` terminate workers left inside an SSH session. |

## Updating an existing installation

- Stop or finish active BiVOSIA jobs before replacing code. Preserve a copy of the old installation.
- Copy the whole project, including `pyproject.toml`, `docs/`, `requirements/` and the inner package.
- In the server's selected environment, run `python -m pip install -e /path/to/project`.
- Run `installation_info()` again and update the saved **Code** path to the reported inner package.
- Keep the installation entry and its default data folder; datasets, model sources and checkpoints
  do not need to move. Restart the local app after updating its installation too.

[Return to the full documentation](DOCUMENTATION.md#remote-installations).
