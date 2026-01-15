# Metashape UAV Processing Automation

Automated end-to-end photogrammetry pipeline for **Agisoft Metashape Pro 2.2.2** for UAV campaigns, supporting:

- Multiple **sites** per fieldwork campaign
- Multiple **sensors** per site:
  - DJI **L1** (RGB)
  - DJI **Mavic 3E** (RGB)
  - **MicaSense Dual** (10-band multispectral)
- **Network processing** (Match / Align / Depth / Dense / DEM / Ortho)
- **Multispectral reflectance calibration** (panel + “Calibration images” group)
- **Network exports** of:
  - DEM (GeoTIFF, BigTIFF)
  - Orthomosaic (GeoTIFF, BigTIFF, reflectance scaled)
  - PDF processing report

The main script (`processing.py`) is designed to be **idempotent**: you can safely rerun it and it will skip work that’s already done.

---

## 1. Folder Structure

The script assumes the following campaign folder structure:

```text
Fieldwork_-Date-2025/
└── UAV/
    ├── processing.py             # this script (inside UAV)
    ├── networkConfig.txt         # network processing configuration
    ├── panel_calibration.csv     # (optional) reflectance panel values
    ├── Site2/
    │   ├── DualCamera/           # MicaSense Dual data
    │   │   ├── Red/
    │   │   └── Blue/
    │   ├── L1/                   # DJI L1 RGB
    │   └── Mavic3E/              # DJI Mavic 3E RGB
    ├── Site11/
    │   ├── L1/
    │   └── Mavic3E/
    └── ...
```

After the script runs, it will create:

```text
UAV/
├── processing/
│   ├── Site2/
│   │   ├── Site2_L1_-Date-2025.psx
│   │   ├── Site2_Mavic3E_-Date-2025.psx
│   │   └── Site2_Dualcamera_-Date-2025.psx
│   └── Site11/
│       └── ...
└── Outputs/
    ├── Site2/
    │   ├── Site2_L1_-Date-2025_DEM.tif
    │   ├── Site2_L1_-Date-2025_Orthomosaic.tif
    │   ├── Site2_L1_-Date-2025_report.pdf
    │   ├── Site2_Dualcamera_-Date-2025_DEM.tif
    │   ├── Site2_Dualcamera_-Date-2025_Orthomosaic.tif
    │   └── Site2_Dualcamera_-Date-2025_report.pdf
    └── ...
```

The campaign suffix (`-Date-2025` in this example) is taken automatically from the parent folder name:
`Fieldwork_-Date-2025/UAV` → `-Date-2025`.

---

## 2. Requirements

- **Agisoft Metashape Pro 2.2.2**
  - Network server & workers configured
- **Python 3.9** (to match Metashape’s embedded Python)
- `processing.py` must live inside the **UAV** folder

The script works on both **Linux** and **Windows** (paths via `os.path`).

---

## 3. Sensor Conventions

Inside each `SiteX` folder, the script auto-detects sensor folders by name.

### DualCamera / MicaSense (multispectral)

- Folder names (case-insensitive):  
  `DualCamera`, `Dualcamera`, `Micasense`, `MicaSense`
- Expected subfolders: `Red/` and `Blue/`
- Multispectral, 10 bands (MicaSense Dual)

Behaviour:

- Collects images from `Red/` & `Blue/`
- Groups by shot ID (e.g. `IMG_0600_*`)
- Keeps only **complete shots** with 10 bands to avoid broken stacks
- Adds photos with `Metashape.MultiplaneLayout` (multi-plane / multi-camera layout)
- Sets `chunk.primary_channel = 9` (band 10, NIR) so NIR is used as primary display channel

### L1 (RGB)

- Folder name: `L1`
- All images in all subfolders are imported

### Mavic3E (RGB)

- Folder names: `Mavic3E`, `M3E`
- Only mission folders (folder name with ≥ 3 underscores) are used, e.g.:

  - `DJI_202511291317_002_Site11Flight2` ✅  
  - `DJI_202511291336_003` ❌ (skipped as “fun” photos)

---

## 4. Network Configuration

The script reads network settings from `networkConfig.txt` (either inside `UAV/` or in the parent folder).

Example:

- `host` – Metashape network server hostname
- `port` – Metashape network port
- `root` – path to the UAV root as seen by the workers (network path)

The script:

- Enables network processing in Metashape preferences
- Connects a `Metashape.NetworkClient`
- Prints server status (version, nodes)
- Uses the batch list to avoid queuing duplicate **active** batches per project

---

## 5. Panel Calibration CSV (MicaSense)

For multispectral reflectance calibration, the script optionally loads a panel CSV before running `CalibrateReflectance`.

Example `panel_calibration.csv`:

```csv
wavelength_nm,reflectance
475,0.27
560,0.23
***
**
*
```

The script looks for panel CSVs in:

- `<Fieldwork>/panel_calibration.csv`
- `<Fieldwork>/UAV/panel_calibration.csv`
- Optional site/sensor-specific names, e.g.  
  `Site2_Dualcamera_Sep2025_panel_calibration.csv` in either folder above.

For DualCamera projects:

1. It finds the **“Calibration images”** camera group in the chunk.
2. Collects the cameras that belong to this group.
3. Calls:

   ```python
   chunk.loadReflectancePanelCalibration(panel_csv, calib_cameras)
   ```

4. Saves the project so calibration is stored in the `.psx`.

---

## 6. Processing Workflow

The `main()` function runs 4 steps:

1. Create .psx **projects**.
2. Import **images** and set primary channels.
3. Submit **network batches** for processing.
4. Submit **network exports** (DEM / ortho / report).

Each step is designed to be **idempotent** (safe to rerun).

---

### Step 1 – Create Metashape Projects

**Function:** `create_projects(uav_root, campaign_suffix, site_filter=None)`

For each `SiteX` inside `UAV`:

- Detects which sensors exist under the site folder (`DualCamera`, `L1`, `Mavic3E`).
- Creates `processing/SiteX/` if needed.
- For each sensor, creates an empty Metashape project:

  ```text
  processing/Site2/Site2_L1_Sep2025.psx
  processing/Site2/Site2_Mavic3E_Sep2025.psx
  processing/Site2/Site2_Dualcamera_Sep2025.psx
  ```

- If a project already exists → **skipped**.

---

### Step 2 – Import Images

**Function:** `import_all_images(uav_root, campaign_suffix, site_filter=None)`

For each `.psx` project:

- Opens the project and gets the first chunk.
- If the chunk already has cameras → **skip import**.
- Otherwise:

  - **L1**  
    - Imports all image files under `L1/` recursively.

  - **Mavic3E**  
    - Scans subfolders under `Mavic3E/`.
    - Uses only folders whose name has **≥ 3 underscores** (mission folders).
    - Imports all images from these mission folders.

  - **DualCamera (MicaSense)**  
    - Collects images from `Red/` and `Blue/`.
    - Groups images by shot ID (e.g. `IMG_0600_*`).
    - Keeps only shots with exactly **10 bands** (complete 10-band acquisitions).
    - Adds photos with `layout=Metashape.MultiplaneLayout` (multi-plane).
    - Sets `chunk.primary_channel = 9` (NIR).

- Saves the project after import.

---

### Step 3 – Network Processing

**Function:** `run_network_batches(uav_root, campaign_suffix, site_filter=None)`

For each `.psx` project:

- Looks up **active batches** on the network server for that project.
  - If any are active → **skip** (don’t double-submit).

- If no active batch, opens the project and inspects the chunk.

#### RGB (L1, Mavic3E)

A helper `chunk_has_rgb_products(chunk)` checks if the chunk already has:

- aligned cameras, or
- point cloud, or
- DEM, or
- orthomosaic.

If yes → **skip RGB processing**.

If not, it submits `build_rgb_tasks()` as a network batch:

1. `MatchPhotos`  
   - `keypoint_limit = 60000`  
   - `tiepoint_limit = 0`  
   - `mask_tiepoints = False`  
   - generic + reference preselection enabled
2. `AlignCameras`
3. `BuildDepthMaps`  
   - `downscale = 1`  
   - `filter_mode = Mild`  
   - `reuse_depth = True`
4. `BuildPointCloud`  
   - `source_data = Depth maps`  
   - `point_colors = True`  
   - `point_confidence = True`  
   - `uniform_sampling = False`
5. `BuildDem`  
   - `source_data = Point cloud`  
   - `interpolation = Enabled`
6. `BuildOrthomosaic`  
   - `surface_data = DEM`  
   - `blending_mode = MosaicBlending`  
   - `fill_holes = True`

#### Multispectral (DualCamera)

For DualCamera projects, the script checks:

- `aligned = chunk_has_alignment(chunk)`  
- `ms_products_done = chunk_has_ms_products(chunk)`  

  (DEM and/or orthomosaic present)

- `calib_group_exists = has_calibration_images_group(chunk)`  

  (“Calibration images” camera group exists)

Logic:

1. **Not aligned** → submit `build_multi_align_tasks()`:
   - `MatchPhotos` (same params as RGB)
   - `AlignCameras`

2. **Aligned & products already exist (DEM/orthomosaic)** → **skip**.

3. **Aligned, no products yet**:

   - If **no “Calibration images” group** → skip (calibration not ready).

   - If calibration group exists:

     - Calls `load_panel_calibration(doc, chunk, uav_root, site_name, project_suffix)`:

       - Looks for panel CSV (see section 5).
       - Uses only the cameras from the “Calibration images” group.
       - Calls `chunk.loadReflectancePanelCalibration(panel_csv, calib_cameras)`.
       - Saves the project.

     - If panel CSV load fails → skip products.

     - If load succeeds → submits `build_multi_products_tasks()`:

       1. `CalibrateReflectance`  
          - `use_reflectance_panels = True`  
          - `use_sun_sensor = False`
       2. `BuildDepthMaps`
       3. `BuildPointCloud`
       4. `BuildDem`
       5. `BuildOrthomosaic`

---

### Step 4 – Network Exports

**Function:** `export_outputs_network(uav_root, campaign_suffix, site_filter=None)`

For each project where:

- there is **no active network batch**, and  
- DEM / orthomosaic have been created,

the script builds **network export tasks**:

- DEM (GeoTIFF / BigTIFF)
- Orthomosaic (GeoTIFF / BigTIFF)
- PDF report

Exports go to:

```text
UAV/Outputs/<SiteName>/
```

with filenames:

- `<Site>_<Sensor>_<Campaign>_DEM.tif`
- `<Site>_<Sensor>_<Campaign>_Orthomosaic.tif`
- `<Site>_<Sensor>_<Campaign>_report.pdf`

**DEM & Orthomosaic export settings:**

- `image_format = TIFF`
- BigTIFF enabled via `ImageCompression.tiff_big = True`
- `projection = WGS 84 / UTM zone 37N (EPSG:32637)`
- `save_alpha = False`
- `white_background = False`
- Optional: tiled TIFF and overviews for performance

**Multispectral Orthomosaic:**

Before creating the ExportRaster task, the script sets a **raster transform** on the chunk:

- `B1/32768, B2/32768, …, B10/32768`

so the exported orthomosaic is in **reflectance units (0–1)**.

Exports are run as a **separate network batch**, not locally.

---

## 7. Running the Script

### From Metashape GUI

1. Open **Metashape Pro 2.2.2**.
2. Go to **Tools → Console**.
3. Change directory to your `UAV` folder:

   ```python
   import os
   os.chdir("/path/to/Fieldwork_Sep2025/UAV")
   ```

4. Run:

   ```python
   import processing  # if the script file is processing.py
   processing.main()
   ```

### From Command Line (using Metashape)

**Linux:**

```bash
/path/to/metashape-pro/metashape.sh -r processing.py
```

**Windows (PowerShell):**

```powershell
cd C:\\Data\\Fieldwork_Sep2025\\UAV
"C:\\Program Files\\Agisoft\\Metashape Pro\\metashape.exe" -r processing.py
```

---

## 8. Debugging a Single Site

At the top of `processing.py` there is:

```python
TEST_SITE_NAME = None
```

To process only one site (e.g. `Site2`), set:

```python
TEST_SITE_NAME = "Site2"
```

Run `processing.main()` as usual; only `Site2` will be processed.  
Set back to `None` to process all sites.

---

## 9. Idempotency / Re-Runs

The script is designed to be rerun safely:

- **Step 1** – skips projects that already exist.
- **Step 2** – skips import if the chunk already has cameras.
- **Step 3** –
  - skips if the server already has **active batches** for a project,
  - skips RGB projects that already have products,
  - skips Dual projects that already have DEM/orthomosaic.
- **Step 4** – skips export tasks if the corresponding output files already exist.

Typical workflow:

1. Run `processing.py` once → projects created, images imported, network batches submitted.
2. Wait for network jobs to finish.
3. Run `processing.py` again → exports are submitted for completed projects, existing work is skipped.
4. You can repeat as often as needed as you add new sites / flights.

---


## 11. Reference

Agisoft Metashape Python API 2.2.x  
[Metashape Python API 2.2.3 (PDF)](https://www.agisoft.com/pdf/metashape_python_api_2_2_3.pdf)



