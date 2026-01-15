import os
import sys

import Metashape


# ----------------------------
# Configuration / Globals
# ----------------------------

# Valid image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

# If you want to debug on a single site inside UAV/, set this to the folder name
# e.g. TEST_SITE_NAME = "Site2"
# If None, all site folders in UAV/ are processed.
TEST_SITE_NAME = None

# DualCamera: expected number of bands per shot (e.g. MicaSense Dual = 10 bands)
DUALCAM_EXPECTED_PLANES = 10

# Sensor configuration
SENSOR_CONFIG = [
    {
        "key": "dual",
        "folder_names": ["DualCamera", "Dualcamera", "Micasense", "MicaSense"],
        "project_suffix": "Dualcamera",  # keep your naming style
    },
    {
        "key": "l1",
        "folder_names": ["L1"],
        "project_suffix": "L1",
    },
    {
        "key": "mavic3e",
        "folder_names": ["Mavic3E", "M3E"],
        "project_suffix": "Mavic3E",
    },
]


# ----------------------------
# Basic helpers
# ----------------------------

def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def get_campaign_suffix(uav_root: str) -> str:
    """
    Extract suffix like 'May2025' from parent folder name:
    Fieldwork_May2025/UAV -> 'May2025'
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(uav_root)))
    if "_" in parent:
        return parent.split("_")[-1]
    return parent


def detect_sensor_folders(site_path: str):
    """
    Look inside a site folder (e.g. UAV/Site2) and find sensor folders.

    Returns dict:
      {
        "dual":  {"folder_path": ..., "project_suffix": ...},
        "l1":    {...},
        "mavic3e": {...}
      }
    """
    sensors = {}
    subdirs = {
        name: os.path.join(site_path, name)
        for name in os.listdir(site_path)
        if os.path.isdir(os.path.join(site_path, name))
    }

    for sensor in SENSOR_CONFIG:
        key = sensor["key"]
        project_suffix = sensor["project_suffix"]
        folder_path = None

        for candidate in sensor["folder_names"]:
            for actual_name, full_path in subdirs.items():
                if actual_name.lower() == candidate.lower():
                    folder_path = full_path
                    break
            if folder_path:
                break

        if folder_path:
            sensors[key] = {
                "folder_path": folder_path,
                "project_suffix": project_suffix,
            }

    return sensors

def load_panel_calibration(doc, chunk, uav_root: str, site_name: str, project_suffix: str = None) -> bool:
    """
    Load reflectance panel calibration from a CSV file into the given chunk
    and save the project so the calibration is stored in the .psx file.

    Uses ONLY the cameras in the 'Calibration images' group (if present).

    Returns True if calibration was loaded, False otherwise.
    """
    # --- find calibration group & cameras ---
    calib_group = None
    for cam in chunk.cameras:
        grp = getattr(cam, "group", None)
        if not grp:
            continue
        label = (getattr(grp, "label", "") or "").strip()
        norm = label.replace(" ", "").lower()
        if norm == "calibrationimages":
            calib_group = grp
            break

    calib_cameras = []
    if calib_group is not None:
        for cam in chunk.cameras:
            if getattr(cam, "group", None) == calib_group:
                calib_cameras.append(cam)

    if not calib_cameras:
        print("    [WARN] No cameras found in 'Calibration images' group; "
              "will fall back to automatic panel detection (all cameras).")
        calib_cameras_arg = None  # let Metashape search all cameras
    else:
        print(f"    [INFO] Using {len(calib_cameras)} cameras from 'Calibration images' group for panel calibration.")
        calib_cameras_arg = calib_cameras

    # --- locate CSV file(s) ---
    campaign_root = os.path.dirname(os.path.abspath(uav_root))

    candidates = []

    # Example generic locations – adjust to your actual structure if needed
    candidates.append(os.path.join(campaign_root, "panel_calibration.csv"))
    candidates.append(os.path.join(uav_root, "panel_calibration.csv"))

    if project_suffix:
        candidates.append(os.path.join(campaign_root, f"{site_name}_{project_suffix}_panel_calibration.csv"))
        candidates.append(os.path.join(uav_root, f"{site_name}_{project_suffix}_panel_calibration.csv"))

    # Remove duplicates
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    # --- try each candidate path ---
    for path in unique_candidates:
        if not path or not os.path.isfile(path):
            continue

        try:
            # key line: pass calib_cameras_arg (list of panel photos) to the loader
            chunk.loadReflectancePanelCalibration(path, calib_cameras_arg)
            doc.save()
            print(f"    [OK] Loaded panel calibration from: {path}")
            return True
        except Exception as e:
            print(f"    [ERROR] Failed to load panel calibration from {path}: {e}")
            return False

    print("    [WARN] No panel calibration CSV found. Tried:")
    for p in unique_candidates:
        print(f"        - {p}")
    return False


# ----------------------------
# Network processing helpers
# ----------------------------

def _find_network_config_file(uav_root: str) -> str:
    """
    Look for networkConfig.txt either in UAV/ or one level above (Fieldwork_*).
    Returns full path or empty string if not found.
    """
    candidates = [
        os.path.join(uav_root, "networkConfig.txt"),
        os.path.join(os.path.dirname(os.path.abspath(uav_root)), "networkConfig.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def read_network_config(uav_root: str):
    """
    Read networkConfig.txt with keys:
        host: halo
        port: 9001
        root: /datawaha/...

    Returns dict { "host": str, "port": int, "root": str } or None if missing/broken.
    """
    cfg_path = _find_network_config_file(uav_root)
    if not cfg_path:
        print("[INFO] No networkConfig.txt found (checked UAV/ and parent). Network processing will be disabled.")
        return None

    print(f"[INFO] Using network config file: {cfg_path}")
    data = {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip().lower()] = value.strip()

    host = data.get("host")
    port_str = data.get("port")
    root = data.get("root")

    if not host or not port_str or not root:
        print("[WARN] networkConfig.txt is missing required keys (host/port/root). Network processing will be disabled.")
        return None

    try:
        port = int(port_str)
    except ValueError:
        print(f"[WARN] Invalid port value in networkConfig.txt: {port_str!r}. Network processing will be disabled.")
        return None

    return {"host": host, "port": port, "root": root}


def configure_network_processing(uav_root: str):
    """
    Enable network processing in Metashape from networkConfig.txt.
    Returns the config dict on success, or None on failure.
    """
    cfg = read_network_config(uav_root)
    if not cfg:
        return None

    settings = Metashape.app.settings
    settings.load()

    settings.network_enable = True
    settings.network_host = cfg["host"]
    settings.network_port = cfg["port"]
    settings.network_path = cfg["root"]

    settings.save()

    print("  Network processing settings updated:")
    print(f"    host: {cfg['host']}")
    print(f"    port: {cfg['port']}")
    print(f"    root: {cfg['root']}")
    print("")

    return cfg


def get_connected_client(uav_root: str):
    """
    Configure network processing and connect a NetworkClient.
    Prints basic server status. Returns (client, cfg) or (None, None) on failure.
    """
    cfg = configure_network_processing(uav_root)
    if not cfg:
        return None, None

    host = cfg["host"]
    port = cfg["port"]

    client = Metashape.NetworkClient()
    try:
        client.connect(host, port)
    except RuntimeError as e:
        print(f"  [ERROR] Could not connect to network server {host}:{port}: {e}")
        return None, None

    # Server version
    version = "unknown"
    try:
        version = client.serverVersion()
    except AttributeError:
        try:
            info = client.serverInfo()
            if isinstance(info, dict):
                version = info.get("version", str(info))
            else:
                version = str(info)
        except Exception as e:
            print(f"  [WARN] Connected, but failed to read server version: {e}")

    print(f"  [OK] Connected to server {host}:{port}. Version: {version}")

    # Node status
    try:
        nodes = client.nodeStatus()
        try:
            node_count = len(nodes)
        except TypeError:
            node_count = "unknown"
        print(f"  [INFO] Server nodeStatus(): {node_count} nodes reported.")
    except AttributeError:
        print("  [INFO] nodeStatus() not available in this Metashape version.")
    except Exception as e:
        print(f"  [WARN] Failed to obtain node status: {e}")

    return client, cfg


def get_server_batches(client):
    """
    Try to query batch list from the server.
    Returns a list-like object or [] on failure.
    """
    try:
        return client.batchList()
    except AttributeError:
        try:
            return client.getBatchList(0)
        except AttributeError:
            print("    [WARN] NetworkClient has no batchList/getBatchList; cannot inspect existing batches.")
            return []
        except Exception as e:
            print(f"    [WARN] Error calling getBatchList: {e}")
            return []
    except Exception as e:
        print(f"    [WARN] Error calling batchList: {e}")
        return []


def project_has_active_batch(client, project_path: str) -> bool:
    """
    Check if the network server already has an active batch for this project.
    We compare by project path (normalized). Any batch whose status is NOT
    finished/cancelled/failed is considered active.
    """
    batches = get_server_batches(client)
    if not batches:
        return False

    norm_proj = os.path.normpath(project_path)

    for b in batches:
        bpath = getattr(b, "path", "") or getattr(b, "project", "") or ""
        if not bpath:
            continue
        if os.path.normpath(bpath) != norm_proj:
            continue

        status = getattr(b, "status", getattr(b, "state", "")) or ""
        status_str = str(status).lower()

        if any(word in status_str for word in ("finished", "done", "completed", "cancelled", "canceled", "failed")):
            continue  # not active

        bid = getattr(b, "id", "?")
        print(f"    [INFO] Existing active batch {bid} for project {project_path} with status '{status_str}'.")
        return True

    return False

# ----------------------------
# Build processing pipelines (based on your JSON)
# ----------------------------

def build_rgb_tasks():
    """
    RGB pipeline, matching the JSON:

      MatchPhotos (keypoint_limit=60000, mask_tiepoints=false, tiepoint_limit=0)
      AlignCameras
      BuildDepthMaps (downscale=1, filter_mode=Mild, reuse_depth=on)
      BuildPointCloud (source_data=Depth maps, point_colors=on,
                       point_confidence=on, uniform_sampling=off)
      BuildDem (source_data=Point cloud, interpolation=Enabled,
                classes=All, resolution=0)
      BuildOrthomosaic (surface_data=DEM, blending_mode=MosaicBlending,
                        fill_holes=on, ghosting_filter=off, cull_faces=off,
                        refine_seamlines=off, resolution=0,
                        refine_roof_edges=off)
    """

    tasks = []

    # --- MatchPhotos ---
        # AlignPhotos = MatchPhotos + AlignCameras
    t = Metashape.Tasks.MatchPhotos()
    t.keypoint_limit = 60000
    t.tiepoint_limit = 0
    t.mask_tiepoints = False
    tasks.append(t)

    t = Metashape.Tasks.AlignCameras()
    tasks.append(t)


    # --- BuildDepthMaps ---
    t = Metashape.Tasks.BuildDepthMaps()
    t.downscale = 1
    t.reuse_depth = True
    # params_text: "filter_mode = Mild"
    t.filter_mode = Metashape.MildFiltering
    tasks.append(t)

    # --- BuildPointCloud ---
    t = Metashape.Tasks.BuildPointCloud()
    # "source_data = Depth maps, point_colors = on, point_confidence = on, uniform_sampling = off"
    t.source_data = Metashape.DepthMapsData
    t.point_colors = True
    t.point_confidence = True
    t.uniform_sampling = False
    tasks.append(t)

    # --- BuildDem ---
    t = Metashape.Tasks.BuildDem()
    # "source_data = Point cloud, interpolation = Enabled, classes = All, resolution = 0"
    t.source_data = Metashape.PointCloudData
    t.interpolation = Metashape.EnabledInterpolation
    # classes = All and resolution = 0 are defaults, so we leave them.
    tasks.append(t)

    # --- BuildOrthomosaic ---
    t = Metashape.Tasks.BuildOrthomosaic()
    # "surface_data = DEM"
    t.surface_data = Metashape.ElevationData
    t.blending_mode = Metashape.MosaicBlending
    t.fill_holes = True
    t.ghosting_filter = False
    t.cull_faces = False
    t.refine_seamlines = False
    # resolution = 0 (auto) is default.
    tasks.append(t)

    return tasks


def build_multi_align_tasks():
    """
    MultiAlignBatch.xml:

    <?xml version="1.0" encoding="UTF-8"?>
    <batchjobs version="2.2.2" save_project="true">
      <job name="AlignPhotos" target="all">
        <keypoint_limit>60000</keypoint_limit>
        <mask_tiepoints>false</mask_tiepoints>
        <tiepoint_limit>0</tiepoint_limit>
      </job>
    </batchjobs>
    """
    tasks = []

    # AlignPhotos = MatchPhotos + AlignCameras
    t = Metashape.Tasks.MatchPhotos()
    t.keypoint_limit = 60000
    t.tiepoint_limit = 0
    t.mask_tiepoints = False
    tasks.append(t)

    t = Metashape.Tasks.AlignCameras()
    tasks.append(t)

    return tasks


def build_multi_products_tasks():
    """
    Multispectral products pipeline (equivalent to MultiProductsBatch):
      BuildDepthMaps
      BuildPointCloud
      BuildDem
      BuildOrthomosaic
    """
    tasks = []

    # --- Calibrate Reflectance ---
    # Use reflectance panels, do NOT use sun sensor
    t = Metashape.Tasks.CalibrateReflectance()
    t.use_reflectance_panels = True
    t.use_sun_sensor = True
    # (optional: t.white_balance = True/False, t.source_data, t.cameras if you want to restrict)
    tasks.append(t)


    # --- BuildDepthMaps ---
    t = Metashape.Tasks.BuildDepthMaps()
    t.downscale = 1
    t.reuse_depth = True
    # params_text: "filter_mode = Mild"
    t.filter_mode = Metashape.MildFiltering
    tasks.append(t)

    # --- BuildPointCloud ---
    t = Metashape.Tasks.BuildPointCloud()
    # "source_data = Depth maps, point_colors = on, point_confidence = on, uniform_sampling = off"
    t.source_data = Metashape.DepthMapsData
    t.point_colors = True
    t.point_confidence = True
    t.uniform_sampling = False
    tasks.append(t)

    # --- BuildDem ---
    t = Metashape.Tasks.BuildDem()
    # "source_data = Point cloud, interpolation = Enabled, classes = All, resolution = 0"
    t.source_data = Metashape.PointCloudData
    t.interpolation = Metashape.EnabledInterpolation
    # classes = All and resolution = 0 are defaults, so we leave them.
    tasks.append(t)

    # --- BuildOrthomosaic ---
    t = Metashape.Tasks.BuildOrthomosaic()
    # "surface_data = DEM"
    t.surface_data = Metashape.ElevationData
    t.blending_mode = Metashape.MosaicBlending
    t.fill_holes = True
    t.ghosting_filter = False
    t.cull_faces = False
    t.refine_seamlines = False
    # resolution = 0 (auto) is default.
    tasks.append(t)

    return tasks


def submit_network_tasks(client, project_path: str, tasks, description: str = "") -> bool:
    """
    Convert Metashape.Tasks.* objects to NetworkTasks, create a batch and start it.
    """
    if not tasks:
        print("    [WARN] No tasks specified, skipping.")
        return False

    # Open project to prepare objects
    doc = Metashape.Document()
    doc.open(project_path)
    if doc.chunks:
        objects = list(doc.chunks)
    else:
        objects = [doc]

    network_tasks = []
    for task in tasks:
        try:
            nt = task.toNetworkTask(objects)
        except TypeError:
            try:
                nt = task.toNetworkTask(doc)
            except Exception as e:
                print(f"    [ERROR] Cannot convert task {getattr(task, 'name', str(task))} to NetworkTask: {e}")
                continue
        except Exception as e:
            print(f"    [ERROR] Cannot convert task {getattr(task, 'name', str(task))} to NetworkTask: {e}")
            continue

        network_tasks.append(nt)

    if not network_tasks:
        print("    [WARN] No network tasks created, skipping.")
        return False

    # Create batch on server
    try:
        batch_id = client.createBatch(project_path, network_tasks, priority=0,
                                      meta={"description": description})
    except TypeError:
        # Older signature
        try:
            batch_id = client.createBatch(project_path, network_tasks)
        except Exception as e:
            print(f"    [ERROR] Failed to create network batch: {e}")
            return False
    except Exception as e:
        print(f"    [ERROR] Failed to create network batch: {e}")
        return False

    # Start batch
    try:
        client.resumeBatch(batch_id)
    except Exception as e:
        print(f"    [WARN] Created batch {batch_id} but failed to resume: {e}")
        return False

    print(f"    [OK] Submitted network batch {batch_id} for: {project_path}")
    return True


def apply_micasense_raster_transform(chunk: Metashape.Chunk, default_bands: int = 10) -> bool:
    """
    Configure chunk.raster_transform so that each band is divided by 32768:
        B1/32768, B2/32768, ... Bn/32768

    Used for MicaSense multispectral orthomosaic export (index value transform).

    Returns True if transform was set successfully.
    """
    try:
        # Try to infer number of bands from sensors
        band_count = 0
        for sensor in chunk.sensors:
            bands = getattr(sensor, "bands", None)
            if bands:
                band_count = max(band_count, len(bands))

        if band_count <= 0:
            band_count = default_bands  # fallback for Dual (10 bands)

        formulas = [f"B{i}/32768" for i in range(1, band_count + 1)]

        rt = chunk.raster_transform
        rt.reset()
        rt.enabled = True
        rt.formula = formulas
        chunk.raster_transform = rt

        print(f"    [INFO] Raster transform set (Bn/32768) for {band_count} bands.")
        return True

    except Exception as e:
        print(f"    [WARN] Failed to set MicaSense raster transform: {e}")
        return False

def apply_micasense_raster_transform(chunk: Metashape.Chunk, default_bands: int = 10) -> bool:
    """
    For MicaSense multispectral orthomosaic:
    set Raster Transform so each band is scaled as Bn/32768.

    Tries to detect the actual number of bands from the orthomosaic or
    multi-camera system; falls back to default_bands (10 for Dual).
    """
    try:
        band_count = 0

        # 1) If orthomosaic already exists, use its band list
        ortho = getattr(chunk, "orthomosaic", None)
        if ortho is not None and hasattr(ortho, "bands") and ortho.bands:
            band_count = len(ortho.bands)

        # 2) If no orthomosaic or bands info, try multi-camera structure
        if band_count == 0:
            # Look for a master sensor with planes (multi-camera system)
            for sensor in chunk.sensors:
                planes = getattr(sensor, "planes", None)
                if planes:
                    band_count = max(band_count, len(planes))

            # If planes were not informative but we have multiple sensors, use that
            if band_count == 0 and len(chunk.sensors) > 1:
                band_count = len(chunk.sensors)

        # 3) Fallback: use default (e.g. 10 bands for MicaSense Dual)
        if band_count <= 0:
            band_count = default_bands

        # Build formulas: B1/32768, B2/32768, ... , Bn/32768
        formulas = [f"B{i}/32768" for i in range(1, band_count + 1)]

        # RasterTransform.formula is a STRING in 2.2, so join with commas
        rt = chunk.raster_transform
        rt.reset()
        rt.enabled = True
        rt.formula = formulas
        chunk.raster_transform = rt

    except Exception as e:
        print(f"    [WARN] Failed to set MicaSense raster transform: {e}")
        return False
    
def make_utm37n_projection() -> Metashape.OrthoProjection:
    """
    WGS 84 / UTM zone 37N (EPSG:32637) projection for DEM & orthos.
    """
    proj = Metashape.OrthoProjection()
    proj.crs = Metashape.CoordinateSystem("EPSG::32637")  # WGS 84 / UTM 37N
    return proj

def make_big_tiff_compression() -> Metashape.ImageCompression:
    """
    Create an ImageCompression object configured for BigTIFF.
    """
    comp = Metashape.ImageCompression()
    # Enable BigTIFF so we don't hit "Maximum TIFF file size exceeded"
    comp.tiff_big = True
    comp.tiff_overviews = True
    # You can choose compression if you want (LZW is a safe choice)
    # comp.tiff_compression = Metashape.ImageCompression.TiffCompressionLZW
    return comp
def chunk_has_ms_products(chunk) -> bool:
    """
    For multispectral (DualCamera), consider 'products' to be:
      - DEM / elevation
      - Orthomosaic

    Returns True if either exists in the chunk.
    """
    # DEM / elevation
    elev = getattr(chunk, "elevation", None)
    if elev is not None:
        return True

    # Orthomosaic
    ortho = getattr(chunk, "orthomosaic", None)
    if ortho is not None:
        return True

    return False

def build_export_tasks_network(chunk, site_outputs_dir: str, base: str, sensor_key: str):
    """
    Build ExportReport + ExportRaster tasks (DEM + Orthomosaic) for network processing.

    - Outputs go into site_outputs_dir.
    - base is like: Site2_Dualcamera_Sep2025
    - sensor_key: "l1", "mavic3e", "dual" (to decide about MS raster transform)

    Returns (tasks, paths_dict) where:
      tasks      = list of Metashape.Tasks.* objects (to pass to submit_network_tasks)
      paths_dict = {"report": path, "dem": path or None, "ortho": path or None}
    """
    tasks = []

    report_path = os.path.join(site_outputs_dir, f"{base}_report.pdf")
    dem_path    = os.path.join(site_outputs_dir, f"{base}_DEM.tif")
    ortho_path  = os.path.join(site_outputs_dir, f"{base}_Orthomosaic.tif")

    paths = {
        "report": report_path,
        "dem": dem_path,
        "ortho": ortho_path,
    }
    ortho_proj = make_utm37n_projection()
    compression = make_big_tiff_compression()
    # --- Report (ExportReport task) ---
    if os.path.exists(report_path):
        print(f"    [SKIP] Report already exists: {os.path.basename(report_path)}")
    else:
        t = Metashape.Tasks.ExportReport()
        t.path = report_path
        t.title = base
        t.description = f"Automatic report for {base}"
        # other options (font_size, page_numbers, include_system_info) left default
        tasks.append(t)
        print("    [TASK] Added ExportReport task.")

    # --- DEM (ExportRaster, ElevationData) ---
    has_dem = getattr(chunk, "elevation", None) is not None
    if not has_dem:
        print("    [WARN] No DEM present in chunk; skipping DEM export task.")
    elif os.path.exists(dem_path):
        print(f"    [SKIP] DEM already exists: {os.path.basename(dem_path)}")
    else:
        t = Metashape.Tasks.ExportRaster()
        t.path = dem_path
        t.source_data = Metashape.ElevationData
        t.image_format = Metashape.ImageFormatTIFF
        t.raster_transform = Metashape.RasterTransformNone
        t.projection = ortho_proj                      # << WGS84 / UTM 37N
        t.save_alpha = False                           # << no alpha
        t.white_background = False                     # << no white bg
        t.split_in_blocks = False                      # single file
        t.image_compression = compression
        tasks.append(t)
        print("    [TASK] Added DEM ExportRaster task.")

    # --- Orthomosaic (ExportRaster, OrthomosaicData) ---
    has_ortho = getattr(chunk, "orthomosaic", None) is not None
    if not has_ortho:
        print("    [WARN] No orthomosaic present; skipping orthomosaic export task.")
    elif os.path.exists(ortho_path):
        print(f"    [SKIP] Orthomosaic already exists: {os.path.basename(ortho_path)}")
    else:
        t = Metashape.Tasks.ExportRaster()
        t.path = ortho_path
        t.source_data = Metashape.OrthomosaicData
        t.image_format = Metashape.ImageFormatTIFF
        # For Dual (multispectral) we will set chunk.raster_transform = Bn/32768
        # and use RasterTransformValue here
        if sensor_key == "dual":
            t.raster_transform = Metashape.RasterTransformValue
        else:
            t.raster_transform = Metashape.RasterTransformNone
        t.projection = ortho_proj                      # << WGS84 / UTM 37N
        t.save_alpha = False                           # << no alpha
        t.white_background = False                     # << no white bg
        t.split_in_blocks = False                      # single file
        t.image_compression = compression
        tasks.append(t)
        print("    [TASK] Added Orthomosaic ExportRaster task.")

    return tasks, paths



# ----------------------------
# Step 1 – create project files
# ----------------------------

def create_projects(uav_root: str, campaign_suffix: str, site_filter: str = None):
    """
    Create processing/<Site>/Site_<Sensor>_<campaign_suffix>.psx
    for each sensor folder present in each site.

    If site_filter is not None, only that site folder name is processed.

    Idempotent:
      - If a project file already exists, it will be skipped.
    """
    processing_root = os.path.join(uav_root, "processing")
    os.makedirs(processing_root, exist_ok=True)

    created = 0
    skipped = 0

    print("=== STEP 1: Creating project files ===")
    print(f"UAV root: {uav_root}")
    print(f"Campaign suffix: {campaign_suffix}")
    if site_filter:
        print(f"(Single-site debug mode: only '{site_filter}' will be processed.)")
    print("")

    for entry in sorted(os.listdir(uav_root)):
        site_path = os.path.join(uav_root, entry)
        if not os.path.isdir(site_path):
            continue
        if entry.lower() == "processing":
            continue
        if site_filter and entry != site_filter:
            continue

        site_name = entry
        sensors = detect_sensor_folders(site_path)

        if not sensors:
            print(f"[WARN] {site_name}: no sensor folders found, skipping.")
            continue

        site_processing_dir = os.path.join(processing_root, site_name)
        os.makedirs(site_processing_dir, exist_ok=True)

        for key, info in sensors.items():
            project_suffix = info["project_suffix"]
            project_name = f"{site_name}_{project_suffix}_{campaign_suffix}.psx"
            project_path = os.path.join(site_processing_dir, project_name)

            if os.path.exists(project_path):
                print(f"[SKIP] Project already exists: {project_path}")
                skipped += 1
                continue

            # Create empty project with a chunk ready to receive photos
            doc = Metashape.Document()
            chunk = doc.addChunk()
            chunk.label = f"{site_name}_{project_suffix}"
            doc.save(project_path)

            created += 1
            print(f"[OK] Created project: {project_path}")

    print(f"=== STEP 1 Done === (created: {created}, skipped existing: {skipped})\n")


# ----------------------------
# Step 2 – collect images
# ----------------------------

def collect_l1_images(l1_root: str):
    """
    L1: import ALL images from ALL subfolders.
    """
    images = []
    for dirpath, dirnames, filenames in os.walk(l1_root):
        for f in filenames:
            if is_image_file(f):
                images.append(os.path.join(dirpath, f))
    images.sort()
    return images


def collect_mavic3e_images(mavic_root: str):
    """
    Mavic3E:
      Only subfolders with at least 3 '_' in the folder name:
        e.g. DJI_202511291317_002_Site11Flight2  -> used
             DJI_202511291336_003                -> skipped
    """
    images = []

    with os.scandir(mavic_root) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            folder_name = entry.name
            underscore_count = folder_name.count("_")
            if underscore_count >= 3:
                # Mission folder – include all images inside
                for dirpath, dirnames, filenames in os.walk(entry.path):
                    for f in filenames:
                        if is_image_file(f):
                            images.append(os.path.join(dirpath, f))
            else:
                # Fun / non-mission folder – skip
                print(f"    [Mavic3E] Skipping folder (only {underscore_count} '_'): {folder_name}")

    images.sort()
    return images


def collect_dualcamera_images(dual_root: str):
    """
    DualCamera (MicaSense Dual or similar):

      Expect subfolders Red/ and Blue/, import all images in both,
      but only keep groups (shots) that have a complete set of bands.

      Assumes filenames like IMG_0600_1.tif ... IMG_0600_10.tif
      and uses everything before the last '_' as shot ID (e.g. 'IMG_0600').

      Any shot that does not have DUALCAM_EXPECTED_PLANES images
      will be skipped to avoid TIFF/libtiff errors.
    """
    all_paths = []

    for sub in os.listdir(dual_root):
        sub_path = os.path.join(dual_root, sub)
        if not os.path.isdir(sub_path):
            continue
        if sub.lower() in ["red", "blue"]:
            for dirpath, dirnames, filenames in os.walk(sub_path):
                for f in filenames:
                    if is_image_file(f):
                        all_paths.append(os.path.join(dirpath, f))

    if not all_paths:
        return []

    # Group by shot_id (prefix before last underscore)
    groups = {}
    for path in all_paths:
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        parts = name.split("_")
        if len(parts) >= 2:
            shot_id = "_".join(parts[:-1])
        else:
            shot_id = name
        groups.setdefault(shot_id, []).append(path)

    complete_groups = {}
    incomplete_groups = {}

    for shot_id, paths in groups.items():
        if len(paths) == DUALCAM_EXPECTED_PLANES:
            complete_groups[shot_id] = sorted(paths)
        else:
            incomplete_groups[shot_id] = sorted(paths)

    if incomplete_groups:
        print("    [DualCamera] WARNING: some shots have missing or extra bands and will be skipped.")
        for shot_id, paths in sorted(incomplete_groups.items()):
            print(f"        Shot '{shot_id}' has {len(paths)} files (expected {DUALCAM_EXPECTED_PLANES}).")

    # Flatten complete groups into a single ordered list:
    ordered_images = []
    for shot_id in sorted(complete_groups.keys()):
        ordered_images.extend(complete_groups[shot_id])

    print(f"    [DualCamera] Found {len(complete_groups)} complete shots "
          f"({len(ordered_images)} images, {DUALCAM_EXPECTED_PLANES} bands each).")

    return ordered_images


# ----------------------------
# Step 2 – import images into projects
# ----------------------------

def open_project_and_chunk(project_path: str, chunk_label: str):
    """
    Open a Metashape project, ensure it has one chunk and return (doc, chunk).
    If there are no chunks yet, create one.
    """
    doc = Metashape.Document()
    doc.open(project_path)

    if doc.chunks:
        chunk = doc.chunk  # active chunk
    else:
        chunk = doc.addChunk()
        chunk.label = chunk_label

    return doc, chunk


def chunk_has_alignment(chunk) -> bool:
    """
    Check if any camera in the chunk has a non-empty transform
    (typical indicator that alignment has been performed).
    """
    try:
        for cam in chunk.cameras:
            if getattr(cam, "transform", None) is not None:
                return True
    except Exception:
        pass
    return False


def chunk_has_rgb_products(chunk) -> bool:
    """
    Check if RGB project already has some processing done:
      - alignment
      - dense point cloud
      - DEM / elevation
      - orthomosaic
    """
    if chunk_has_alignment(chunk):
        return True

    # Dense point cloud
    try:
        pc = getattr(chunk, "point_cloud", None)
        if pc is not None and hasattr(pc, "points") and len(pc.points) > 0:
            return True
    except Exception:
        pass

    # DEM / elevation(s)
    for attr in ("elevation", "elevations", "dem"):
        if hasattr(chunk, attr):
            val = getattr(chunk, attr)
            try:
                if val:
                    return True
            except Exception:
                if val is not None:
                    return True

    # Orthomosaic(s)
    for attr in ("orthomosaic", "orthomosaics"):
        if hasattr(chunk, attr):
            val = getattr(chunk, attr)
            try:
                if val:
                    return True
            except Exception:
                if val is not None:
                    return True

    return False


def has_calibration_images_group(chunk) -> bool:
    """
    Check if there is a camera group named 'Calibration images'
    by inspecting cameras and their assigned groups.

    Returns True if ANY camera belongs to a group whose label
    normalizes to 'calibrationimages'.

    If debug is True, prints the group labels it finds.
    """
    labels_seen = set()
    calib_found = False

    try:
        cams = chunk.cameras
    except AttributeError:
        return False

    for cam in cams:
        grp = getattr(cam, "group", None)
        if not grp:
            continue

        label = (getattr(grp, "label", "") or "").strip()
        labels_seen.add(label)

        norm = label.replace(" ", "").lower()
        if norm == "calibrationimages":
            calib_found = True

    return calib_found

def import_all_images(uav_root: str, campaign_suffix: str, site_filter: str = None):
    """
    Step 2 – import images into each project.

    Idempotent:
      - If a chunk already has cameras, import is skipped for that sensor.
      - For DualCamera, we still (re)set primary_channel to NIR (band 10, index 9),
        which is safe to run multiple times.
    """
    processing_root = os.path.join(uav_root, "processing")

    print("=== STEP 2: Importing images into projects ===")
    if site_filter:
        print(f"(Single-site debug mode: only '{site_filter}' will be processed.)")

    imported_chunks = 0
    skipped_chunks = 0

    for entry in sorted(os.listdir(uav_root)):
        site_path = os.path.join(uav_root, entry)
        if not os.path.isdir(site_path):
            continue
        if entry.lower() == "processing":
            continue
        if site_filter and entry != site_filter:
            continue

        site_name = entry
        sensors = detect_sensor_folders(site_path)
        if not sensors:
            continue

        site_processing_dir = os.path.join(processing_root, site_name)

        print(f"\n--- Site: {site_name} ---")

        for key, info in sensors.items():
            sensor_folder = info["folder_path"]
            project_suffix = info["project_suffix"]
            project_name = f"{site_name}_{project_suffix}_{campaign_suffix}.psx"
            project_path = os.path.join(site_processing_dir, project_name)

            if not os.path.exists(project_path):
                print(f"  [WARN] Project not found for {key}: {project_path}")
                continue

            print(f"  Sensor: {key} ({sensor_folder})")
            chunk_label = f"{site_name}_{project_suffix}"
            doc, chunk = open_project_and_chunk(project_path, chunk_label)

            # === Idempotency: if we already have cameras, skip re-import ===
            if len(chunk.cameras) > 0:
                print(f"    [SKIP] Chunk already has {len(chunk.cameras)} cameras (images already imported).")

                if key == "dual":
                    # Even on re-run, keep primary channel consistent (NIR band 10).
                    try:
                        chunk.primary_channel = 9
                        print("    [DualCamera] Primary channel set to index 9 (Channel 10 - NIR).")
                    except Exception as e:
                        print(f"    [WARN] Failed to set primary channel on existing chunk: {e}")

                doc.save()
                skipped_chunks += 1
                continue

            # === No cameras yet: do the actual import ===
            if key == "l1":
                images = collect_l1_images(sensor_folder)
                if not images:
                    print("    [WARN] No L1 images found.")
                    doc.save()
                    continue

                print(f"    Adding {len(images)} L1 images...")
                chunk.addPhotos(filenames=images)

            elif key == "mavic3e":
                images = collect_mavic3e_images(sensor_folder)
                if not images:
                    print("    [WARN] No Mavic3E mission images found.")
                    doc.save()
                    continue

                print(f"    Adding {len(images)} Mavic3E images (mission folders only)...")
                chunk.addPhotos(filenames=images)

            elif key == "dual":
                images = collect_dualcamera_images(sensor_folder)
                if not images:
                    print("    [WARN] No valid DualCamera images found (Red/Blue).")
                    doc.save()
                    continue

                print(f"    Adding {len(images)} DualCamera images as MULTISPECTRAL (multi-plane layout)...")
                chunk.addPhotos(
                    filenames=images,
                    layout=Metashape.MultiplaneLayout
                )

                # Set primary channel to band 10 (index 9) = NIR
                try:
                    chunk.primary_channel = 9
                    print("    [DualCamera] Primary channel set to index 9 (Channel 10 - NIR).")
                except Exception as e:
                    print(f"    [WARN] Failed to set primary channel: {e}")

            else:
                print(f"    [WARN] Unknown sensor key: {key}")
                doc.save()
                continue

            # Save project after importing (and setting primary channel if dual)
            doc.save()
            imported_chunks += 1
            print(f"    [OK] Saved project: {project_path}")

    print(f"\n=== STEP 2 Done === (imported chunks: {imported_chunks}, skipped (already imported): {skipped_chunks})")


# ----------------------------
# Step 3 – Network batch submission
# ----------------------------

def run_network_batches(uav_root: str, campaign_suffix: str, site_filter: str = None):
    """
    Step 3: connect to the network server, check current batches,
    and submit new batches based on project state.

      * RGB (L1, Mavic3E):
          - If server already has an active batch for the project -> skip.
          - Else, if the chunk already has alignment / point cloud / DEM / orthomosaic -> skip.
          - Else, submit RGB task pipeline (build_rgb_tasks()).

      * Multispectral (DualCamera):
          - If server already has an active batch for the project -> skip.
          - Else:
              - If NO alignment -> submit build_multi_align_tasks().
              - If alignment exists AND DEM/orthomosaic already exist -> skip (products done).
              - If alignment exists, no DEM/orthomosaic yet:
                    - If 'Calibration images' group exists and panel CSV loads OK
                      -> submit build_multi_products_tasks().
                    - Otherwise -> skip (calibration not ready / missing panel file).
    """
    print("\n=== STEP 3: Network batch submission ===")
    client, cfg = get_connected_client(uav_root)
    if not client:
        print("  [INFO] Network server not available or configuration invalid, skipping batch submission.")
        return

    processing_root = os.path.join(uav_root, "processing")

    for entry in sorted(os.listdir(uav_root)):
        site_path = os.path.join(uav_root, entry)
        if not os.path.isdir(site_path):
            continue
        if entry.lower() == "processing":
            continue
        if site_filter and entry != site_filter:
            continue

        site_name = entry
        sensors = detect_sensor_folders(site_path)
        if not sensors:
            continue

        site_processing_dir = os.path.join(processing_root, site_name)
        if not os.path.isdir(site_processing_dir):
            print(f"[WARN] {site_name}: processing folder missing, skipping network batches.")
            continue

        print(f"\n--- Network batches for Site: {site_name} ---")

        for key, info in sensors.items():
            project_suffix = info["project_suffix"]
            sensor_folder = info["folder_path"]

            project_name = f"{site_name}_{project_suffix}_{campaign_suffix}.psx"
            project_path = os.path.join(site_processing_dir, project_name)

            if not os.path.exists(project_path):
                print(f"  [WARN] Project not found ({key}): {project_path}")
                continue

            print(f"  Sensor: {key} -> project: {project_name}")

            # Check if server already has an active batch for this project
            if project_has_active_batch(client, project_path):
                print("    [SKIP] Server already has active processing for this project.")
                continue

            # Open project to inspect current state
            doc = Metashape.Document()
            doc.open(project_path)
            if not doc.chunks:
                print("    [WARN] Project has no chunks, skipping.")
                continue
            chunk = doc.chunks[0]

            # ------------ RGB (L1, Mavic3E) ------------
            if key in ("l1", "mavic3e"):
                if chunk_has_rgb_products(chunk):
                    print("    [SKIP] RGB project already has alignment / point cloud / DEM / orthomosaic.")
                else:
                    tasks = build_rgb_tasks()
                    submit_network_tasks(
                        client,
                        project_path,
                        tasks,
                        description=f"RGB {site_name} {project_suffix}",
                    )

            # ------------ Multispectral (DualCamera) ------------
            elif key == "dual":
                aligned = chunk_has_alignment(chunk)
                ms_products_done = chunk_has_ms_products(chunk)
                calib_group_exists = has_calibration_images_group(chunk)

                print(
                    f"    [INFO] DualCamera status: "
                    f"aligned={aligned}, products_done={ms_products_done}, "
                    f"calibration_group={calib_group_exists}"
                )

                # 1) No alignment yet -> run alignment batch only
                if not aligned:
                    tasks = build_multi_align_tasks()
                    submit_network_tasks(
                        client,
                        project_path,
                        tasks,
                        description=f"MS Align {site_name}",
                    )

                else:
                    # 2) Alignment exists
                    if ms_products_done:
                        # DEM and/or orthomosaic already there -> nothing to do
                        print("    [SKIP] Multispectral products already exist (DEM and/or orthomosaic).")
                    else:
                        # 3) No products yet, but alignment is done
                        if not calib_group_exists:
                            print("    [INFO] Alignment exists but no 'Calibration images' group; "
                                  "skipping multispectral products batch for now.")
                        else:
                            # Calibration group exists -> try to load panel CSV
                            loaded = load_panel_calibration(
                                doc,
                                chunk,
                                uav_root,
                                site_name,
                                project_suffix=project_suffix,
                            )

                            if not loaded:
                                print("    [WARN] Panel calibration file not loaded; "
                                      "skipping multispectral products batch.")
                            else:
                                tasks = build_multi_products_tasks()
                                submit_network_tasks(
                                    client,
                                    project_path,
                                    tasks,
                                    description=f"MS Products {site_name}",
                                )

            else:
                print(f"    [INFO] No network batch defined for sensor key '{key}'.")

    print("\n=== STEP 3 Done (network batch submission) ===")


def export_outputs_network(uav_root: str, campaign_suffix: str, site_filter: str = None):
    """
    STEP 4: Submit ExportReport + ExportRaster (DEM, Orthomosaic) as NETWORK tasks.

    Structure on disk:
      UAV/
        Outputs/
          Site2/
            Site2_L1_Sep2025_DEM.tif
            Site2_L1_Sep2025_Orthomosaic.tif
            Site2_L1_Sep2025_report.pdf
            Site2_Dualcamera_Sep2025_DEM.tif
            Site2_Dualcamera_Sep2025_Orthomosaic.tif
            Site2_Dualcamera_Sep2025_report.pdf
            ...

    - For RGB (L1, Mavic3E): normal exports.
    - For Dual (multispectral): before creating orthomosaic task, set
      Bn/32768 raster transform on the chunk, save project, then use
      RasterTransformValue in ExportRaster task.

    - Idempotent: if an output file already exists, no task is created.
    - If project still has ACTIVE network batches (processing), we skip
      exports now; run the script again after processing finishes.
    """
    client, cfg = get_connected_client(uav_root)
    if not client:
        print("\n[INFO] Network server not available, skipping network exports (step 4).")
        return

    processing_root = os.path.join(uav_root, "processing")
    outputs_root = os.path.join(uav_root, "Outputs")
    os.makedirs(outputs_root, exist_ok=True)

    print("\n=== STEP 4: Submitting export tasks to network ===")

    for entry in sorted(os.listdir(uav_root)):
        site_path = os.path.join(uav_root, entry)
        if not os.path.isdir(site_path):
            continue

        low = entry.lower()
        if low in ("processing", "outputs"):
            continue

        if site_filter and entry != site_filter:
            continue

        site_name = entry
        sensors = detect_sensor_folders(site_path)
        if not sensors:
            continue

        site_processing_dir = os.path.join(processing_root, site_name)
        if not os.path.isdir(site_processing_dir):
            print(f"[WARN] {site_name}: processing folder missing, skipping exports.")
            continue

        site_outputs_dir = os.path.join(outputs_root, site_name)
        os.makedirs(site_outputs_dir, exist_ok=True)

        print(f"\n--- Site: {site_name} ---")

        for key, info in sensors.items():
            project_suffix = info["project_suffix"]
            project_name = f"{site_name}_{project_suffix}_{campaign_suffix}.psx"
            project_path = os.path.join(site_processing_dir, project_name)

            if not os.path.exists(project_path):
                print(f"  [WARN] Project not found for {key}: {project_path}")
                continue

            print(f"  Sensor: {key} -> project: {project_name}")

            # Do not queue exports while processing batches are ACTIVE
            if project_has_active_batch(client, project_path):
                print("    [SKIP] Project still has active batches on server; "
                      "skipping export tasks for now.")
                continue

            # Open project to inspect chunk (DEM/ortho presence) and, for Dual,
            # apply MicaSense raster transform (Bn/32768)
            doc = Metashape.Document()
            doc.open(project_path)
            if not doc.chunks:
                print("    [WARN] Project has no chunks, skipping.")
                continue
            chunk = doc.chunks[0]

            base = f"{site_name}_{project_suffix}_{campaign_suffix}"

            # For Dual (multispectral) set the Bn/32768 transform BEFORE creating tasks
            if key == "dual":
                if apply_micasense_raster_transform(chunk):
                    doc.save()

            tasks, paths = build_export_tasks_network(chunk, site_outputs_dir, base, key)

            if not tasks:
                print("    [INFO] No export tasks needed (all outputs exist or no DEM/ortho).")
                doc.save()
                continue

            # Submit export tasks as a separate network batch
            submit_network_tasks(
                client,
                project_path,
                tasks,
                description=f"Export outputs {base}"
            )

            doc.save()
            print(f"    [OK] Submitted {len(tasks)} export tasks to network for {base}.")

    print("\n=== STEP 4 Done (network exports submitted) ===")



# ----------------------------
# Main helpers
# ----------------------------

def get_uav_root_from_script():
    """
    The script is inside UAV folder as you described.
    """
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    else:
        return os.getcwd()


def main():
    uav_root = get_uav_root_from_script()
    campaign_suffix = get_campaign_suffix(uav_root)

    # If you want to debug only one site, set TEST_SITE_NAME above.
    site_filter = TEST_SITE_NAME

    # Step 1: create .psx files
    create_projects(uav_root, campaign_suffix, site_filter=site_filter)

    # Step 2: import images into each project, and set DualCamera primary channel to NIR (channel 10)
    import_all_images(uav_root, campaign_suffix, site_filter=site_filter)

    # Step 3: configure & test network processing and submit appropriate batches
    run_network_batches(uav_root, campaign_suffix, site_filter=site_filter)

    # Step 4: Export
    export_outputs_network(uav_root, campaign_suffix, site_filter=site_filter)



if __name__ == "__main__":
    main()
