"""Lettura DICOM -> unita' Hounsfield, con i metadati che servono agli assi
di stratificazione.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom


def read_hu(path: str | Path, with_pixels: bool = True):
    """Legge una slice.  Ritorna (hu, padding_value, meta).

    `hu` e' float32 in unita' Hounsfield.  `padding_value` e' il valore di
    riempimento fuori campo dichiarato dal costruttore, o None.
    """
    d = pydicom.dcmread(str(path), stop_before_pixels=not with_pixels)
    pad = getattr(d, "PixelPaddingValue", None)
    slope = float(getattr(d, "RescaleSlope", 1.0))
    inter = float(getattr(d, "RescaleIntercept", 0.0))
    if pad is not None:
        # PixelPaddingValue e' in valori memorizzati, non in HU
        pad = float(pad) * slope + inter

    kern = getattr(d, "ConvolutionKernel", None)
    if kern is not None and not isinstance(kern, str):
        kern = "/".join(str(k) for k in kern)
    ps = getattr(d, "PixelSpacing", [np.nan, np.nan])

    meta = {
        "patient_id": str(getattr(d, "PatientID", "")),
        "series_uid": str(getattr(d, "SeriesInstanceUID", "")),
        "sop_uid": str(getattr(d, "SOPInstanceUID", "")),
        "series_description": str(getattr(d, "SeriesDescription", "")),
        "instance_number": int(getattr(d, "InstanceNumber", -1)),
        "z": float(getattr(d, "ImagePositionPatient", [0, 0, np.nan])[2]),
        "manufacturer": str(getattr(d, "Manufacturer", "")).split()[0].upper()
        if getattr(d, "Manufacturer", "")
        else "",
        "model": str(getattr(d, "ManufacturerModelName", "")),
        "body_part": str(getattr(d, "BodyPartExamined", "")),
        "kernel": str(kern) if kern is not None else "",
        "slice_thickness": float(getattr(d, "SliceThickness", np.nan)),
        "pixel_spacing": float(ps[0]),
        "kvp": float(getattr(d, "KVP", np.nan)),
        "recon_diameter": float(getattr(d, "ReconstructionDiameter", np.nan)),
        "rescale_slope": slope,
        "rescale_intercept": inter,
        "padding_value_hu": pad,
        "rows": int(getattr(d, "Rows", 0)),
        "cols": int(getattr(d, "Columns", 0)),
    }
    if not with_pixels:
        return None, pad, meta
    hu = d.pixel_array.astype(np.float32) * slope + inter
    return hu, pad, meta


def cell_of(meta: dict) -> str:
    """Cella di protocollo: costruttore | regione | kernel | spessore."""
    return (
        f"{meta['manufacturer']}|{meta['body_part']}|{meta['kernel']}|"
        f"{meta['slice_thickness']:g}"
    )
