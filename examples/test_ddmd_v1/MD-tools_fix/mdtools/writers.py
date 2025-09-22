import h5py
import numpy as np
from typing import List, Optional

AMINO_ACID_MAP = {
    "ALA": 1,
    "ARG": 2,
    "ASN": 3,
    "ASP": 4,
    "CYS": 5,
    "GLN": 6,
    "GLU": 7,
    "GLY": 8,
    "HIS": 9,
    "HIE": 9,  # HIS, HIE, HID are the same
    "HID": 9,  # residue with different protonation
    "ILE": 10,
    "LEU": 11,
    "LYS": 12,
    "MET": 13,
    "PHE": 14,
    "PRO": 15,
    "SER": 16,
    "THR": 17,
    "TRP": 18,
    "TYR": 19,
    "VAL": 20,
}


def write_aminoacid_int_seq(h5_file: h5py.File, residues: List[str]):
    data = np.array([AMINO_ACID_MAP[r] for r in residues], dtype="int8")
    h5_file.create_dataset(
        "amino_acids", data=data, dtype="int8", 
        #fletcher32=True, 
        chunks=(1,)
    )


def write_contact_map(
    h5_file: h5py.File,
    rows: List[np.ndarray],
    cols: List[np.ndarray],
    vals: Optional[List[np.ndarray]] = None,
):

    # Helper function to create ragged array
    def ragged(data):
        a = np.empty(len(data), dtype=object)
        a[...] = data
        return a

    # list of np arrays of shape (2 * X) where X varies
    data = ragged([np.concatenate(row_col) for row_col in zip(rows, cols)])
    h5_file.create_dataset(
        "contact_map",
        data=data,
        dtype=h5py.vlen_dtype(np.dtype("int16")),
        #fletcher32=True,
        chunks=(1,) + data.shape[1:],
    )

    # Write optional values field for contact map. Could contain CA-distances.
    if vals is not None:
        data = ragged(vals)
        h5_file.create_dataset(
            "contact_map_values",
            data=data,
            dtype=h5py.vlen_dtype(np.dtype("float32")),
            #fletcher32=True,
            chunks=(1,) + data.shape[1:],
        )


def write_point_cloud(h5_file: h5py.File, point_cloud: np.ndarray):
    h5_file.create_dataset(
        "point_cloud",
        data=point_cloud,
        dtype="float32",
        #fletcher32=True,
        chunks=(1,) + point_cloud.shape[1:],
    )


def write_rmsd(h5_file: h5py.File, rmsd):
    h5_file.create_dataset(
        "rmsd", data=rmsd, dtype="float16", fletcher32=True, chunks=(1,)
    )


def write_fraction_of_contacts(h5_file: h5py.File, fnc):
    h5_file.create_dataset(
        "fnc", data=fnc, dtype="float16", fletcher32=True, chunks=(1,)
    )
