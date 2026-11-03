MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
                    10, 11, 12, 19, 7, 8, 9, 20]


# ---------------------------------------------------------------------------
# OFFICIAL fingertip vertex ids, per convention.
#
# A dataset's 21-joint GT is MANO's 16 articulated joints plus 5 vertices read
# off the posed mesh. Which 5 vertices is a per-toolkit choice, so a dataset's
# numbers are only reproducible with ITS convention. The 16 articulated joints
# agree to 0.000 mm across conventions; only the tips move, by ~3.5 mm, which is
# ~1.16 mm of 21-joint PA-MPJPE.
#
#   manopth  (hassony2/manopth manolayer.py)
#       right: th_verts[:, [745, 317, 444, 556, 673]]
#       left : th_verts[:, [745, 317, 445, 556, 673]]   <- middle differs by hand
#   smplx    (smplx.vertex_ids['mano'])
#       both : [744, 320, 443, 554, 671]
#
# Order is [thumb, index, middle, ring, pinky], matching MANO_TO_OPENPOSE's
# expectation that entries 16..20 are the tips.
TIP_VERTEX_IDS = {
    ("ho3dv2_gt", "right"): [745, 320, 444, 555, 657],
    ("ho3dv2_gt", "left"): [745, 320, 444, 555, 657],
    # HO3D, from its OFFICIAL setup_mano.py, which patches the legacy MANO
    # webuser lbs.py with:
    #     for tp in [744, 333, 444, 555, 672]:  # THUMB, INDEX, MIDDLE, RING, PINKY
    # HO3D also patches MANO_LEFT.pkl by multiplying shapedirs by -1, i.e. it
    # expects the smplx issue-#48 left-shapedir fix APPLIED (DexYCB expects it
    # REVERTED -- see dexycb-decode-and-corruption).
    ("ho3d", "right"): [744, 333, 444, 555, 672],
    ("ho3d", "left"):  [744, 333, 444, 555, 672],
    # manopth, from hassony2/manopth manolayer.py. DexYCB's official toolkit
    # (dex-ycb-toolkit layers/mano_layer.py) wraps this layer directly.
    # NOTE the middle tip differs BY HAND: 444 right, 445 left.
    ("manopth", "right"): [745, 317, 444, 556, 673],
    ("manopth", "left"):  [745, 317, 445, 556, 673],
    # smplx.vertex_ids['mano'] -- what HaMeR / WiLoR / HaWoR read, so it is the
    # matching convention for hot3d, whose GT is generated with that readout.
    ("smplx", "right"):   [744, 320, 443, 554, 671],
    ("smplx", "left"):    [744, 320, 443, 554, 671],
}


SMPLX_MANO_RIGHT = None
SMPLX_MANO_LEFT = None
