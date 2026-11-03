"""
ホッチキス留めフック + 引っ掛け式モールディング（サンプル v3）
  - 45° くさび（クリート）を上下2段 → フックの縦幅全体でモールディングを支える
  - 下端にスナップ式のツメ → 一度はまると持ち上げても外れない

座標系: x = 壁に沿った横方向, y = 壁から手前方向 (壁面 y=0), z = 上方向 [mm]
使い方: python3 molding_hook.py  → hook.stl / molding.stl / assembly.stl
"""
import numpy as np
import manifold3d as m3d
import trimesh

# ---------------- ホッチキス針（10号） ----------------
STAPLE_CROWN = 8.5      # 針の外幅
LEG_HOLE_W   = 1.3      # 脚用ガイド穴（横）
LEG_HOLE_H   = 1.8      # 脚用ガイド穴（縦）
STAPLE_TILT  = 15       # ガイド穴の傾き [deg]（下向きに斜め）

# ---------------- フック ----------------
W        = 60.0   # 幅
T        = 4.0    # 枠・リブ部の厚み
WEB      = 1.2    # 針を打つ帯の厚み
BAND_H   = 18.0   # 針打ち帯の高さ（手持ちのホッチキスのヘッド幅 + 2mm 程度に）
CLEAT_D  = 8.0    # くさびの出幅（= 爪の掛かり深さ）
CLEAT_H  = 16.0   # くさびの高さ
BOT_RIB  = 4.0    # 最下部リブの高さ（ここにツメの受けがある）
STAPLE_X = [-20.0, -7.0, 7.0, 20.0]   # 各帯の針位置 → 2帯 × 4 = 8本

# 下から: 下リブ / 帯1 / くさび1 / 帯2 / くさび2
BAND1   = (BOT_RIB, BOT_RIB + BAND_H)
CLEAT1  = BAND1[1] + 4.0   # 帯1とくさび1の間に4mmの補強リブ
BAND2   = (CLEAT1 + CLEAT_H, CLEAT1 + CLEAT_H + BAND_H)
CLEAT2  = BAND2[1]
H       = CLEAT2 + CLEAT_H
BANDS   = [BAND1, BAND2]
CLEATS  = [CLEAT1, CLEAT2]

# ツメの受け（下リブ前面の小さな出っ張り）
RIDGE_R  = 1.5    # 出っ張り量
RIDGE_H  = 3.0    # 高さ（上側は45°面取り）

# ---------------- モールディング ----------------
M_LEN    = 200.0  # 長さ（サンプル）
M_D      = 30.0   # 奥行き（フック表面から手前）
M_BOTTOM = -8.0   # 下端 z
M_TOP    = H + 6  # 上端 z（上面にテープライト）
CLEAR    = 0.3    # すき間
LIFT     = CLEAT_D + 1.0   # 取り付け時に持ち上げる量
FLAP_T   = 1.6    # ツメ板の厚み（しなる部分）
SLOT_W   = 1.6    # ツメ板の後ろのスリット幅（ここにマイナスドライバーを入れて外す）
FLAP_TOP = 12.0   # スリット上端 z（ツメ板の長さを決める）


def to_manifold_x(cs, width):
    """CrossSection (y,z) を x 方向に押し出して (x,y,z) の Manifold にする"""
    solid = m3d.Manifold.extrude(cs, width).translate([0, 0, -width / 2])
    mesh = solid.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    v2 = np.ascontiguousarray(np.stack([v[:, 2], v[:, 0], v[:, 1]], axis=1), dtype=np.float32)
    tris = np.ascontiguousarray(np.asarray(mesh.tri_verts), dtype=np.uint32)
    return m3d.Manifold(m3d.Mesh(vert_properties=v2, tri_verts=tris))


def poly(pts):
    cs = m3d.CrossSection([pts])
    return cs if cs.area() > 0 else m3d.CrossSection([pts[::-1]])


def cleat_cs(z0):
    # 下面: 45°上り（サポート不要） / 上面: 壁側へ45°下り（爪が掛かる）
    return poly([(T - 0.01, z0), (T + CLEAT_D, z0 + CLEAT_D),
                 (T + CLEAT_D, z0 + CLEAT_H), (T - 0.01, z0 + CLEAT_H - CLEAT_D)])


def ridge_cs():
    return poly([(T - 0.01, 0.0), (T + RIDGE_R, 0.0), (T + RIDGE_R, RIDGE_H - RIDGE_R),
                 (T - 0.01, RIDGE_H)])


def box(x0, x1, y0, y1, z0, z1):
    return m3d.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def make_hook(with_ridge=True):
    body = box(-W / 2, W / 2, 0, T, 0, H)
    for z0, z1 in BANDS:
        body = body - box(-W / 2 - 1, W / 2 + 1, WEB, T + 1, z0, z1)
    for z0 in CLEATS:
        body = body + to_manifold_x(cleat_cs(z0), W)
    if with_ridge:
        body = body + to_manifold_x(ridge_cs(), W)
    for z0, z1 in BANDS:
        zc = (z0 + z1) / 2
        for xc in STAPLE_X:
            for s in (-1, 1):
                xh = xc + s * (STAPLE_CROWN - 0.6) / 2
                h = m3d.Manifold.cube([LEG_HOLE_W, 12, LEG_HOLE_H], center=True)
                body = body - h.rotate([-STAPLE_TILT, 0, 0]).translate([xh, 0, zc])
    return body


def make_molding():
    L = M_LEN + 2
    y_back = T + CLEAR
    cut = m3d.CrossSection()
    # くさびの逃げ：はまった位置 〜 LIFT だけ下（=持ち上げた時の相対位置）までの掃引
    for z0 in CLEATS:
        c = cleat_cs(z0).offset(CLEAR, m3d.JoinType.Miter)
        cut = cut + (c + c.translate([0, -LIFT])).hull()
    # 下部：出っ張り（受け）の逃げ。ツメ板の内面は出っ張りの前面に沿う
    y_flap_in = T + RIDGE_R + CLEAR
    cut = cut + poly([(0, M_BOTTOM - 1), (y_flap_in, M_BOTTOM - 1),
                      (y_flap_in, RIDGE_H + LIFT + 0.6), (0, RIDGE_H + LIFT + 0.6)])
    # ツメ板の後ろのスリット（ツメ板が手前にしなれるように）
    ys = y_flap_in + FLAP_T
    cut = cut + poly([(ys, M_BOTTOM - 1), (ys + SLOT_W, M_BOTTOM - 1),
                      (ys + SLOT_W, FLAP_TOP), (ys, FLAP_TOP)])
    prof = poly([(y_back, M_BOTTOM), (T + M_D, M_BOTTOM),
                 (T + M_D, M_TOP), (y_back, M_TOP)]) - cut
    # ツメ（返し）：上面が出っ張りの下面に掛かる。下面は45°で、押し下げるとしなって乗り越える
    zt = -CLEAR
    yo = y_flap_in + 0.5
    barb = poly([(y_back, zt), (yo, zt), (yo, zt - 1.0 - (yo - y_back)), (y_back, zt - 1.0)])
    prof = prof + barb
    return to_manifold_x(prof, M_LEN)


def to_trimesh(man):
    mesh = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts))


if __name__ == "__main__":
    hook = make_hook()
    molding = make_molding()
    to_trimesh(hook).export("hook.stl")
    to_trimesh(molding).export("molding.stl")

    asm = make_molding()
    for xo in (-60, 60):
        asm = asm + hook.translate([xo, 0, 0])
    to_trimesh(asm).export("assembly.stl")

    print("hook height:", H, " molding height:", M_TOP - M_BOTTOM)
    print("seated interference:", round((hook ^ molding).volume(), 3))
    bare = make_hook(with_ridge=False)
    path = [(bare ^ molding.translate([0, 0, dz])).volume() for dz in np.linspace(0, LIFT, 19)]
    print("drop-in path interference (excluding snap):", round(max(path), 3))
    snap = [(hook ^ molding.translate([0, 0, dz])).volume() for dz in np.linspace(0, LIFT, 19)]
    print("snap overlap along path (= flap deflection region):", [round(s, 1) for s in snap])
    print("watertight:", to_trimesh(hook).is_watertight, to_trimesh(molding).is_watertight)
